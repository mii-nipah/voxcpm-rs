use cubecl_core::{
    ir::{
        Arithmetic, BinaryOperator, Bitwise, ClampOperator, Comparison, ElemType, ExpandElement,
        FloatKind, FmaOperator, Instruction, IntKind, Operation, Operator, Plane, Scope, Select,
        StorageType, Type, UIntKind, UnaryOperator, Variable,
    },
    prelude::{IntExpand, assign, expand_erf, expand_hypot, expand_rhypot},
};
use cubecl_opt::{IrTransformer, TransformAction};

use crate::bitwise::{small_int_reverse, u64_count_bits, u64_ffs, u64_leading_zeros, u64_reverse};

/// Expand erf
#[derive(Debug)]
pub(crate) struct ErfTransform;

impl IrTransformer for ErfTransform {
    fn maybe_transform(&self, scope: &mut Scope, inst: &Instruction) -> TransformAction {
        match &inst.operation {
            Operation::Arithmetic(Arithmetic::Erf(op)) => {
                let mut scope = scope.child();
                expand_erf(&mut scope, op.input, inst.out.unwrap());
                TransformAction::Replace(into_instructions(scope))
            }
            _ => TransformAction::Ignore,
        }
    }
}

/// Expand hypot
#[derive(Debug)]
pub(crate) struct HypotTransform;

impl IrTransformer for HypotTransform {
    fn maybe_transform(&self, scope: &mut Scope, inst: &Instruction) -> TransformAction {
        match &inst.operation {
            Operation::Arithmetic(Arithmetic::Hypot(op)) => {
                let mut scope = scope.child();
                expand_hypot(&mut scope, op.lhs, op.rhs, inst.out.unwrap());
                TransformAction::Replace(into_instructions(scope))
            }
            _ => TransformAction::Ignore,
        }
    }
}

/// Expand hypot
#[derive(Debug)]
pub(crate) struct RhypotTransform;

impl IrTransformer for RhypotTransform {
    fn maybe_transform(&self, scope: &mut Scope, inst: &Instruction) -> TransformAction {
        match &inst.operation {
            Operation::Arithmetic(Arithmetic::Rhypot(op)) => {
                let mut scope = scope.child();
                expand_rhypot(&mut scope, op.lhs, op.rhs, inst.out.unwrap());
                TransformAction::Replace(into_instructions(scope))
            }
            _ => TransformAction::Ignore,
        }
    }
}

/// Transform operations that only support 32 bits using polyfills
#[derive(Debug)]
pub(crate) struct BitwiseTransform;

impl IrTransformer for BitwiseTransform {
    fn maybe_transform(&self, scope: &mut Scope, inst: &Instruction) -> TransformAction {
        let op = match &inst.operation {
            Operation::Bitwise(op) => op,
            _ => return TransformAction::Ignore,
        };
        match op {
            Bitwise::CountOnes(op) if is_u64(op.input) => {
                let mut scope = scope.child();
                scope.register_type::<IntExpand<0>>(op.input.storage_type());
                let res = u64_count_bits::expand::<IntExpand<0>>(
                    &mut scope,
                    ExpandElement::Plain(op.input).into(),
                );
                assign::expand_no_check(&mut scope, res, ExpandElement::Plain(inst.out()).into());
                TransformAction::Replace(into_instructions(scope))
            }
            Bitwise::ReverseBits(op) if op.input.storage_type().size() != 4 => {
                let mut scope = scope.child();
                scope.register_type::<IntExpand<0>>(op.input.ty.storage_type());
                let input = ExpandElement::Plain(op.input);
                match op.input.storage_type().size() {
                    8 => {
                        let res = u64_reverse::expand::<IntExpand<0>>(&mut scope, input.into());
                        assign::expand_no_check(
                            &mut scope,
                            res,
                            ExpandElement::Plain(inst.out()).into(),
                        );
                        TransformAction::Replace(into_instructions(scope))
                    }
                    width => {
                        let res = small_int_reverse::expand::<IntExpand<0>>(
                            &mut scope,
                            input.into(),
                            width as u32 * 8,
                        );
                        assign::expand_no_check(
                            &mut scope,
                            res,
                            ExpandElement::Plain(inst.out()).into(),
                        );
                        TransformAction::Replace(into_instructions(scope))
                    }
                }
            }
            Bitwise::LeadingZeros(op) if is_u64(op.input) => {
                let mut scope = scope.child();
                scope.register_type::<IntExpand<0>>(op.input.storage_type());
                let res = u64_leading_zeros::expand::<IntExpand<0>>(
                    &mut scope,
                    ExpandElement::Plain(op.input).into(),
                );
                assign::expand_no_check(&mut scope, res, ExpandElement::Plain(inst.out()).into());
                TransformAction::Replace(into_instructions(scope))
            }
            Bitwise::FindFirstSet(op) if is_u64(op.input) => {
                let mut scope = scope.child();
                scope.register_type::<IntExpand<0>>(op.input.storage_type());
                let res = u64_ffs::expand::<IntExpand<0>>(
                    &mut scope,
                    ExpandElement::Plain(op.input).into(),
                );
                assign::expand_no_check(&mut scope, res, ExpandElement::Plain(inst.out()).into());
                TransformAction::Replace(into_instructions(scope))
            }
            _ => TransformAction::Ignore,
        }
    }
}

fn is_u64(var: Variable) -> bool {
    matches!(
        var.ty.elem_type(),
        ElemType::Int(IntKind::I64) | ElemType::UInt(UIntKind::U64)
    )
}

fn into_instructions(mut scope: Scope) -> Vec<Instruction> {
    scope.process([]).instructions
}

/// Promote bf16 ops to f32, since most Vulkan drivers (radv in particular as
/// of mesa 25.2) don't natively implement bf16 arithmetic / comparison /
/// GLSL.std.450 ext-inst ops on top of `SPV_KHR_bfloat16`. The extension
/// only mandates `OpFMul`, `OpFMulAdd` (FMA) and `OpDot` for bf16; everything
/// else either crashes the driver or returns wrong results
/// (e.g. `OpFOrdGreaterThan` on bf16 effectively returns false → max-reduce
/// returns `bf16::MIN` → softmax NaN).
///
/// This transform rewrites unsupported bf16 instructions to:
///     cast lhs/rhs/in: bf16 -> f32
///     run the op in f32
///     cast result f32 -> bf16
/// Native ops (Mul/Fma/Dot) are left alone so cooperative-matrix and dot-
/// product paths still hit the hardware bf16 units.
#[derive(Debug)]
pub(crate) struct Bf16PromoteTransform;

fn is_bf16(var: &Variable) -> bool {
    matches!(
        var.ty.elem_type(),
        ElemType::Float(FloatKind::BF16)
    )
}

fn promoted_type(orig: Type) -> Option<Type> {
    let line = orig.line_size();
    let promoted_storage = StorageType::Scalar(ElemType::Float(FloatKind::F32));
    Some(Type::new(promoted_storage).line(line))
}

fn cast_to_f32(scope: &mut Scope, src: Variable) -> Variable {
    if !is_bf16(&src) {
        return src;
    }
    let f32_ty = promoted_type(src.ty).expect("bf16 var must have line/scalar type");
    let new_var = scope.create_local(f32_ty);
    scope.register(Instruction::new(
        Operator::Cast(UnaryOperator { input: src }),
        *new_var,
    ));
    *new_var
}

fn emit_cast_back(scope: &mut Scope, f32_result: Variable, original_out: Variable) {
    scope.register(Instruction::new(
        Operator::Cast(UnaryOperator {
            input: f32_result,
        }),
        original_out,
    ));
}

/// Wrap a bf16 unary arithmetic op `op_ctor` in cast / op-in-f32 / cast-back.
fn rewrite_unary<F>(scope: &mut Scope, op: &UnaryOperator, out: Variable, op_ctor: F)
where
    F: FnOnce(UnaryOperator) -> Operation,
{
    let in_f = cast_to_f32(scope, op.input);
    let f32_out = scope.create_local(promoted_type(out.ty).unwrap());
    scope.register(Instruction::new(
        op_ctor(UnaryOperator { input: in_f }),
        *f32_out,
    ));
    emit_cast_back(scope, *f32_out, out);
}

/// Wrap a bf16 binary arithmetic op in cast / op-in-f32 / cast-back.
fn rewrite_binary<F>(scope: &mut Scope, op: &BinaryOperator, out: Variable, op_ctor: F)
where
    F: FnOnce(BinaryOperator) -> Operation,
{
    let lhs_f = cast_to_f32(scope, op.lhs);
    let rhs_f = cast_to_f32(scope, op.rhs);
    let f32_out = scope.create_local(promoted_type(out.ty).unwrap());
    scope.register(Instruction::new(
        op_ctor(BinaryOperator {
            lhs: lhs_f,
            rhs: rhs_f,
        }),
        *f32_out,
    ));
    emit_cast_back(scope, *f32_out, out);
}

/// Wrap a bf16 comparison op (output bool) in cast / op-in-f32 (no cast back).
fn rewrite_comparison_binary<F>(
    scope: &mut Scope,
    op: &BinaryOperator,
    out: Variable,
    op_ctor: F,
) where
    F: FnOnce(BinaryOperator) -> Operation,
{
    let lhs_f = cast_to_f32(scope, op.lhs);
    let rhs_f = cast_to_f32(scope, op.rhs);
    scope.register(Instruction::new(
        op_ctor(BinaryOperator {
            lhs: lhs_f,
            rhs: rhs_f,
        }),
        out,
    ));
}

/// Wrap a bf16 unary comparison op (IsNan/IsInf, output bool).
fn rewrite_comparison_unary<F>(scope: &mut Scope, op: &UnaryOperator, out: Variable, op_ctor: F)
where
    F: FnOnce(UnaryOperator) -> Operation,
{
    let in_f = cast_to_f32(scope, op.input);
    scope.register(Instruction::new(
        op_ctor(UnaryOperator { input: in_f }),
        out,
    ));
}

/// Returns true if any input or output is bf16.
fn has_bf16(inst: &Instruction) -> bool {
    if let Some(out) = inst.out {
        if is_bf16(&out) {
            return true;
        }
    }
    let mut any = false;
    let mut visit = |v: &Variable| {
        if is_bf16(v) {
            any = true;
        }
    };
    match &inst.operation {
        Operation::Arithmetic(a) => match a {
            Arithmetic::Add(o)
            | Arithmetic::Sub(o)
            | Arithmetic::Mul(o)
            | Arithmetic::Div(o)
            | Arithmetic::Modulo(o)
            | Arithmetic::Remainder(o)
            | Arithmetic::Max(o)
            | Arithmetic::Min(o)
            | Arithmetic::Powf(o)
            | Arithmetic::Powi(o)
            | Arithmetic::ArcTan2(o)
            | Arithmetic::Hypot(o)
            | Arithmetic::Rhypot(o)
            | Arithmetic::Dot(o)
            | Arithmetic::SaturatingAdd(o)
            | Arithmetic::SaturatingSub(o)
            | Arithmetic::MulHi(o) => {
                visit(&o.lhs);
                visit(&o.rhs);
            }
            Arithmetic::Abs(u)
            | Arithmetic::Exp(u)
            | Arithmetic::Log(u)
            | Arithmetic::Log1p(u)
            | Arithmetic::Cos(u)
            | Arithmetic::Sin(u)
            | Arithmetic::Tan(u)
            | Arithmetic::Tanh(u)
            | Arithmetic::Sinh(u)
            | Arithmetic::Cosh(u)
            | Arithmetic::ArcCos(u)
            | Arithmetic::ArcSin(u)
            | Arithmetic::ArcTan(u)
            | Arithmetic::ArcSinh(u)
            | Arithmetic::ArcCosh(u)
            | Arithmetic::ArcTanh(u)
            | Arithmetic::Degrees(u)
            | Arithmetic::Radians(u)
            | Arithmetic::Sqrt(u)
            | Arithmetic::InverseSqrt(u)
            | Arithmetic::Round(u)
            | Arithmetic::Floor(u)
            | Arithmetic::Ceil(u)
            | Arithmetic::Trunc(u)
            | Arithmetic::Erf(u)
            | Arithmetic::Recip(u)
            | Arithmetic::Neg(u)
            | Arithmetic::Magnitude(u)
            | Arithmetic::Normalize(u) => {
                visit(&u.input);
            }
            Arithmetic::Clamp(c) => {
                visit(&c.input);
                visit(&c.min_value);
                visit(&c.max_value);
            }
            Arithmetic::Fma(f) => {
                visit(&f.a);
                visit(&f.b);
                visit(&f.c);
            }
        },
        Operation::Comparison(c) => match c {
            Comparison::Lower(o)
            | Comparison::LowerEqual(o)
            | Comparison::Equal(o)
            | Comparison::NotEqual(o)
            | Comparison::GreaterEqual(o)
            | Comparison::Greater(o) => {
                visit(&o.lhs);
                visit(&o.rhs);
            }
            Comparison::IsNan(u) | Comparison::IsInf(u) => {
                visit(&u.input);
            }
        },
        Operation::Plane(p) => match p {
            Plane::Sum(u)
            | Plane::InclusiveSum(u)
            | Plane::ExclusiveSum(u)
            | Plane::Prod(u)
            | Plane::InclusiveProd(u)
            | Plane::ExclusiveProd(u)
            | Plane::Min(u)
            | Plane::Max(u) => {
                visit(&u.input);
            }
            Plane::Broadcast(o)
            | Plane::Shuffle(o)
            | Plane::ShuffleXor(o)
            | Plane::ShuffleUp(o)
            | Plane::ShuffleDown(o) => {
                // shuffle/broadcast pass values through unchanged; promotion
                // not necessary, but it's also harmless. We skip to avoid
                // changing layout of cooperative operations.
                let _ = o;
            }
            _ => {}
        },
        Operation::Operator(Operator::Select(s)) => {
            visit(&s.then);
            visit(&s.or_else);
        }
        _ => {}
    }
    any
}

impl IrTransformer for Bf16PromoteTransform {
    fn maybe_transform(&self, scope: &mut Scope, inst: &Instruction) -> TransformAction {
        if !has_bf16(inst) {
            return TransformAction::Ignore;
        }

        // SPV_KHR_bfloat16 advertises bf16 OpFMul / FMA / Dot as native, but
        // in practice many drivers (radv on mesa 25.2 in particular) don't
        // implement them correctly through the NIR translator -- only the
        // special bfmul/bffma/bfdot intrinsics work, and those aren't what
        // the generic OpFMul / OpFMA / OpDot lower to. So we promote
        // everything (Add/Sub/Mul/Div/Fma/Dot/...) uniformly.
        let out = match inst.out {
            Some(o) => o,
            None => return TransformAction::Ignore,
        };

        let mut child = scope.child();

        match &inst.operation {
            Operation::Arithmetic(a) => match a {
                Arithmetic::Add(o) => rewrite_binary(&mut child, o, out, |op| {
                    Operation::Arithmetic(Arithmetic::Add(op))
                }),
                Arithmetic::Sub(o) => rewrite_binary(&mut child, o, out, |op| {
                    Operation::Arithmetic(Arithmetic::Sub(op))
                }),
                Arithmetic::Mul(o) => rewrite_binary(&mut child, o, out, |op| {
                    Operation::Arithmetic(Arithmetic::Mul(op))
                }),
                Arithmetic::Dot(o) => rewrite_binary(&mut child, o, out, |op| {
                    Operation::Arithmetic(Arithmetic::Dot(op))
                }),
                Arithmetic::Fma(f) => {
                    let a_f = cast_to_f32(&mut child, f.a);
                    let b_f = cast_to_f32(&mut child, f.b);
                    let c_f = cast_to_f32(&mut child, f.c);
                    let f32_out = child.create_local(promoted_type(out.ty).unwrap());
                    child.register(Instruction::new(
                        Operation::Arithmetic(Arithmetic::Fma(FmaOperator {
                            a: a_f,
                            b: b_f,
                            c: c_f,
                        })),
                        *f32_out,
                    ));
                    emit_cast_back(&mut child, *f32_out, out);
                }
                Arithmetic::Div(o) => rewrite_binary(&mut child, o, out, |op| {
                    Operation::Arithmetic(Arithmetic::Div(op))
                }),
                Arithmetic::Modulo(o) => rewrite_binary(&mut child, o, out, |op| {
                    Operation::Arithmetic(Arithmetic::Modulo(op))
                }),
                Arithmetic::Remainder(o) => rewrite_binary(&mut child, o, out, |op| {
                    Operation::Arithmetic(Arithmetic::Remainder(op))
                }),
                Arithmetic::Max(o) => rewrite_binary(&mut child, o, out, |op| {
                    Operation::Arithmetic(Arithmetic::Max(op))
                }),
                Arithmetic::Min(o) => rewrite_binary(&mut child, o, out, |op| {
                    Operation::Arithmetic(Arithmetic::Min(op))
                }),
                Arithmetic::Powf(o) => rewrite_binary(&mut child, o, out, |op| {
                    Operation::Arithmetic(Arithmetic::Powf(op))
                }),
                Arithmetic::Powi(o) => rewrite_binary(&mut child, o, out, |op| {
                    Operation::Arithmetic(Arithmetic::Powi(op))
                }),
                Arithmetic::ArcTan2(o) => rewrite_binary(&mut child, o, out, |op| {
                    Operation::Arithmetic(Arithmetic::ArcTan2(op))
                }),
                Arithmetic::Hypot(o) => rewrite_binary(&mut child, o, out, |op| {
                    Operation::Arithmetic(Arithmetic::Hypot(op))
                }),
                Arithmetic::Rhypot(o) => rewrite_binary(&mut child, o, out, |op| {
                    Operation::Arithmetic(Arithmetic::Rhypot(op))
                }),
                Arithmetic::Abs(u) => rewrite_unary(&mut child, u, out, |op| {
                    Operation::Arithmetic(Arithmetic::Abs(op))
                }),
                Arithmetic::Exp(u) => rewrite_unary(&mut child, u, out, |op| {
                    Operation::Arithmetic(Arithmetic::Exp(op))
                }),
                Arithmetic::Log(u) => rewrite_unary(&mut child, u, out, |op| {
                    Operation::Arithmetic(Arithmetic::Log(op))
                }),
                Arithmetic::Log1p(u) => rewrite_unary(&mut child, u, out, |op| {
                    Operation::Arithmetic(Arithmetic::Log1p(op))
                }),
                Arithmetic::Cos(u) => rewrite_unary(&mut child, u, out, |op| {
                    Operation::Arithmetic(Arithmetic::Cos(op))
                }),
                Arithmetic::Sin(u) => rewrite_unary(&mut child, u, out, |op| {
                    Operation::Arithmetic(Arithmetic::Sin(op))
                }),
                Arithmetic::Tan(u) => rewrite_unary(&mut child, u, out, |op| {
                    Operation::Arithmetic(Arithmetic::Tan(op))
                }),
                Arithmetic::Tanh(u) => rewrite_unary(&mut child, u, out, |op| {
                    Operation::Arithmetic(Arithmetic::Tanh(op))
                }),
                Arithmetic::Sinh(u) => rewrite_unary(&mut child, u, out, |op| {
                    Operation::Arithmetic(Arithmetic::Sinh(op))
                }),
                Arithmetic::Cosh(u) => rewrite_unary(&mut child, u, out, |op| {
                    Operation::Arithmetic(Arithmetic::Cosh(op))
                }),
                Arithmetic::ArcCos(u) => rewrite_unary(&mut child, u, out, |op| {
                    Operation::Arithmetic(Arithmetic::ArcCos(op))
                }),
                Arithmetic::ArcSin(u) => rewrite_unary(&mut child, u, out, |op| {
                    Operation::Arithmetic(Arithmetic::ArcSin(op))
                }),
                Arithmetic::ArcTan(u) => rewrite_unary(&mut child, u, out, |op| {
                    Operation::Arithmetic(Arithmetic::ArcTan(op))
                }),
                Arithmetic::ArcSinh(u) => rewrite_unary(&mut child, u, out, |op| {
                    Operation::Arithmetic(Arithmetic::ArcSinh(op))
                }),
                Arithmetic::ArcCosh(u) => rewrite_unary(&mut child, u, out, |op| {
                    Operation::Arithmetic(Arithmetic::ArcCosh(op))
                }),
                Arithmetic::ArcTanh(u) => rewrite_unary(&mut child, u, out, |op| {
                    Operation::Arithmetic(Arithmetic::ArcTanh(op))
                }),
                Arithmetic::Degrees(u) => rewrite_unary(&mut child, u, out, |op| {
                    Operation::Arithmetic(Arithmetic::Degrees(op))
                }),
                Arithmetic::Radians(u) => rewrite_unary(&mut child, u, out, |op| {
                    Operation::Arithmetic(Arithmetic::Radians(op))
                }),
                Arithmetic::Sqrt(u) => rewrite_unary(&mut child, u, out, |op| {
                    Operation::Arithmetic(Arithmetic::Sqrt(op))
                }),
                Arithmetic::InverseSqrt(u) => rewrite_unary(&mut child, u, out, |op| {
                    Operation::Arithmetic(Arithmetic::InverseSqrt(op))
                }),
                Arithmetic::Round(u) => rewrite_unary(&mut child, u, out, |op| {
                    Operation::Arithmetic(Arithmetic::Round(op))
                }),
                Arithmetic::Floor(u) => rewrite_unary(&mut child, u, out, |op| {
                    Operation::Arithmetic(Arithmetic::Floor(op))
                }),
                Arithmetic::Ceil(u) => rewrite_unary(&mut child, u, out, |op| {
                    Operation::Arithmetic(Arithmetic::Ceil(op))
                }),
                Arithmetic::Trunc(u) => rewrite_unary(&mut child, u, out, |op| {
                    Operation::Arithmetic(Arithmetic::Trunc(op))
                }),
                Arithmetic::Recip(u) => rewrite_unary(&mut child, u, out, |op| {
                    Operation::Arithmetic(Arithmetic::Recip(op))
                }),
                Arithmetic::Neg(u) => rewrite_unary(&mut child, u, out, |op| {
                    Operation::Arithmetic(Arithmetic::Neg(op))
                }),
                Arithmetic::Magnitude(u) => rewrite_unary(&mut child, u, out, |op| {
                    Operation::Arithmetic(Arithmetic::Magnitude(op))
                }),
                Arithmetic::Normalize(u) => rewrite_unary(&mut child, u, out, |op| {
                    Operation::Arithmetic(Arithmetic::Normalize(op))
                }),
                Arithmetic::Erf(_) => {
                    // Already polyfilled by ErfTransform; let it run there.
                    return TransformAction::Ignore;
                }
                Arithmetic::Clamp(c) => {
                    let in_f = cast_to_f32(&mut child, c.input);
                    let min_f = cast_to_f32(&mut child, c.min_value);
                    let max_f = cast_to_f32(&mut child, c.max_value);
                    let f32_out = child.create_local(promoted_type(out.ty).unwrap());
                    child.register(Instruction::new(
                        Operation::Arithmetic(Arithmetic::Clamp(ClampOperator {
                            input: in_f,
                            min_value: min_f,
                            max_value: max_f,
                        })),
                        *f32_out,
                    ));
                    emit_cast_back(&mut child, *f32_out, out);
                }
                // Skip ones we said we'd skip (Mul/Fma/Dot handled above) +
                // odd cases we don't expect on bf16 (SaturatingAdd/Sub/MulHi
                // are int-only in practice). Defensively pass through.
                _ => return TransformAction::Ignore,
            },
            Operation::Comparison(c) => match c {
                Comparison::Lower(o) => rewrite_comparison_binary(&mut child, o, out, |op| {
                    Operation::Comparison(Comparison::Lower(op))
                }),
                Comparison::LowerEqual(o) => {
                    rewrite_comparison_binary(&mut child, o, out, |op| {
                        Operation::Comparison(Comparison::LowerEqual(op))
                    })
                }
                Comparison::Equal(o) => rewrite_comparison_binary(&mut child, o, out, |op| {
                    Operation::Comparison(Comparison::Equal(op))
                }),
                Comparison::NotEqual(o) => rewrite_comparison_binary(&mut child, o, out, |op| {
                    Operation::Comparison(Comparison::NotEqual(op))
                }),
                Comparison::GreaterEqual(o) => {
                    rewrite_comparison_binary(&mut child, o, out, |op| {
                        Operation::Comparison(Comparison::GreaterEqual(op))
                    })
                }
                Comparison::Greater(o) => rewrite_comparison_binary(&mut child, o, out, |op| {
                    Operation::Comparison(Comparison::Greater(op))
                }),
                Comparison::IsNan(u) => rewrite_comparison_unary(&mut child, u, out, |op| {
                    Operation::Comparison(Comparison::IsNan(op))
                }),
                Comparison::IsInf(u) => rewrite_comparison_unary(&mut child, u, out, |op| {
                    Operation::Comparison(Comparison::IsInf(op))
                }),
            },
            Operation::Plane(p) => match p {
                Plane::Sum(u) => rewrite_unary(&mut child, u, out, |op| {
                    Operation::Plane(Plane::Sum(op))
                }),
                Plane::InclusiveSum(u) => rewrite_unary(&mut child, u, out, |op| {
                    Operation::Plane(Plane::InclusiveSum(op))
                }),
                Plane::ExclusiveSum(u) => rewrite_unary(&mut child, u, out, |op| {
                    Operation::Plane(Plane::ExclusiveSum(op))
                }),
                Plane::Prod(u) => rewrite_unary(&mut child, u, out, |op| {
                    Operation::Plane(Plane::Prod(op))
                }),
                Plane::InclusiveProd(u) => rewrite_unary(&mut child, u, out, |op| {
                    Operation::Plane(Plane::InclusiveProd(op))
                }),
                Plane::ExclusiveProd(u) => rewrite_unary(&mut child, u, out, |op| {
                    Operation::Plane(Plane::ExclusiveProd(op))
                }),
                Plane::Min(u) => rewrite_unary(&mut child, u, out, |op| {
                    Operation::Plane(Plane::Min(op))
                }),
                Plane::Max(u) => rewrite_unary(&mut child, u, out, |op| {
                    Operation::Plane(Plane::Max(op))
                }),
                _ => return TransformAction::Ignore,
            },
            Operation::Operator(Operator::Select(s)) => {
                // Select acts as a ternary: cond ? then : or_else. Both branches
                // are bf16 here. Promote both, run select in f32, cast back.
                let then_f = cast_to_f32(&mut child, s.then);
                let else_f = cast_to_f32(&mut child, s.or_else);
                let f32_out = child.create_local(promoted_type(out.ty).unwrap());
                child.register(Instruction::new(
                    Operation::Operator(Operator::Select(Select {
                        cond: s.cond,
                        then: then_f,
                        or_else: else_f,
                    })),
                    *f32_out,
                ));
                emit_cast_back(&mut child, *f32_out, out);
            }
            _ => return TransformAction::Ignore,
        }

        TransformAction::Replace(into_instructions(child))
    }
}
