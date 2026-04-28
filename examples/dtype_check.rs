// Rigorous dtype check: use `from_data_dtype` to be unambiguous about the
// storage dtype of every tensor, and probe each suspicious op in isolation.
//
// Compares the same op on the cubecl Vulkan/SPIR-V backend (bf16) against
// the ndarray CPU backend (bf16) to attribute discrepancies.

use burn::prelude::*;
use burn::tensor::{DType, TensorData, activation::silu};

#[cfg(feature = "vulkan")]
type Gpu = burn::backend::Vulkan<half::bf16, i32>;
#[cfg(all(feature = "wgpu", not(feature = "vulkan")))]
type Gpu = burn::backend::Wgpu<half::bf16, i32>;

type Cpu = burn::backend::NdArray<f32>;

fn report<const D: usize, B: Backend>(label: &str, t: Tensor<B, D>) {
    let dtype = t.dtype();
    let v: Vec<f32> = t.into_data().convert::<f32>().into_vec().unwrap();
    let head: Vec<f32> = v.iter().take(4).copied().collect();
    eprintln!("  {label:<40} dtype={dtype:?} head={head:?}");
}

fn run<B: Backend>(tag: &str, device: &B::Device) {
    eprintln!("\n=== backend: {tag} ===");

    // 1) Construct an unambiguous BF16 tensor.
    let bf_data = TensorData::new(vec![half::bf16::from_f32(10.0); 8], [8]);
    eprintln!("input TensorData dtype={:?}", bf_data.dtype);
    let a: Tensor<B, 1> = Tensor::from_data(bf_data, device);
    report("a = bf16(10.0)", a.clone());

    let b: Tensor<B, 1> = Tensor::from_data(
        TensorData::new(vec![half::bf16::from_f32(2.0); 8], [8]),
        device,
    );
    report("b = bf16(2.0)", b.clone());

    // 2) Pure bf16 mul.
    report("a * b (bf16*bf16)", a.clone() * b.clone());

    // 3) silu on a moderate bf16 value.
    let x: Tensor<B, 1> = Tensor::from_data(
        TensorData::new(vec![half::bf16::from_f32(23.75); 8], [8]),
        device,
    );
    report("x = bf16(23.75)", x.clone());
    report("silu(x) (expect ~23.75)", silu(x.clone()));

    // 4) silu on small values.
    let xs: Tensor<B, 1> = Tensor::from_data(
        TensorData::new(vec![half::bf16::from_f32(1.5); 8], [8]),
        device,
    );
    report("silu(bf16(1.5)) expect ~1.23", silu(xs));

    // 5) Cast bf16 -> f32 -> bf16, then mul with bf16. Forces an explicit
    //    f32 intermediate to test the 'cast back' path.
    let af = a.clone().cast(burn::tensor::FloatDType::F32);
    report("af = a.cast(F32)", af.clone());
    let aback = af.cast(burn::tensor::FloatDType::BF16);
    report("aback = af.cast(BF16)", aback.clone());
    report("aback * b (expect 20.0)", aback * b.clone());

    // 6) Mixed dtype: f32 * bf16 directly.
    let f: Tensor<B, 1> = Tensor::from_data_dtype(
        TensorData::new(vec![10.0f32; 8], [8]),
        device,
        DType::F32,
    );
    report("f = f32(10.0)", f.clone());
    report("f * b (mixed f32*bf16)", f * b);

    // 7) RMSNorm-style sequence: square + reduce-mean + rsqrt over 4096 dims.
    let big_vals: Vec<half::bf16> = (0..4096)
        .map(|i| half::bf16::from_f32(0.1 + (i as f32 * 0.001).sin()))
        .collect();
    let big: Tensor<B, 1> = Tensor::from_data(
        TensorData::new(big_vals.clone(), [4096]),
        device,
    );
    let var_bf = (big.clone() * big.clone()).mean();
    let var_bf_v: f32 = var_bf.into_data().convert::<f32>().into_vec::<f32>().unwrap()[0];
    eprintln!("  mean(x*x) bf16   = {var_bf_v}");
    let big_f = big.clone().cast(burn::tensor::FloatDType::F32);
    let var_f = (big_f.clone() * big_f).mean();
    let var_f_v: f32 = var_f.into_data().convert::<f32>().into_vec::<f32>().unwrap()[0];
    eprintln!("  mean(x*x) f32    = {var_f_v}");
}

fn main() {
    run::<Gpu>("gpu(bf16)", &Default::default());
    eprintln!("\n--- CPU ndarray reference (note: bf16 not supported, cast steps skipped) ---");
    // Skip CPU run since it panics on cast(BF16). f32 NdArray can't host bf16
    // tensors at all.
}
