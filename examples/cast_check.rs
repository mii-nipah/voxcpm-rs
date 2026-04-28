// Quick check whether Tensor::cast(F32) actually changes dtype + storage on
// the bf16 Vulkan backend. If `cast` is a silent no-op, the variance kernel
// in MiniCpmRmsNorm overflows in bf16 and explodes to NaN.

use burn::prelude::*;
use burn::tensor::{Element, FloatDType, TensorData};

#[cfg(feature = "vulkan")]
type Backend = burn::backend::Vulkan<half::bf16, i32>;
#[cfg(all(feature = "wgpu", not(feature = "vulkan")))]
type Backend = burn::backend::Wgpu<half::bf16, i32>;

fn main() {
    let device = Default::default();
    eprintln!("default float dtype: {:?}", <Backend as burn::tensor::backend::Backend>::FloatElem::dtype());

    // 1) Make a bf16 tensor with values ~1.0.
    let data: Vec<f32> = (0..8).map(|i| 1.0 + i as f32 * 0.01).collect();
    let t: Tensor<Backend, 1> = Tensor::<Backend, 1>::from_data(TensorData::new(data, [8]), &device);
    eprintln!("t dtype before cast: {:?}", t.dtype());

    // 2) Cast to F32.
    let tf = t.clone().cast(FloatDType::F32);
    eprintln!("tf dtype after cast(F32): {:?}", tf.dtype());

    // 3) Compute sum of squares in both, compare.
    let sum_bf16 = (t.clone() * t.clone()).sum().into_data().convert::<f32>().into_vec::<f32>().unwrap()[0];
    let sum_f32  = (tf.clone() * tf.clone()).sum().into_data().convert::<f32>().into_vec::<f32>().unwrap()[0];
    eprintln!("sum(t^2) in bf16 path: {sum_bf16}");
    eprintln!("sum(t^2) in f32  path: {sum_f32}");

    // 4) Big-tensor stress: 4096 elements ~= one row of hidden dim.
    let big: Vec<f32> = (0..4096).map(|i| 0.1 + (i as f32).sin() * 0.05).collect();
    let bt: Tensor<Backend, 1> = Tensor::<Backend, 1>::from_data(TensorData::new(big, [4096]), &device);
    let btf = bt.clone().cast(FloatDType::F32);
    let m_bf16 = (bt.clone() * bt.clone()).mean().into_data().convert::<f32>().into_vec::<f32>().unwrap()[0];
    let m_f32  = (btf.clone() * btf.clone()).mean().into_data().convert::<f32>().into_vec::<f32>().unwrap()[0];
    eprintln!("mean(x^2) in bf16: {m_bf16}");
    eprintln!("mean(x^2) in f32 : {m_f32}");

    let v_bf16 = m_bf16.is_nan();
    let v_f32 = m_f32.is_nan();
    eprintln!("nan? bf16={v_bf16} f32={v_f32}");

    // 5) Cast f32 -> bf16 round-trip: take an f32 tensor, cast to bf16, then
    // multiply by another bf16 tensor and read the result.
    eprintln!("--- f32->bf16 round trip ---");
    let a_f: Tensor<Backend, 1> = Tensor::<Backend, 1>::from_data(
        TensorData::new(vec![10.0f32; 8], [8]), &device).cast(FloatDType::F32);
    eprintln!("a_f dtype: {:?}", a_f.dtype());
    let a_bf = a_f.clone().cast(FloatDType::BF16);
    eprintln!("a_bf dtype after cast: {:?}", a_bf.dtype());
    let b_bf: Tensor<Backend, 1> = Tensor::<Backend, 1>::from_data(
        TensorData::new(vec![2.0f32; 8], [8]), &device);
    eprintln!("b_bf dtype: {:?}", b_bf.dtype());
    let prod = a_bf * b_bf;
    let v: Vec<f32> = prod.into_data().convert::<f32>().into_vec().unwrap();
    eprintln!("(10.0 cast f32->bf16) * 2.0 (bf16) = {:?} (expected ~20.0)", &v[..4]);

    // 6) Add: bf16-from-f32 + bf16-native
    let a_f: Tensor<Backend, 1> = Tensor::<Backend, 1>::from_data(
        TensorData::new(vec![3.0f32; 8], [8]), &device).cast(FloatDType::F32);
    let a_bf = a_f.cast(FloatDType::BF16);
    let b_bf: Tensor<Backend, 1> = Tensor::<Backend, 1>::from_data(
        TensorData::new(vec![5.0f32; 8], [8]), &device);
    let s = a_bf + b_bf;
    let v: Vec<f32> = s.into_data().convert::<f32>().into_vec().unwrap();
    eprintln!("(3.0 cast f32->bf16) + 5.0 (bf16) = {:?} (expected ~8.0)", &v[..4]);

    // 7) f32 input, bf16 weight matmul (Linear forward path).
    eprintln!("--- mixed dtype matmul ---");
    let a_f: Tensor<Backend, 2> = Tensor::<Backend, 2>::from_data(
        TensorData::new(vec![1.0f32; 4 * 8], [4, 8]), &device).cast(FloatDType::F32);
    let w_bf: Tensor<Backend, 2> = Tensor::<Backend, 2>::from_data(
        TensorData::new(vec![0.5f32; 8 * 4], [8, 4]), &device);
    eprintln!("a_f={:?} w_bf={:?}", a_f.dtype(), w_bf.dtype());
    let r = a_f.matmul(w_bf);
    eprintln!("r dtype: {:?}", r.dtype());
    let v: Vec<f32> = r.into_data().convert::<f32>().into_vec().unwrap();
    eprintln!("ones[4,8] @ halves[8,4] = {:?} (expected 4.0 each)", &v[..8]);
}
