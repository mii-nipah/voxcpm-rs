// Compare bf16 matmul output against f32 reference to detect accumulator bugs
// in cubecl-matmul / cubecl-spirv on the active hardware.

use burn::prelude::*;
use burn::tensor::{Element, FloatDType, TensorData};

#[cfg(feature = "vulkan")]
type Backend = burn::backend::Vulkan<half::bf16, i32>;
#[cfg(all(feature = "wgpu", not(feature = "vulkan")))]
type Backend = burn::backend::Wgpu<half::bf16, i32>;

fn rand_vec(n: usize, seed: u64) -> Vec<f32> {
    // Deterministic-ish PRNG.
    let mut s = seed;
    (0..n).map(|_| {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u = ((s >> 33) as u32) as f32 / u32::MAX as f32 * 2.0 - 1.0;
        u * 0.05
    }).collect()
}

fn main() {
    let device = Default::default();
    eprintln!("default float dtype: {:?}", <Backend as burn::tensor::backend::Backend>::FloatElem::dtype());

    let m = 64;
    let k = 1024;
    let n = 2048;

    let a_data = rand_vec(m * k, 1);
    let b_data = rand_vec(k * n, 2);

    // bf16 matmul (the backend's native dtype).
    let a_bf = Tensor::<Backend, 2>::from_data(TensorData::new(a_data.clone(), [m, k]), &device);
    let b_bf = Tensor::<Backend, 2>::from_data(TensorData::new(b_data.clone(), [k, n]), &device);
    let c_bf = a_bf.matmul(b_bf);

    // f32 matmul (cast inputs).
    let a_f = Tensor::<Backend, 2>::from_data(TensorData::new(a_data.clone(), [m, k]), &device).cast(FloatDType::F32);
    let b_f = Tensor::<Backend, 2>::from_data(TensorData::new(b_data.clone(), [k, n]), &device).cast(FloatDType::F32);
    let c_f = a_f.matmul(b_f);

    let v_bf: Vec<f32> = c_bf.into_data().convert::<f32>().into_vec().unwrap();
    let v_f: Vec<f32> = c_f.into_data().convert::<f32>().into_vec().unwrap();

    let max_bf = v_bf.iter().fold(0f32, |a, b| a.max(b.abs()));
    let max_f = v_f.iter().fold(0f32, |a, b| a.max(b.abs()));
    let mean_abs_bf = v_bf.iter().map(|x| x.abs()).sum::<f32>() / v_bf.len() as f32;
    let mean_abs_f = v_f.iter().map(|x| x.abs()).sum::<f32>() / v_f.len() as f32;

    let mut diffs: Vec<f32> = v_bf.iter().zip(v_f.iter()).map(|(a, b)| (a - b).abs()).collect();
    diffs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = diffs[diffs.len() / 2];
    let p99 = diffs[diffs.len() * 99 / 100];
    let max_diff = diffs.last().copied().unwrap_or(0.0);

    eprintln!("bf16 max={max_bf:.4} mean_abs={mean_abs_bf:.4}");
    eprintln!("f32  max={max_f:.4} mean_abs={mean_abs_f:.4}");
    eprintln!("|bf - f|: median={median:.4} p99={p99:.4} max={max_diff:.4}");
    eprintln!("ratio bf/f mean: {:.4}", mean_abs_bf / mean_abs_f);
}
