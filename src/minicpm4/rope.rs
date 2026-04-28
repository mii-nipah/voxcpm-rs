//! Rotary position embedding with the MiniCPM LongRoPE scaling variant.

use crate::config::MiniCpm4Config;
use burn::prelude::*;
use burn::tensor::{Int, TensorData};

/// Precomputed cosine/sine caches.
#[derive(Module, Debug)]
pub struct MiniCpmLongRope<B: Backend> {
    pub cos: Tensor<B, 2>, // [max_seq_len, head_dim]
    pub sin: Tensor<B, 2>,
    pub head_dim: usize,
}

impl<B: Backend> MiniCpmLongRope<B> {
    pub fn new(config: &MiniCpm4Config, device: &B::Device) -> Self {
        let head_dim = config.head_dim();
        let base = config.rope_theta as f64;
        let max_positions = config.max_position_embeddings;
        let original = config.rope_scaling.original_max_position_embeddings;

        let scale = max_positions as f64 / original as f64;
        let scaling_factor = (1.0 + scale.ln() / (original as f64).ln()).sqrt().max(1.0);

        let ext_factors: Vec<f64> = if max_positions > original {
            config.rope_scaling.long_factor.iter().map(|x| *x as f64).collect()
        } else {
            config.rope_scaling.short_factor.iter().map(|x| *x as f64).collect()
        };

        let half = head_dim / 2;
        assert_eq!(ext_factors.len(), half, "rope_scaling factor length must equal head_dim/2");

        let mut cos = vec![0f32; max_positions * head_dim];
        let mut sin = vec![0f32; max_positions * head_dim];
        for t in 0..max_positions {
            for i in 0..half {
                let inv_freq = 1.0 / base.powf(2.0 * i as f64 / head_dim as f64);
                let freq = (t as f64) * inv_freq / ext_factors[i];
                let (s, c) = freq.sin_cos();
                let c = c * scaling_factor;
                let s = s * scaling_factor;
                cos[t * head_dim + i] = c as f32;
                cos[t * head_dim + i + half] = c as f32;
                sin[t * head_dim + i] = s as f32;
                sin[t * head_dim + i + half] = s as f32;
            }
        }

        let cos = Tensor::from_data(TensorData::new(cos, [max_positions, head_dim]), device);
        let sin = Tensor::from_data(TensorData::new(sin, [max_positions, head_dim]), device);

        Self { cos, sin, head_dim }
    }

    /// Select the (cos, sin) rows at the given `position_ids` (shape `[S]`).
    /// Returns tensors of shape `[S, head_dim]`.
    pub fn gather(&self, position_ids: Tensor<B, 1, Int>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let cos = self.cos.clone().select(0, position_ids.clone());
        let sin = self.sin.clone().select(0, position_ids);
        (cos, sin)
    }
}

/// Rotate the last dim by splitting into two halves and returning `[-x2, x1]`.
pub fn rotate_half<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    let dims = x.dims();
    let last = D - 1;
    let half = dims[last] / 2;
    let x1 = x.clone().narrow(last, 0, half);
    let x2 = x.narrow(last, half, half);
    Tensor::cat(vec![x2.neg(), x1], last)
}

/// Apply RoPE to q, k (both shape `[B, H, S, D]`) given cos/sin of shape `[S, D]`.
pub fn apply_rotary_pos_emb<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    cos: Tensor<B, 2>,
    sin: Tensor<B, 2>,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    // Note: Python casts q/k to f32 for the rotation. We *don't* mirror that
    // on bf16 backends because chaining a `cast(F32)` -> ops -> `cast(BF16)`
    // through the cubecl Vulkan/SPIR-V graph triggers a downstream
    // dtype-promotion bug that inflates subsequent additions by 40-100x. The
    // rotation itself is just element-wise mul/add over `head_dim`, which
    // bf16 handles cleanly without any reduction. RMSNorm and SiLU are the
    // ones that genuinely need f32 (and they upcast internally).
    let cos4: Tensor<B, 4> = cos.unsqueeze();
    let sin4: Tensor<B, 4> = sin.unsqueeze();
    let q_embed = q.clone() * cos4.clone() + rotate_half(q) * sin4.clone();
    let k_embed = k.clone() * cos4 + rotate_half(k) * sin4;
    (q_embed, k_embed)
}
