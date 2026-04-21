//! Grouped-query multi-head attention with rotary embeddings.

use crate::config::MiniCpm4Config;
use crate::minicpm4::rope::apply_rotary_pos_emb;
use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation::softmax;
use burn::tensor::{Int, TensorData};

/// `(key_cache, value_cache)` for a single layer, shape `[B, num_kv_heads, S_max, head_dim]`.
pub type LayerKv<B> = (Tensor<B, 4>, Tensor<B, 4>);

#[derive(Module, Debug)]
pub struct MiniCpmAttention<B: Backend> {
    pub q_proj: Linear<B>,
    pub k_proj: Linear<B>,
    pub v_proj: Linear<B>,
    pub o_proj: Linear<B>,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub scale: f64,
}

impl<B: Backend> MiniCpmAttention<B> {
    pub fn new(config: &MiniCpm4Config, device: &B::Device) -> Self {
        let head_dim = config.head_dim();
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;
        let hidden = config.hidden_size;

        let q_proj = LinearConfig::new(hidden, num_heads * head_dim).with_bias(false).init(device);
        let k_proj = LinearConfig::new(hidden, num_kv_heads * head_dim).with_bias(false).init(device);
        let v_proj = LinearConfig::new(hidden, num_kv_heads * head_dim).with_bias(false).init(device);
        let o_proj = LinearConfig::new(num_heads * head_dim, hidden).with_bias(false).init(device);

        Self {
            q_proj, k_proj, v_proj, o_proj,
            num_heads, num_kv_heads, head_dim,
            scale: 1.0 / (head_dim as f64).sqrt(),
        }
    }

    /// Prefill pass. Returns `(out, (k, v))` where (k, v) are pre-RoPE-applied
    /// per-head tensors of shape `[B, Hkv, S, D]`.
    pub fn forward(
        &self,
        hidden_states: Tensor<B, 3>,
        position_emb: Option<(Tensor<B, 2>, Tensor<B, 2>)>,
        is_causal: bool,
    ) -> (Tensor<B, 3>, LayerKv<B>) {
        let [bsz, q_len, _] = hidden_states.dims();

        let q = self.q_proj.forward(hidden_states.clone());
        let k = self.k_proj.forward(hidden_states.clone());
        let v = self.v_proj.forward(hidden_states);

        let q = q.reshape([bsz, q_len, self.num_heads, self.head_dim]).swap_dims(1, 2);
        let k = k.reshape([bsz, q_len, self.num_kv_heads, self.head_dim]).swap_dims(1, 2);
        let v = v.reshape([bsz, q_len, self.num_kv_heads, self.head_dim]).swap_dims(1, 2);

        let (q, k) = if let Some((cos, sin)) = position_emb {
            apply_rotary_pos_emb(q, k, cos, sin)
        } else {
            (q, k)
        };

        let n_rep = self.num_heads / self.num_kv_heads;
        let k_full = repeat_kv(k.clone(), n_rep);
        let v_full = repeat_kv(v.clone(), n_rep);

        let attn = self.sdpa(q, k_full, v_full, is_causal, None);

        let attn = attn.swap_dims(1, 2).reshape([bsz, q_len, self.num_heads * self.head_dim]);
        let out = self.o_proj.forward(attn);
        (out, (k, v))
    }

    /// Single-step decoding against a cached KV tensor.
    pub fn forward_step(
        &self,
        hidden_states: Tensor<B, 2>,
        position_emb: Option<(Tensor<B, 2>, Tensor<B, 2>)>,
        position_id: usize,
        kv_cache: &mut LayerKv<B>,
    ) -> Tensor<B, 2> {
        let [bsz, _] = hidden_states.dims();
        let device = hidden_states.device();

        let q = self.q_proj.forward(hidden_states.clone());
        let k = self.k_proj.forward(hidden_states.clone());
        let v = self.v_proj.forward(hidden_states);

        let q: Tensor<B, 4> = q.reshape([bsz, 1, self.num_heads, self.head_dim]).swap_dims(1, 2);
        let k: Tensor<B, 4> = k.reshape([bsz, 1, self.num_kv_heads, self.head_dim]).swap_dims(1, 2);
        let v: Tensor<B, 4> = v.reshape([bsz, 1, self.num_kv_heads, self.head_dim]).swap_dims(1, 2);

        let (q, k) = if let Some((cos, sin)) = position_emb {
            apply_rotary_pos_emb(q, k, cos, sin)
        } else {
            (q, k)
        };

        let (key_cache, value_cache) = kv_cache;
        *key_cache = key_cache.clone().slice_assign(
            [0..bsz, 0..self.num_kv_heads, position_id..position_id + 1, 0..self.head_dim],
            k,
        );
        *value_cache = value_cache.clone().slice_assign(
            [0..bsz, 0..self.num_kv_heads, position_id..position_id + 1, 0..self.head_dim],
            v,
        );

        let max_len = key_cache.dims()[2];

        let idx = Tensor::<B, 1, Int>::arange(0..max_len as i64, &device);
        let mask = idx.greater_elem(position_id as i64); // [max_len], true where masked
        let mask: Tensor<B, 4, burn::tensor::Bool> = mask.unsqueeze();

        let n_rep = self.num_heads / self.num_kv_heads;
        let k_full = repeat_kv(key_cache.clone(), n_rep);
        let v_full = repeat_kv(value_cache.clone(), n_rep);

        let attn = self.sdpa(q, k_full, v_full, false, Some(mask));

        let attn = attn.swap_dims(1, 2).reshape([bsz, self.num_heads * self.head_dim]);
        self.o_proj.forward(attn)
    }

    fn sdpa(
        &self,
        q: Tensor<B, 4>,
        k: Tensor<B, 4>,
        v: Tensor<B, 4>,
        is_causal: bool,
        attn_mask: Option<Tensor<B, 4, burn::tensor::Bool>>,
    ) -> Tensor<B, 4> {
        let [b, h, s_q, _d] = q.dims();
        let s_k = k.dims()[2];

        let scores = q.matmul(k.swap_dims(2, 3)).mul_scalar(self.scale);

        let scores = if is_causal {
            let device = scores.device();
            let mut data = vec![false; s_q * s_k];
            for i in 0..s_q {
                for j in 0..s_k {
                    data[i * s_k + j] = j > i;
                }
            }
            let mask = Tensor::<B, 1, burn::tensor::Bool>::from_data(
                TensorData::new(data, [s_q * s_k]),
                &device,
            )
            .reshape([1usize, 1, s_q, s_k]);
            let bcast: Tensor<B, 4, burn::tensor::Bool> = mask.expand([b, h, s_q, s_k]);
            scores.mask_fill(bcast, f32::NEG_INFINITY)
        } else if let Some(mask) = attn_mask {
            let bcast: Tensor<B, 4, burn::tensor::Bool> = mask.expand([b, h, s_q, s_k]);
            scores.mask_fill(bcast, f32::NEG_INFINITY)
        } else {
            scores
        };

        let attn = softmax(scores, 3);
        attn.matmul(v)
    }
}

/// Repeat the key/value heads along the head axis so GQA can be done as a normal MHA.
pub fn repeat_kv<B: Backend>(x: Tensor<B, 4>, n_rep: usize) -> Tensor<B, 4> {
    if n_rep == 1 {
        return x;
    }
    let [b, hkv, s, d] = x.dims();
    let x: Tensor<B, 5> = x.unsqueeze_dim(2);
    let x = x.expand([b, hkv, n_rep, s, d]);
    x.reshape([b, hkv * n_rep, s, d])
}
