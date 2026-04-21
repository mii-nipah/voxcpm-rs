//! Local DiT V2: transformer backbone used as the flow-matching velocity estimator.

use crate::config::MiniCpm4Config;
use crate::minicpm4::MiniCpmModel;
use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation::silu;
use burn::tensor::TensorData;

/// Sinusoidal timestep embedding (even dim required).
#[derive(Module, Debug)]
pub struct SinusoidalPosEmb<B: Backend> {
    pub dim: usize,
    _phantom: core::marker::PhantomData<B>,
}

impl<B: Backend> SinusoidalPosEmb<B> {
    pub fn new(dim: usize) -> Self {
        assert_eq!(dim % 2, 0, "SinusoidalPosEmb requires even dim");
        Self { dim, _phantom: core::marker::PhantomData }
    }

    /// `x`: `[N]` (scalar timesteps). Returns `[N, dim]`.
    pub fn forward(&self, x: Tensor<B, 1>, scale: f64) -> Tensor<B, 2> {
        let device = x.device();
        let half = self.dim / 2;
        let emb_scale = (10000f64).ln() / (half as f64 - 1.0);
        let freqs: Vec<f32> = (0..half).map(|i| (-emb_scale * i as f64).exp() as f32).collect();
        let freqs = Tensor::<B, 1>::from_data(TensorData::new(freqs, [half]), &device);
        let angles = x.unsqueeze_dim::<2>(1).mul_scalar(scale) * freqs.unsqueeze::<2>();
        Tensor::cat(vec![angles.clone().sin(), angles.cos()], 1)
    }
}

#[derive(Module, Debug)]
pub struct TimestepEmbedding<B: Backend> {
    pub linear_1: Linear<B>,
    pub linear_2: Linear<B>,
}

impl<B: Backend> TimestepEmbedding<B> {
    pub fn new(in_channels: usize, time_embed_dim: usize, out_dim: Option<usize>, device: &B::Device) -> Self {
        let out = out_dim.unwrap_or(time_embed_dim);
        Self {
            linear_1: LinearConfig::new(in_channels, time_embed_dim).init(device),
            linear_2: LinearConfig::new(time_embed_dim, out).init(device),
        }
    }

    pub fn forward(&self, sample: Tensor<B, 2>) -> Tensor<B, 2> {
        let h = self.linear_1.forward(sample);
        let h = silu(h);
        self.linear_2.forward(h)
    }
}

#[derive(Module, Debug)]
pub struct VoxCpmLocDiTV2<B: Backend> {
    pub in_proj: Linear<B>,
    pub cond_proj: Linear<B>,
    pub out_proj: Linear<B>,
    pub time_embeddings: SinusoidalPosEmb<B>,
    pub time_mlp: TimestepEmbedding<B>,
    pub delta_time_mlp: TimestepEmbedding<B>,
    pub decoder: MiniCpmModel<B>,
    pub in_channels: usize,
}

impl<B: Backend> VoxCpmLocDiTV2<B> {
    pub fn new(config: MiniCpm4Config, in_channels: usize, device: &B::Device) -> Self {
        assert_eq!(config.vocab_size, 0, "vocab_size must be 0 for local DiT");
        let hidden = config.hidden_size;
        Self {
            in_proj: LinearConfig::new(in_channels, hidden).init(device),
            cond_proj: LinearConfig::new(in_channels, hidden).init(device),
            out_proj: LinearConfig::new(hidden, in_channels).init(device),
            time_embeddings: SinusoidalPosEmb::new(hidden),
            time_mlp: TimestepEmbedding::new(hidden, hidden, None, device),
            delta_time_mlp: TimestepEmbedding::new(hidden, hidden, None, device),
            decoder: MiniCpmModel::new(config, device),
            in_channels,
        }
    }

    /// * `x`: `[N, C, T]` noisy sample.
    /// * `mu`: `[N, C_mu]` (will be reshaped to `[N, C_mu/H, H]`).
    /// * `t`, `dt`: `[N]`.
    /// * `cond`: `[N, C, T']`.
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        mu: Tensor<B, 2>,
        t: Tensor<B, 1>,
        cond: Tensor<B, 3>,
        dt: Tensor<B, 1>,
    ) -> Tensor<B, 3> {
        let [n, _c, time_len] = x.dims();
        let x = self.in_proj.forward(x.swap_dims(1, 2));
        let hidden = x.dims()[2];

        let cond = self.cond_proj.forward(cond.swap_dims(1, 2));
        let prefix = cond.dims()[1];

        let t_emb = self.time_embeddings.forward(t, 1000.0);
        let t_emb = self.time_mlp.forward(t_emb);
        let dt_emb = self.time_embeddings.forward(dt, 1000.0);
        let dt_emb = self.delta_time_mlp.forward(dt_emb);
        let t_emb = t_emb + dt_emb;

        let mu_tokens = mu.dims()[1] / hidden;
        let mu = mu.reshape([n, mu_tokens, hidden]);

        let t_tok: Tensor<B, 3> = t_emb.unsqueeze_dim(1);

        let seq = Tensor::cat(vec![mu, t_tok, cond, x], 1);
        let (hidden_out, _) = self.decoder.forward(seq, false);

        let skip = prefix + mu_tokens + 1;
        let hidden_out = hidden_out.narrow(1, skip, time_len);
        let out = self.out_proj.forward(hidden_out);
        out.swap_dims(1, 2)
    }
}
