//! MiniCPM-4 transformer backbone used by the text-semantic LM, the residual
//! acoustic LM, the local encoder, and the local DiT estimator.

pub mod attention;
pub mod cache;
pub mod layer;
pub mod mlp;
pub mod model;
pub mod rope;

pub use attention::MiniCpmAttention;
pub use cache::StaticKvCache;
pub use layer::MiniCpmDecoderLayer;
pub use mlp::MiniCpmMlp;
pub use model::MiniCpmModel;
pub use rope::MiniCpmLongRope;

use burn::module::Param;
use burn::prelude::*;

/// MiniCPM's RMSNorm.
#[derive(Module, Debug)]
pub struct MiniCpmRmsNorm<B: Backend> {
    pub weight: Param<Tensor<B, 1>>,
    pub eps: f64,
}

impl<B: Backend> MiniCpmRmsNorm<B> {
    pub fn new(hidden_size: usize, eps: f64, device: &B::Device) -> Self {
        Self {
            weight: Param::from_tensor(Tensor::ones([hidden_size], device)),
            eps,
        }
    }

    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        // RMS = mean(x*x) along last dim; rsqrt = 1/sqrt(...). Avoiding
        // `powf_scalar(2.0)` / `powf_scalar(-0.5)` cuts two general-pow
        // kernels per call (dominant in DiT decoder hot loop).
        let x_sq = x.clone() * x.clone();
        let variance = x_sq.mean_dim(D - 1);
        let inv_rms = variance.add_scalar(self.eps).sqrt().recip();
        let x = x * inv_rms;
        x * self.weight.val().unsqueeze()
    }
}
