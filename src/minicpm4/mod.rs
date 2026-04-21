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
        let variance = x.clone().powf_scalar(2.0).mean_dim(D - 1);
        let x = x * variance.add_scalar(self.eps).powf_scalar(-0.5);
        x * self.weight.val().unsqueeze()
    }
}
