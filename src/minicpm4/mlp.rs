//! Gated MLP used inside each MiniCPM-4 decoder layer.

use crate::config::MiniCpm4Config;
use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation::silu;

#[derive(Module, Debug)]
pub struct MiniCpmMlp<B: Backend> {
    pub gate_proj: Linear<B>,
    pub up_proj: Linear<B>,
    pub down_proj: Linear<B>,
}

impl<B: Backend> MiniCpmMlp<B> {
    pub fn new(config: &MiniCpm4Config, device: &B::Device) -> Self {
        let hidden = config.hidden_size;
        let inter = config.intermediate_size;
        Self {
            gate_proj: LinearConfig::new(hidden, inter).with_bias(false).init(device),
            up_proj: LinearConfig::new(hidden, inter).with_bias(false).init(device),
            down_proj: LinearConfig::new(inter, hidden).with_bias(false).init(device),
        }
    }

    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let gated = silu(self.gate_proj.forward(x.clone()));
        let up = self.up_proj.forward(x);
        self.down_proj.forward(gated * up)
    }
}
