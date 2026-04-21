//! Full MiniCPM-4 model: optional token embedding, N decoder layers, final RMSNorm.

use crate::config::MiniCpm4Config;
use crate::minicpm4::attention::LayerKv;
use crate::minicpm4::cache::StaticKvCache;
use crate::minicpm4::layer::MiniCpmDecoderLayer;
use crate::minicpm4::rope::MiniCpmLongRope;
use crate::minicpm4::MiniCpmRmsNorm;
use burn::module::Ignored;
use burn::nn::{Embedding, EmbeddingConfig};
use burn::prelude::*;
use burn::tensor::Int;

#[derive(Module, Debug)]
pub struct MiniCpmModel<B: Backend> {
    /// Token embedding table. `None` when `vocab_size == 0` (e.g. local DiT).
    pub embed_tokens: Option<Embedding<B>>,
    pub layers: Vec<MiniCpmDecoderLayer<B>>,
    pub norm: MiniCpmRmsNorm<B>,
    /// Optional rotary embedding cache (skipped when `no_rope=true`).
    pub rope: Option<MiniCpmLongRope<B>>,
    pub config: Ignored<MiniCpm4Config>,
}

impl<B: Backend> MiniCpmModel<B> {
    pub fn new(config: MiniCpm4Config, device: &B::Device) -> Self {
        let embed_tokens = (config.vocab_size > 0)
            .then(|| EmbeddingConfig::new(config.vocab_size, config.hidden_size).init(device));
        let layers = (0..config.num_hidden_layers)
            .map(|_| MiniCpmDecoderLayer::new(&config, device))
            .collect();
        let norm = MiniCpmRmsNorm::new(config.hidden_size, config.rms_norm_eps as f64, device);
        let rope = (!config.no_rope).then(|| MiniCpmLongRope::new(&config, device));
        Self {
            embed_tokens,
            layers,
            norm,
            rope,
            config: Ignored(config),
        }
    }

    pub fn forward(
        &self,
        inputs_embeds: Tensor<B, 3>,
        is_causal: bool,
    ) -> (Tensor<B, 3>, Vec<LayerKv<B>>) {
        let s = inputs_embeds.dims()[1];
        let position_emb = self.rope.as_ref().map(|r| {
            let ids = Tensor::<B, 1, Int>::arange(0..s as i64, &inputs_embeds.device());
            r.gather(ids)
        });

        let mut hidden = inputs_embeds;
        let mut caches = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            let (h, kv) = layer.forward(hidden, position_emb.clone(), is_causal);
            hidden = h;
            caches.push(kv);
        }
        (self.norm.forward(hidden), caches)
    }

    pub fn forward_step(
        &self,
        inputs_embeds: Tensor<B, 2>,
        position_id: usize,
        cache: &mut StaticKvCache<B>,
    ) -> Tensor<B, 2> {
        let position_emb = self.rope.as_ref().map(|r| {
            let ids = Tensor::<B, 1, Int>::arange(
                position_id as i64..(position_id + 1) as i64,
                &inputs_embeds.device(),
            );
            r.gather(ids)
        });

        let mut hidden = inputs_embeds;
        for (i, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward_step(hidden, position_emb.clone(), position_id, cache.layer_mut(i));
        }
        // Apply final norm (broadcast over the singleton time dim).
        self.norm.forward(hidden)
    }

    /// Embed token ids; panics when this model was built without a vocab.
    pub fn embed(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.embed_tokens
            .as_ref()
            .expect("embed called on a MiniCpmModel without an embedding table")
            .forward(tokens)
    }

    pub fn scale_emb(&self) -> f64 {
        if self.config.0.use_mup {
            self.config.0.scale_emb as f64
        } else {
            1.0
        }
    }
}
