use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::nn::{GroupNorm, GroupNormConfig, LayerNorm, LayerNormConfig, PaddingConfig1d, Linear, LinearConfig};
use burn::prelude::*;

// ---------------------------------------------------------------------------
// HuBERT Feature Extractor Layer 0
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct HubertFeatureExtractorLayer0<B: Backend> {
    pub conv: Conv1d<B>,
    pub norm: GroupNorm<B>,
}

impl<B: Backend> HubertFeatureExtractorLayer0<B> {
    pub fn new(conv_dim: usize, kernel: usize, stride: usize, device: &B::Device) -> Self {
        let conv = Conv1dConfig::new(1, conv_dim, kernel)
            .with_stride(stride)
            .with_bias(false)
            .init(device);
        let norm = GroupNormConfig::new(conv_dim, conv_dim).init(device);
        Self { conv, norm }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let y = self.conv.forward(x);
        let y = self.norm.forward(y);
        burn::tensor::activation::gelu(y)
    }
}

// ---------------------------------------------------------------------------
// HuBERT Feature Extractor Layers 1..6
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct HubertFeatureExtractorLayerN<B: Backend> {
    pub conv: Conv1d<B>,
}

impl<B: Backend> HubertFeatureExtractorLayerN<B> {
    pub fn new(conv_dim: usize, kernel: usize, stride: usize, device: &B::Device) -> Self {
        let conv = Conv1dConfig::new(conv_dim, conv_dim, kernel)
            .with_stride(stride)
            .with_bias(false)
            .init(device);
        Self { conv }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let y = self.conv.forward(x);
        burn::tensor::activation::gelu(y)
    }
}

// ---------------------------------------------------------------------------
// HuBERT Feature Extractor
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct HubertFeatureExtractor<B: Backend> {
    pub layer0: HubertFeatureExtractorLayer0<B>,
    pub layers: Vec<HubertFeatureExtractorLayerN<B>>,
}

impl<B: Backend> HubertFeatureExtractor<B> {
    pub fn new(device: &B::Device) -> Self {
        let conv_dims = [512, 512, 512, 512, 512, 512, 512];
        let conv_kernels = [10, 3, 3, 3, 3, 2, 2];
        let conv_strides = [5, 2, 2, 2, 2, 2, 2];

        let layer0 = HubertFeatureExtractorLayer0::new(conv_dims[0], conv_kernels[0], conv_strides[0], device);
        let mut layers = Vec::with_capacity(6);
        for i in 1..7 {
            layers.push(HubertFeatureExtractorLayerN::new(conv_dims[i], conv_kernels[i], conv_strides[i], device));
        }

        Self { layer0, layers }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let mut y = self.layer0.forward(x);
        for layer in &self.layers {
            y = layer.forward(y);
        }
        y
    }
}

// ---------------------------------------------------------------------------
// HuBERT Feature Projection
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct HubertFeatureProjection<B: Backend> {
    pub layer_norm: LayerNorm<B>,
    pub projection: Linear<B>,
}

impl<B: Backend> HubertFeatureProjection<B> {
    pub fn new(in_dim: usize, out_dim: usize, device: &B::Device) -> Self {
        let layer_norm = LayerNormConfig::new(in_dim).init(device);
        let projection = LinearConfig::new(in_dim, out_dim).init(device);
        Self { layer_norm, projection }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let y = self.layer_norm.forward(x);
        self.projection.forward(y)
    }
}

// ---------------------------------------------------------------------------
// HuBERT Positional Conv Embedding
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct HubertPositionalConvEmbedding<B: Backend> {
    pub conv: Conv1d<B>,
}

impl<B: Backend> HubertPositionalConvEmbedding<B> {
    pub fn new(hidden_size: usize, kernel_size: usize, groups: usize, device: &B::Device) -> Self {
        let conv = Conv1dConfig::new(hidden_size, hidden_size, kernel_size)
            .with_padding(PaddingConfig1d::Explicit(kernel_size / 2))
            .with_groups(groups)
            .init(device);
        Self { conv }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // Input: [B, L, C] -> Transpose to [B, C, L]
        let x_t = x.swap_dims(1, 2);
        let y = self.conv.forward(x_t);
        // Remove 1 padding element at the end because kernel is even (128)
        let len = y.dims()[2];
        let y = y.narrow(2, 0, len - 1);
        let y = burn::tensor::activation::gelu(y);
        y.swap_dims(1, 2) // [B, L, C]
     }
}

// ---------------------------------------------------------------------------
// HuBERT Attention
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct HubertAttention<B: Backend> {
    pub q_proj: Linear<B>,
    pub k_proj: Linear<B>,
    pub v_proj: Linear<B>,
    pub out_proj: Linear<B>,
    pub num_heads: usize,
    pub head_dim: usize,
    pub scale: f64,
}

impl<B: Backend> HubertAttention<B> {
    pub fn new(embed_dim: usize, num_heads: usize, device: &B::Device) -> Self {
        let head_dim = embed_dim / num_heads;
        let scale = 1.0 / (head_dim as f64).sqrt();
        Self {
            q_proj: LinearConfig::new(embed_dim, embed_dim).init(device),
            k_proj: LinearConfig::new(embed_dim, embed_dim).init(device),
            v_proj: LinearConfig::new(embed_dim, embed_dim).init(device),
            out_proj: LinearConfig::new(embed_dim, embed_dim).init(device),
            num_heads,
            head_dim,
            scale,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, l, c] = x.dims();
        let q = self.q_proj.forward(x.clone());
        let k = self.k_proj.forward(x.clone());
        let v = self.v_proj.forward(x);

        let q = q.reshape([b, l, self.num_heads, self.head_dim]).swap_dims(1, 2); // [B, H, L, D]
        let k = k.reshape([b, l, self.num_heads, self.head_dim]).swap_dims(1, 2); // [B, H, L, D]
        let v = v.reshape([b, l, self.num_heads, self.head_dim]).swap_dims(1, 2); // [B, H, L, D]

        let scores = q.matmul(k.swap_dims(2, 3)).mul_scalar(self.scale); // [B, H, L, L]
        let attn = burn::tensor::activation::softmax(scores, 3);
        let out = attn.matmul(v); // [B, H, L, D]
        let out = out.swap_dims(1, 2).reshape([b, l, c]); // [B, L, C]
        self.out_proj.forward(out)
    }
}

// ---------------------------------------------------------------------------
// HuBERT FeedForward
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct HubertFeedForward<B: Backend> {
    pub intermediate_dense: Linear<B>,
    pub output_dense: Linear<B>,
}

impl<B: Backend> HubertFeedForward<B> {
    pub fn new(hidden_size: usize, intermediate_size: usize, device: &B::Device) -> Self {
        Self {
            intermediate_dense: LinearConfig::new(hidden_size, intermediate_size).init(device),
            output_dense: LinearConfig::new(intermediate_size, hidden_size).init(device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let y = self.intermediate_dense.forward(x);
        let y = burn::tensor::activation::gelu(y);
        self.output_dense.forward(y)
    }
}

// ---------------------------------------------------------------------------
// HuBERT Encoder Layer
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct HubertEncoderLayer<B: Backend> {
    pub attention: HubertAttention<B>,
    pub layer_norm: LayerNorm<B>,
    pub feed_forward: HubertFeedForward<B>,
    pub final_layer_norm: LayerNorm<B>,
}

impl<B: Backend> HubertEncoderLayer<B> {
    pub fn new(hidden_size: usize, num_heads: usize, intermediate_size: usize, device: &B::Device) -> Self {
        Self {
            attention: HubertAttention::<B>::new(hidden_size, num_heads, device),
            layer_norm: LayerNormConfig::new(hidden_size).init(device),
            feed_forward: HubertFeedForward::<B>::new(hidden_size, intermediate_size, device),
            final_layer_norm: LayerNormConfig::new(hidden_size).init(device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let y = x.clone() + self.attention.forward(x);
        let y = self.layer_norm.forward(y);
        let y = y.clone() + self.feed_forward.forward(y);
        self.final_layer_norm.forward(y)
    }
}

// ---------------------------------------------------------------------------
// HuBERT Encoder
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct HubertEncoder<B: Backend> {
    pub pos_conv_embed: HubertPositionalConvEmbedding<B>,
    pub layer_norm: LayerNorm<B>,
    pub layers: Vec<HubertEncoderLayer<B>>,
}

impl<B: Backend> HubertEncoder<B> {
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        intermediate_size: usize,
        num_layers: usize,
        device: &B::Device,
    ) -> Self {
        let layers = (0..num_layers)
            .map(|_| HubertEncoderLayer::<B>::new(hidden_size, num_heads, intermediate_size, device))
            .collect();

        Self {
            pos_conv_embed: HubertPositionalConvEmbedding::<B>::new(hidden_size, 128, 16, device),
            layer_norm: LayerNormConfig::new(hidden_size).init(device),
            layers,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> (Tensor<B, 3>, Vec<Tensor<B, 3>>) {
        let pos = self.pos_conv_embed.forward(x.clone());
        let mut hidden = x + pos;
        hidden = self.layer_norm.forward(hidden);

        let mut all_hidden = Vec::with_capacity(self.layers.len() + 1);
        all_hidden.push(hidden.clone());

        for layer in &self.layers {
            hidden = layer.forward(hidden);
            all_hidden.push(hidden.clone());
        }

        (hidden, all_hidden)
    }
}

// ---------------------------------------------------------------------------
// HuBERT Model
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct HubertModel<B: Backend> {
    pub feature_extractor: HubertFeatureExtractor<B>,
    pub feature_projection: HubertFeatureProjection<B>,
    pub encoder: HubertEncoder<B>,
}

impl<B: Backend> HubertModel<B> {
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        intermediate_size: usize,
        num_layers: usize,
        device: &B::Device,
    ) -> Self {
        Self {
            feature_extractor: HubertFeatureExtractor::<B>::new(device),
            feature_projection: HubertFeatureProjection::<B>::new(512, hidden_size, device),
            encoder: HubertEncoder::<B>::new(hidden_size, num_heads, intermediate_size, num_layers, device),
        }
    }

    pub fn forward(&self, input_values: Tensor<B, 3>) -> Vec<Tensor<B, 3>> {
        let feats = self.feature_extractor.forward(input_values);
        let feats_t = feats.swap_dims(1, 2);
        let projected = self.feature_projection.forward(feats_t);
        let (_, all_hidden) = self.encoder.forward(projected);
        all_hidden
    }
}
