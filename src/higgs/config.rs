use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DacConfig {
    #[serde(default = "default_encoder_hidden_size")]
    pub encoder_hidden_size: usize,
    #[serde(default = "default_downsampling_ratios")]
    pub downsampling_ratios: Vec<usize>,
    #[serde(default = "default_decoder_hidden_size")]
    pub decoder_hidden_size: usize,
    #[serde(default = "default_upsampling_ratios")]
    pub upsampling_ratios: Vec<usize>,
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
}

fn default_encoder_hidden_size() -> usize { 64 }
fn default_downsampling_ratios() -> Vec<usize> { vec![8, 5, 4, 2, 3] }
fn default_decoder_hidden_size() -> usize { 1024 }
fn default_upsampling_ratios() -> Vec<usize> { vec![8, 5, 4, 2, 3] }
fn default_hidden_size() -> usize { 256 }

impl Default for DacConfig {
    fn default() -> Self {
        Self {
            encoder_hidden_size: 64,
            downsampling_ratios: vec![8, 5, 4, 2, 3],
            decoder_hidden_size: 1024,
            upsampling_ratios: vec![8, 5, 4, 2, 3],
            hidden_size: 256,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HubertConfig {
    #[serde(default = "default_hubert_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_num_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: usize,
}

fn default_hubert_hidden_size() -> usize { 768 }
fn default_num_hidden_layers() -> usize { 12 }
fn default_num_attention_heads() -> usize { 12 }
fn default_intermediate_size() -> usize { 3072 }

impl Default for HubertConfig {
    fn default() -> Self {
        Self {
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HiggsConfig {
    #[serde(default = "default_target_bandwidths")]
    pub target_bandwidths: Vec<f32>,
    #[serde(default = "default_sample_rate")]
    pub sample_rate: usize,
    #[serde(default = "default_kernel_size")]
    pub kernel_size: usize,
    #[serde(default = "default_channel_ratios")]
    pub channel_ratios: Vec<f32>,
    #[serde(default = "default_strides")]
    pub strides: Vec<usize>,
    #[serde(default = "default_block_dilations")]
    pub block_dilations: Vec<usize>,
    #[serde(default = "default_unit_kernel_size")]
    pub unit_kernel_size: usize,
    #[serde(default = "default_codebook_size")]
    pub codebook_size: usize,
    #[serde(default = "default_codebook_dim")]
    pub codebook_dim: usize,
    #[serde(default = "default_semantic_sample_rate")]
    pub semantic_sample_rate: usize,
    #[serde(default = "default_downsample_factor")]
    pub downsample_factor: usize,
    #[serde(default)]
    pub acoustic_model_config: DacConfig,
    #[serde(default)]
    pub semantic_model_config: HubertConfig,
}

fn default_target_bandwidths() -> Vec<f32> { vec![0.5, 1.0, 1.5, 2.0] }
fn default_sample_rate() -> usize { 24000 }
fn default_kernel_size() -> usize { 3 }
fn default_channel_ratios() -> Vec<f32> { vec![1.0, 1.0] }
fn default_strides() -> Vec<usize> { vec![1, 1] }
fn default_block_dilations() -> Vec<usize> { vec![1, 1] }
fn default_unit_kernel_size() -> usize { 3 }
fn default_codebook_size() -> usize { 1024 }
fn default_codebook_dim() -> usize { 64 }
fn default_semantic_sample_rate() -> usize { 16000 }
fn default_downsample_factor() -> usize { 320 }

impl Default for HiggsConfig {
    fn default() -> Self {
        Self {
            target_bandwidths: vec![0.5, 1.0, 1.5, 2.0],
            sample_rate: 24000,
            kernel_size: 3,
            channel_ratios: vec![1.0, 1.0],
            strides: vec![1, 1],
            block_dilations: vec![1, 1],
            unit_kernel_size: 3,
            codebook_size: 1024,
            codebook_dim: 64,
            semantic_sample_rate: 16000,
            downsample_factor: 320,
            acoustic_model_config: DacConfig::default(),
            semantic_model_config: HubertConfig::default(),
        }
    }
}
