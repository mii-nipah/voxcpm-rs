use crate::config::MiniCpm4Config;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OmniVoiceConfig {
    pub audio_vocab_size: usize,
    pub audio_mask_id: usize,
    pub num_audio_codebook: usize,
    pub audio_codebook_weights: Vec<f32>,
    pub llm_config: MiniCpm4Config,
}

#[derive(Clone, Debug)]
pub struct OmniVoiceGenerationConfig {
    pub num_step: usize,
    pub guidance_scale: f32,
    pub t_shift: f32,
    pub layer_penalty_factor: f32,
    pub position_temperature: f32,
    pub class_temperature: f32,
    pub denoise: bool,
    pub preprocess_prompt: bool,
    pub postprocess_output: bool,
}

impl Default for OmniVoiceGenerationConfig {
    fn default() -> Self {
        Self {
            num_step: 32,
            guidance_scale: 2.0,
            t_shift: 0.1,
            layer_penalty_factor: 5.0,
            position_temperature: 5.0,
            class_temperature: 0.0,
            denoise: true,
            preprocess_prompt: true,
            postprocess_output: true,
        }
    }
}

impl OmniVoiceGenerationConfig {
    pub fn builder() -> OmniVoiceGenerationConfigBuilder {
        OmniVoiceGenerationConfigBuilder::default()
    }
}

#[derive(Clone, Debug, Default)]
pub struct OmniVoiceGenerationConfigBuilder {
    inner: OmniVoiceGenerationConfig,
}

impl OmniVoiceGenerationConfigBuilder {
    pub fn num_step(mut self, n: usize) -> Self {
        self.inner.num_step = n;
        self
    }

    pub fn guidance_scale(mut self, scale: f32) -> Self {
        self.inner.guidance_scale = scale;
        self
    }

    pub fn t_shift(mut self, t: f32) -> Self {
        self.inner.t_shift = t;
        self
    }

    pub fn layer_penalty_factor(mut self, factor: f32) -> Self {
        self.inner.layer_penalty_factor = factor;
        self
    }

    pub fn position_temperature(mut self, temp: f32) -> Self {
        self.inner.position_temperature = temp;
        self
    }

    pub fn class_temperature(mut self, temp: f32) -> Self {
        self.inner.class_temperature = temp;
        self
    }

    pub fn denoise(mut self, enable: bool) -> Self {
        self.inner.denoise = enable;
        self
    }

    pub fn preprocess_prompt(mut self, enable: bool) -> Self {
        self.inner.preprocess_prompt = enable;
        self
    }

    pub fn postprocess_output(mut self, enable: bool) -> Self {
        self.inner.postprocess_output = enable;
        self
    }

    pub fn build(self) -> OmniVoiceGenerationConfig {
        self.inner
    }
}
