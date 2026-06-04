use crate::tokenizer::TextTokenizer;
use crate::omnivoice::config::{OmniVoiceConfig, OmniVoiceGenerationConfig};
use crate::omnivoice::duration::RuleDurationEstimator;
use crate::omnivoice::model::OmniVoiceModel;
use crate::higgs::model::HiggsTokenizer;
use crate::voxcpm2::wrapper::{PromptAudio, CancelToken};
use crate::{Error, Result};

use burn::prelude::*;
use burn::tensor::TensorData;
use std::path::Path;

/// Options for OmniVoice speech generation.
#[derive(Debug, Clone)]
pub struct OmniVoiceOptions {
    /// Classifier-free guidance scale. Higher = closer to conditioning. Default: 2.0.
    pub guidance_scale: f32,
    /// Number of iterative decoding steps. Default: 32.
    pub num_step: usize,
    /// Style instruction for voice design (e.g. "whispering", "excited").
    pub instruct: Option<String>,
    /// Language code (always None in Rust library, language is auto-detected).
    pub lang: Option<String>,
    /// Prompt conditioning. See [`OmniVoicePrompt`].
    pub prompt: OmniVoicePrompt,
    /// Speaking speed factor. > 1.0 is faster, < 1.0 is slower.
    pub speed: f32,
    /// Fixed output duration in seconds. Overrides speed if both are provided.
    pub duration: Option<f32>,
    /// Cooperative cancellation token.
    pub cancel: Option<CancelToken>,
}

impl Default for OmniVoiceOptions {
    fn default() -> Self {
        Self {
            guidance_scale: 2.0,
            num_step: 32,
            instruct: None,
            lang: None,
            prompt: OmniVoicePrompt::None,
            speed: 1.0,
            duration: None,
            cancel: None,
        }
    }
}

impl OmniVoiceOptions {
    pub fn builder() -> OmniVoiceOptionsBuilder {
        OmniVoiceOptionsBuilder::default()
    }
}

#[derive(Debug, Clone, Default)]
pub struct OmniVoiceOptionsBuilder {
    inner: OmniVoiceOptions,
}

impl OmniVoiceOptionsBuilder {
    pub fn guidance_scale(mut self, scale: f32) -> Self {
        self.inner.guidance_scale = scale;
        self
    }

    pub fn num_step(mut self, steps: usize) -> Self {
        self.inner.num_step = steps;
        self
    }

    pub fn instruct(mut self, instr: String) -> Self {
        self.inner.instruct = Some(instr);
        self
    }

    pub fn lang(mut self, l: String) -> Self {
        self.inner.lang = Some(l);
        self
    }

    pub fn prompt(mut self, p: OmniVoicePrompt) -> Self {
        self.inner.prompt = p;
        self
    }

    pub fn speed(mut self, s: f32) -> Self {
        self.inner.speed = s;
        self
    }

    pub fn duration(mut self, d: f32) -> Self {
        self.inner.duration = Some(d);
        self
    }

    pub fn cancel(mut self, c: CancelToken) -> Self {
        self.inner.cancel = Some(c);
        self
    }

    pub fn build(self) -> OmniVoiceOptions {
        self.inner
    }
}

/// Prompt conditioning for OmniVoice.
#[derive(Debug, Clone, Default)]
pub enum OmniVoicePrompt {
    #[default]
    None,
    Clone {
        audio: PromptAudio,
        text: String,
    },
}

#[derive(Debug)]
pub struct OmniVoice<B: Backend> {
    pub model: OmniVoiceModel<B>,
    pub audio_tokenizer: HiggsTokenizer<B>,
    pub text_tokenizer: TextTokenizer,
    pub duration_estimator: RuleDurationEstimator,
    device: B::Device,
}

impl<B: Backend> OmniVoice<B> {
    /// Load a pretrained OmniVoice checkpoint from a local directory.
    pub fn from_local(path: impl AsRef<Path>, device: &B::Device) -> Result<Self> {
        let path = path.as_ref();
        
        // 1. Load configs
        let config_bytes = std::fs::read_to_string(path.join("config.json"))?;
        let config: OmniVoiceConfig = serde_json::from_str(&config_bytes)?;

        let tokenizer_config_bytes = std::fs::read_to_string(path.join("audio_tokenizer/config.json"))?;
        let tokenizer_config: crate::higgs::config::HiggsConfig = serde_json::from_str(&tokenizer_config_bytes)?;

        // 2. Load tokenizers
        let text_tokenizer = TextTokenizer::from_local(path)?;
        let audio_tokenizer = HiggsTokenizer::new(tokenizer_config, device);
        let duration_estimator = RuleDurationEstimator::new();

        // 3. Create model
        let mut model = OmniVoiceModel::new(config, device);

        // 4. Load weights
        let main_result = crate::weights::load_omnivoice(&mut model, path)?;
        log::info!(
            "OmniVoice main weights loaded — applied={}, skipped={}, missing={}, unused={}",
            main_result.applied.len(),
            main_result.skipped.len(),
            main_result.missing.len(),
            main_result.unused.len(),
        );

        let mut wrapped_tokenizer = audio_tokenizer;
        let tok_result = crate::weights::load_higgs_tokenizer(&mut wrapped_tokenizer, &path.join("audio_tokenizer"))?;
        log::info!(
            "Higgs audio tokenizer weights loaded — applied={}, skipped={}, missing={}, unused={}",
            tok_result.applied.len(),
            tok_result.skipped.len(),
            tok_result.missing.len(),
            tok_result.unused.len(),
        );
        if !tok_result.missing.is_empty() {
            log::warn!("Higgs tokenizer MISSING keys: {:?}", tok_result.missing);
        }
        if !tok_result.unused.is_empty() {
            log::warn!("Higgs tokenizer UNUSED keys: {:?}", tok_result.unused);
        }

        Ok(Self {
            model,
            audio_tokenizer: wrapped_tokenizer,
            text_tokenizer,
            duration_estimator,
            device: device.clone(),
        })
    }

    pub fn sample_rate(&self) -> u32 {
        24000
    }

    pub fn device(&self) -> &B::Device {
        &self.device
    }

    /// Generate audio for the given text.
    pub fn generate(&self, text: &str, opts: OmniVoiceOptions) -> Result<Vec<f32>> {
        // Evaluate input speed/duration and estimate output tokens
        let mut ref_text_opt = None;
        let mut ref_audio_tokens_opt = None;
        let mut ref_rms_opt = None;
        let mut ref_duration_opt = None;

        if let OmniVoicePrompt::Clone { audio, text: ref_text } = &opts.prompt {
            let mut pcm_24k = load_prompt_audio(audio, 24000)?;

            // Calculate RMS on CPU before clipping
            let sum_sq: f32 = pcm_24k.iter().map(|x| x * x).sum();
            let ref_rms = (sum_sq / pcm_24k.len().max(1) as f32).sqrt();
            ref_rms_opt = Some(ref_rms);

            // Clip to hop_length boundary
            let hop = self.audio_tokenizer.hop_length;
            let clip_size = pcm_24k.len() % hop;
            if clip_size > 0 && pcm_24k.len() > clip_size {
                pcm_24k.truncate(pcm_24k.len() - clip_size);
            }

            let len_24k = pcm_24k.len();
            let ref_len_tokens = len_24k / hop;
            ref_duration_opt = Some(ref_len_tokens);

            // Run audio tokenizer encode
            let acoustic_tensor = Tensor::<B, 3>::from_data(
                TensorData::new(pcm_24k, [1, 1, len_24k]),
                &self.device,
            );

            let tokens = self.audio_tokenizer.encode(acoustic_tensor, None); // [1, C, T]
            ref_audio_tokens_opt = Some(tokens);
            ref_text_opt = Some(ref_text.clone());
        }

        // Estimate target duration
        let target_len = if let Some(fixed_duration) = opts.duration {
            (fixed_duration * 25.0) as usize
        } else {
            let num_ref_tokens = ref_duration_opt;
            let est_tokens = crate::omnivoice::model::estimate_target_tokens(
                &self.duration_estimator,
                text,
                ref_text_opt.as_deref(),
                num_ref_tokens,
                opts.speed,
            );
            est_tokens
        };

        // Construct OmniVoiceGenerationConfig
        let gen_config = OmniVoiceGenerationConfig::builder()
            .num_step(opts.num_step)
            .guidance_scale(opts.guidance_scale)
            .build();

        // Run unmasking loop
        let results = self.model.generate_iterative(
            vec![text.to_string()],
            vec![target_len],
            vec![ref_text_opt.unwrap_or_default()],
            vec![ref_audio_tokens_opt],
            vec![opts.lang],
            vec![opts.instruct],
            &gen_config,
            &self.text_tokenizer,
            opts.cancel.as_ref(),
        )?;

        // Decode generated audio
        let tokens = results.into_iter().next().ok_or_else(|| Error::Other("generation returned no results".to_string()))?;
        let decoded = self.audio_tokenizer.decode(tokens); // [1, 1, T]
        let decoded_data = decoded.into_data().iter::<f32>().collect::<Vec<_>>();

        // Post-process audio
        let mut processed = remove_silence(&decoded_data, 24000, 500, 100, 100);

        // Normalize volume
        if let Some(ref_rms) = ref_rms_opt {
            if ref_rms < 0.1 {
                for sample in &mut processed {
                    *sample = *sample * ref_rms / 0.1;
                }
            }
        } else {
            let mut peak = 0.0f32;
            for &sample in &processed {
                let abs = sample.abs();
                if abs > peak {
                    peak = abs;
                }
            }
            if peak > 1e-6 {
                for sample in &mut processed {
                    *sample = *sample / peak * 0.5;
                }
            }
        }

        processed = fade_and_pad_audio(&processed, 24000, 0.1, 0.1);

        Ok(processed)
    }
}

fn load_prompt_audio(audio: &PromptAudio, target_sr: u32) -> Result<Vec<f32>> {
    match audio {
        PromptAudio::File(p) => crate::audio::load_audio_as(p, target_sr),
        PromptAudio::Encoded(bytes) => crate::audio::load_audio_bytes_as(bytes, target_sr),
        PromptAudio::Pcm { samples, sample_rate } => {
            crate::audio::resample(samples, *sample_rate, target_sr)
        }
    }
}

fn remove_silence(
    samples: &[f32],
    sample_rate: u32,
    mid_sil_ms: u32,
    lead_sil_ms: u32,
    trail_sil_ms: u32,
) -> Vec<f32> {
    if samples.is_empty() {
        return Vec::new();
    }

    let threshold = 0.003162; // -50 dB
    let lead_sil_samples = (lead_sil_ms as f32 * sample_rate as f32 / 1000.0) as usize;
    let trail_sil_samples = (trail_sil_ms as f32 * sample_rate as f32 / 1000.0) as usize;

    let mut start_idx = 0;
    while start_idx < samples.len() && samples[start_idx].abs() < threshold {
        start_idx += 1;
    }
    start_idx = start_idx.saturating_sub(lead_sil_samples);

    let mut end_idx = samples.len();
    while end_idx > start_idx && samples[end_idx - 1].abs() < threshold {
        end_idx -= 1;
    }
    end_idx = (end_idx + trail_sil_samples).min(samples.len());

    let trimmed = &samples[start_idx..end_idx];
    if trimmed.is_empty() {
        return Vec::new();
    }

    if mid_sil_ms == 0 {
        return trimmed.to_vec();
    }

    let mid_sil_samples = (mid_sil_ms as f32 * sample_rate as f32 / 1000.0) as usize;
    let mut result = Vec::with_capacity(trimmed.len());

    let mut i = 0;
    while i < trimmed.len() {
        if trimmed[i].abs() < threshold {
            let mut silence_len = 0;
            while i + silence_len < trimmed.len() && trimmed[i + silence_len].abs() < threshold {
                silence_len += 1;
            }
            if silence_len > mid_sil_samples {
                let half_keep = mid_sil_samples / 2;
                result.extend(&trimmed[i..i + half_keep]);
                result.extend(&trimmed[i + silence_len - half_keep..i + silence_len]);
            } else {
                result.extend(&trimmed[i..i + silence_len]);
            }
            i += silence_len;
        } else {
            result.push(trimmed[i]);
            i += 1;
        }
    }
    result
}

fn fade_and_pad_audio(
    samples: &[f32],
    sample_rate: u32,
    pad_duration: f32,
    fade_duration: f32,
) -> Vec<f32> {
    if samples.is_empty() {
        return Vec::new();
    }

    let fade_samples = (fade_duration * sample_rate as f32) as usize;
    let pad_samples = (pad_duration * sample_rate as f32) as usize;

    let mut processed = samples.to_vec();
    let n = processed.len();

    if fade_samples > 0 {
        let k = fade_samples.min(n / 2);
        if k > 0 {
            for i in 0..k {
                let ratio = i as f32 / k as f32;
                processed[i] *= ratio;
            }
            for i in 0..k {
                let ratio = (k - 1 - i) as f32 / k as f32;
                processed[n - k + i] *= ratio;
            }
        }
    }

    if pad_samples > 0 {
        let mut padded = vec![0.0f32; pad_samples];
        padded.extend_from_slice(&processed);
        padded.resize(padded.len() + pad_samples, 0.0f32);
        padded
    } else {
        processed
    }
}
