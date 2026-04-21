//! High-level convenience wrapper around [`VoxCpm2Model`] that exposes a
//! Python-SDK-style `generate()` API.

use crate::tokenizer::TextTokenizer;
use crate::voxcpm2::model::{AUDIO_START_TOKEN, VoxCpm2Model};
use crate::VoxCpm2Config;
use burn::prelude::*;
use burn::tensor::{Int, TensorData};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct GenerateOptions {
    pub cfg_value: f32,
    pub inference_timesteps: usize,
    pub min_len: usize,
    pub max_len: usize,
    pub prompt_wav: Option<PathBuf>,
    pub prompt_text: Option<String>,
    pub reference_wav: Option<PathBuf>,
}

impl Default for GenerateOptions {
    fn default() -> Self {
        Self {
            cfg_value: 2.0,
            inference_timesteps: 10,
            min_len: 2,
            max_len: 2000,
            prompt_wav: None,
            prompt_text: None,
            reference_wav: None,
        }
    }
}

/// Cached reference/prompt features for repeated generation.
///
/// Storage-only for now: `generate` currently only supports zero-shot mode.
#[derive(Debug, Clone, Default)]
pub struct PromptCache {
    pub prompt_text: Option<String>,
    pub prompt_wav: Option<PathBuf>,
    pub reference_wav: Option<PathBuf>,
}

#[derive(Debug)]
pub struct VoxCPM<B: Backend> {
    pub model: VoxCpm2Model<B>,
    pub tokenizer: TextTokenizer,
    device: B::Device,
}

impl<B: Backend> VoxCPM<B> {
    /// Construct a fresh (randomly-initialized) model from a
    /// [`VoxCpm2Config`][crate::VoxCpm2Config]. For inference you almost
    /// certainly want [`Self::from_local`] instead.
    pub fn from_config(
        config: VoxCpm2Config,
        tokenizer: TextTokenizer,
        device: &B::Device,
    ) -> Self {
        Self {
            model: VoxCpm2Model::new(config, device),
            tokenizer,
            device: device.clone(),
        }
    }

    /// Load a pretrained VoxCPM2 checkpoint from a local directory.
    ///
    /// The directory is expected to contain:
    /// - `config.json` — a [`VoxCpm2Config`] JSON.
    /// - `tokenizer.json` — a HuggingFace `tokenizers` file.
    /// - `model.safetensors` — the main model weights.
    /// - `audiovae.safetensors` — the AudioVAE weights.
    pub fn from_local(path: impl AsRef<Path>, device: &B::Device) -> crate::Result<Self> {
        let path = path.as_ref();
        let config_bytes = std::fs::read_to_string(path.join("config.json"))?;
        let config: VoxCpm2Config = serde_json::from_str(&config_bytes)?;
        let tokenizer = TextTokenizer::from_local(path)?;
        let mut model = VoxCpm2Model::<B>::new(config, device);
        let result = crate::weights::load_pretrained(&mut model, path)?;
        eprintln!(
            "voxcpm-rs: weights loaded — applied={}, skipped={}, missing={}, unused={}, errors={}",
            result.applied.len(),
            result.skipped.len(),
            result.missing.len(),
            result.unused.len(),
            result.errors.len(),
        );
        if !result.missing.is_empty() {
            eprintln!("voxcpm-rs: missing module params (first 20):");
            for (k, ctx) in result.missing.iter().take(20) {
                eprintln!("  {k} [{ctx}]");
            }
        }
        if !result.unused.is_empty() {
            eprintln!("voxcpm-rs: unused checkpoint tensors (first 20):");
            for k in result.unused.iter().take(20) {
                eprintln!("  {k}");
            }
        }
        if !result.errors.is_empty() {
            eprintln!("voxcpm-rs: load errors (first 20):");
            for e in result.errors.iter().take(20) {
                eprintln!("  {e:?}");
            }
        }
        Ok(Self {
            model,
            tokenizer,
            device: device.clone(),
        })
    }

    pub fn sample_rate(&self) -> u32 {
        self.model.sample_rate() as u32
    }

    /// Generate an audio waveform (mono `f32` samples at [`Self::sample_rate`]).
    ///
    /// Currently zero-shot only: `opts.prompt_wav` / `opts.reference_wav` are
    /// ignored and a warning-style error is returned if they're set.
    pub fn generate(&self, text: &str, opts: GenerateOptions) -> crate::Result<Vec<f32>> {
        if opts.prompt_wav.is_some() || opts.reference_wav.is_some() {
            return Err(crate::Error::Unsupported(
                "prompt / reference audio paths are not yet supported by VoxCPM::generate".into(),
            ));
        }

        // 1) Tokenize + append audio start.
        let mut tokens = self.tokenizer.encode(text)?;
        tokens.push(AUDIO_START_TOKEN);
        let s = tokens.len();

        let device = &self.device;

        // 2) Build input tensors for zero-shot mode: all positions are text.
        let text_token: Tensor<B, 2, Int> = Tensor::from_data(
            TensorData::new(tokens.clone(), [1, s]),
            device,
        );
        let text_mask_vals: Vec<f32> = vec![1.0; s];
        let text_mask: Tensor<B, 2> =
            Tensor::from_data(TensorData::new(text_mask_vals, [1, s]), device);
        let feat_mask_vals: Vec<f32> = vec![0.0; s];
        let feat_mask: Tensor<B, 2> =
            Tensor::from_data(TensorData::new(feat_mask_vals, [1, s]), device);

        let p = self.model.patch_size();
        let d = self.model.latent_dim();
        let feat: Tensor<B, 4> = Tensor::zeros([1, s, p, d], device);

        // 3) Run the main inference loop: latent patches → [B, D, T*P].
        let latent = self.model.inference(
            text_token,
            text_mask,
            feat,
            feat_mask,
            opts.min_len,
            opts.max_len,
            opts.inference_timesteps,
            opts.cfg_value as f64,
        );

        // 4) VAE decode → waveform [B, 1, T*P*decoder_stride].
        let wav = self.model.audio_vae.decode(latent);
        let wav = wav.squeeze_dim::<2>(1); // [B, T_out]
        let wav = wav.squeeze_dim::<1>(0); // [T_out]
        let data = wav.into_data();
        let samples: Vec<f32> = data
            .as_slice::<f32>()
            .map_err(|_| crate::Error::Other("unexpected VAE output dtype".into()))?
            .to_vec();
        Ok(samples)
    }
}

