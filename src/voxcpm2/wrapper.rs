//! High-level convenience wrapper around [`VoxCpm2Model`] that exposes a
//! Python-SDK-style `generate()` API.

use crate::tokenizer::TextTokenizer;
use crate::voxcpm2::model::{
    AUDIO_START_TOKEN, REF_AUDIO_END_TOKEN, REF_AUDIO_START_TOKEN, VoxCpm2Model,
};
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
/// Storage-only stub — `generate` does not yet consume these caches.
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

    /// Decode a pre-computed latent `[1, D, T]` through the AudioVAE and
    /// return the waveform tensor `[1, 1, T_out]`.
    pub fn audio_vae_decode(&self, feat: Tensor<B, 3>) -> Tensor<B, 3> {
        self.model.audio_vae.decode(feat)
    }

    /// Padding mode for prompt audio prior to VAE encoding.
    /// - `Right`: pad zeros at the end (used for reference audio in voice cloning).
    /// - `Left`: pad zeros at the start so the *valid* audio sits at the end of
    ///   the sequence (used for continuation prompts).
    fn encode_prompt_wav(
        &self,
        wav_path: &Path,
        padding_mode: PadMode,
    ) -> crate::Result<Tensor<B, 3>> {
        let encoder_sr = self.model.audio_vae.sample_rate() as u32;
        let mut samples = crate::audio::load_audio_as(wav_path, encoder_sr)?;
        let p = self.model.patch_size();
        let chunk = self.model.audio_vae.config.0.chunk_size();
        let patch_len = p * chunk;
        let n = samples.len();
        if n == 0 {
            return Err(crate::Error::AudioDecode(format!(
                "prompt audio decoded to 0 samples: {}",
                wav_path.display()
            )));
        }
        let rem = n % patch_len;
        if rem != 0 {
            let pad = patch_len - rem;
            match padding_mode {
                PadMode::Right => samples.resize(n + pad, 0.0),
                PadMode::Left => {
                    let mut new = vec![0.0f32; pad];
                    new.extend_from_slice(&samples);
                    samples = new;
                }
            }
        }
        let n_padded = samples.len();
        let audio: Tensor<B, 3> =
            Tensor::from_data(TensorData::new(samples, [1, 1, n_padded]), &self.device);
        let feat = self.model.audio_vae.encode(audio); // [1, D, T*P]
        let [_, d, tp] = feat.dims();
        debug_assert_eq!(tp % p, 0);
        let t = tp / p;
        // [1, D, T*P] -> [D, T, P] -> [T, P, D]
        let feat: Tensor<B, 3> = feat.reshape([d, t, p]);
        let feat = feat.swap_dims(0, 1).swap_dims(1, 2);
        Ok(feat)
    }

    /// Generate an audio waveform (mono `f32` samples at [`Self::sample_rate`]).
    ///
    /// Supported modes (matching upstream `VoxCPM2Model._generate`):
    /// - **Zero-shot** (default): no prompt audio, model improvises a voice.
    /// - **Reference** (`opts.reference_wav`): voice cloning via a structurally
    ///   isolated reference audio prefix (no transcript required).
    /// - **Continuation** (`opts.prompt_wav` + `opts.prompt_text`): the model
    ///   continues from the prompt audio in the same speaker's voice. The
    ///   prompt text is the transcript of the prompt audio and is prepended to
    ///   `text` before tokenization.
    /// - **Combined** (`reference_wav` + `prompt_wav` + `prompt_text`): both
    ///   the reference prefix and the continuation suffix.
    ///
    /// `prompt_wav` and `prompt_text` must be provided together.
    pub fn generate(&self, text: &str, opts: GenerateOptions) -> crate::Result<Vec<f32>> {
        if opts.prompt_wav.is_some() != opts.prompt_text.is_some() {
            return Err(crate::Error::Unsupported(
                "prompt_wav and prompt_text must be provided together (continuation mode)".into(),
            ));
        }

        let device = &self.device;
        let p = self.model.patch_size();
        let d = self.model.latent_dim();

        // 1) Tokenize text. In continuation modes prompt_text is prepended.
        let full_text: String = match opts.prompt_text.as_deref() {
            Some(pt) => format!("{pt}{text}"),
            None => text.to_string(),
        };
        let mut text_tokens = self.tokenizer.encode(&full_text)?;
        text_tokens.push(AUDIO_START_TOKEN);
        let text_len = text_tokens.len();

        // 2) Encode optional prompt audios.
        let ref_feat_opt = match opts.reference_wav.as_ref() {
            Some(path) => Some(self.encode_prompt_wav(path, PadMode::Right)?),
            None => None,
        };
        let prompt_feat_opt = match opts.prompt_wav.as_ref() {
            Some(path) => Some(self.encode_prompt_wav(path, PadMode::Left)?),
            None => None,
        };

        // 3) Build the full sequence of tokens / masks / feats.
        let z_patch = |n: usize| -> Tensor<B, 3> { Tensor::<B, 3>::zeros([n, p, d], device) };

        let mut tokens: Vec<i64> = Vec::new();
        let mut t_mask: Vec<f32> = Vec::new();
        let mut f_mask: Vec<f32> = Vec::new();
        let mut feat_chunks: Vec<Tensor<B, 3>> = Vec::new();

        // [a] Optional reference prefix: [REF_START, ref×N, REF_END]
        if let Some(ref_feat) = ref_feat_opt {
            let ref_len = ref_feat.dims()[0];
            tokens.push(REF_AUDIO_START_TOKEN);
            tokens.extend(std::iter::repeat_n(0i64, ref_len));
            tokens.push(REF_AUDIO_END_TOKEN);
            t_mask.push(1.0);
            t_mask.extend(std::iter::repeat_n(0.0, ref_len));
            t_mask.push(1.0);
            f_mask.push(0.0);
            f_mask.extend(std::iter::repeat_n(1.0, ref_len));
            f_mask.push(0.0);
            feat_chunks.push(z_patch(1));
            feat_chunks.push(ref_feat);
            feat_chunks.push(z_patch(1));
        }

        // [b] Text tokens (always present).
        tokens.extend_from_slice(&text_tokens);
        t_mask.extend(std::iter::repeat_n(1.0, text_len));
        f_mask.extend(std::iter::repeat_n(0.0, text_len));
        feat_chunks.push(z_patch(text_len));

        // [c] Optional continuation suffix: zero text-tokens at the audio
        //     positions, ones in the audio mask, and the prompt latent patches.
        if let Some(prompt_feat) = prompt_feat_opt {
            let prompt_len = prompt_feat.dims()[0];
            tokens.extend(std::iter::repeat_n(0i64, prompt_len));
            t_mask.extend(std::iter::repeat_n(0.0, prompt_len));
            f_mask.extend(std::iter::repeat_n(1.0, prompt_len));
            feat_chunks.push(prompt_feat);
        }

        let s = tokens.len();
        let feat_seq = if feat_chunks.len() == 1 {
            feat_chunks.pop().unwrap()
        } else {
            Tensor::cat(feat_chunks, 0)
        };
        let text_token_t: Tensor<B, 2, Int> =
            Tensor::from_data(TensorData::new(tokens, [1, s]), device);
        let text_mask_t: Tensor<B, 2> =
            Tensor::from_data(TensorData::new(t_mask, [1, s]), device);
        let feat_mask_t: Tensor<B, 2> =
            Tensor::from_data(TensorData::new(f_mask, [1, s]), device);
        let feat_t: Tensor<B, 4> = feat_seq.unsqueeze_dim(0); // [1, S, P, D]

        // 4) Run the main inference loop: latent patches → [B, D, T*P].
        let latent = self.model.inference(
            text_token_t,
            text_mask_t,
            feat_t,
            feat_mask_t,
            opts.min_len,
            opts.max_len,
            opts.inference_timesteps,
            opts.cfg_value as f64,
        );

        // 5) VAE decode → waveform [B, 1, T_out].
        let wav = self.model.audio_vae.decode(latent);
        let wav = wav.squeeze_dim::<2>(1); // [B, T_out]
        let wav = wav.squeeze_dim::<1>(0); // [T_out]
        let data = wav.into_data();
        // Backend-agnostic: VAE may produce f32, f16 or bf16 depending on
        // the active Backend; convert to f32 for output regardless.
        let samples: Vec<f32> = data
            .convert::<f32>()
            .into_vec::<f32>()
            .map_err(|_| crate::Error::Other("unexpected VAE output dtype".into()))?;
        Ok(samples)
    }
}

#[derive(Debug, Clone, Copy)]
enum PadMode {
    Right,
    Left,
}

