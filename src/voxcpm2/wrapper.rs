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
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// Cooperative cancellation handle for [`VoxCPM::generate`].
///
/// Cheap to clone (`Arc<AtomicBool>` underneath). Signal cancellation from
/// any thread by calling [`CancelToken::cancel`]; the in-flight `generate`
/// call will check between autoregressive steps and bail with
/// [`crate::Error::Cancelled`]. Cancel latency is bounded by one diffusion
/// step (~200 ms on `wgpu` at default `timesteps=10`).
///
/// ```no_run
/// use std::thread;
/// use voxcpm_rs::{CancelToken, GenerateOptions, Prompt, VoxCPM};
/// # type B = burn::backend::NdArray<f32>;
/// # let model: VoxCPM<B> = unimplemented!();
/// let cancel = CancelToken::new();
///
/// // Cancel from a watchdog thread after 2 s.
/// {
///     let cancel = cancel.clone();
///     thread::spawn(move || {
///         thread::sleep(std::time::Duration::from_secs(2));
///         cancel.cancel();
///     });
/// }
///
/// let opts = GenerateOptions::builder().cancel(cancel).build();
/// match model.generate("a very long text...", opts) {
///     Ok(wav) => { /* completed */ }
///     Err(voxcpm_rs::Error::Cancelled) => { /* user cancelled */ }
///     Err(e) => return Err(e.into()),
/// }
/// # Ok::<_, Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, Default)]
pub struct CancelToken(Arc<AtomicBool>);

impl CancelToken {
    /// Create a new, un-cancelled token.
    pub fn new() -> Self {
        Self::default()
    }

    /// Signal cancellation. Idempotent; safe to call from any thread.
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Relaxed);
    }

    /// Whether [`Self::cancel`] has been called.
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }
}

/// Source of prompt audio for [`Prompt::Reference`], [`Prompt::Continuation`]
/// and [`Prompt::Combined`].
///
/// Supports three input modes — pick whichever fits your pipeline:
/// - [`PromptAudio::File`] — path to an encoded audio file (WAV/FLAC/MP3/...).
/// - [`PromptAudio::Encoded`] — encoded audio bytes already in memory
///   (same format support as `File`, just sourced from a `Vec<u8>`).
/// - [`PromptAudio::Pcm`] — raw mono `f32` samples and their sample rate.
///   Use this when you already have decoded audio (e.g. from a microphone
///   capture, an in-process resampler, or a TTS chain).
///
/// `From<PathBuf>` / `From<&Path>` / `From<&str>` are implemented for
/// ergonomics, so paths can be passed directly without wrapping.
#[derive(Debug, Clone)]
pub enum PromptAudio {
    /// Decode an audio file from disk.
    File(PathBuf),
    /// Decode an encoded audio buffer (any format Symphonia supports).
    Encoded(Vec<u8>),
    /// Use already-decoded mono `f32` samples at the given sample rate.
    Pcm {
        /// Mono PCM samples in `[-1.0, 1.0]`.
        samples: Vec<f32>,
        /// Sample rate of `samples` in Hz.
        sample_rate: u32,
    },
}

impl From<PathBuf> for PromptAudio {
    fn from(p: PathBuf) -> Self {
        PromptAudio::File(p)
    }
}
impl From<&Path> for PromptAudio {
    fn from(p: &Path) -> Self {
        PromptAudio::File(p.to_path_buf())
    }
}
impl From<&str> for PromptAudio {
    fn from(p: &str) -> Self {
        PromptAudio::File(PathBuf::from(p))
    }
}

/// How the model should be conditioned on prompt audio.
///
/// See [`VoxCPM::generate`] for what each mode does conceptually.
#[derive(Debug, Clone, Default)]
pub enum Prompt {
    /// No prompt audio — the model improvises a voice.
    #[default]
    None,
    /// Voice cloning via a structurally isolated reference audio prefix.
    /// No transcript required; the audio is bracketed by `[REF_AUDIO_*]` tokens.
    Reference {
        /// Audio of the speaker to clone. See [`PromptAudio`].
        audio: PromptAudio,
    },
    /// In-context continuation: the model literally finishes an utterance
    /// whose start is `audio` (transcribed by `text`).
    Continuation {
        /// Audio containing the start of the utterance. See [`PromptAudio`].
        audio: PromptAudio,
        /// Transcript of `audio` — prepended to the target text before tokenization.
        text: String,
    },
    /// Reference prefix *and* continuation suffix in the same sequence.
    /// Useful when continuation alone drifts off the speaker.
    Combined {
        /// Reference audio (prefix, isolated by `[REF_AUDIO_*]` tokens).
        reference_audio: PromptAudio,
        /// Continuation audio (suffix, autoregression starts from its end).
        prompt_audio: PromptAudio,
        /// Transcript of `prompt_audio`.
        prompt_text: String,
    },
}

/// Sampling / decoding knobs for [`VoxCPM::generate`].
#[derive(Debug, Clone)]
pub struct GenerateOptions {
    /// Classifier-free guidance scale. Higher → closer to the conditioning
    /// (text + prompt), lower → more diverse. Typical range: `1.5..=3.0`.
    pub cfg_value: f32,
    /// Number of Euler steps in the diffusion sampler. Linear cost; lower is
    /// faster but quality drops below ~6.
    pub inference_timesteps: usize,
    /// Minimum number of latent patches to generate before the stop head is
    /// allowed to fire. Guards against immediate cutoffs on very short text.
    pub min_len: usize,
    /// Hard upper bound on the number of latent patches per call. Each patch
    /// is `patch_size * chunk_size / sample_rate` seconds of audio (~80 ms).
    pub max_len: usize,
    /// Prompt-conditioning mode. See [`Prompt`].
    pub prompt: Prompt,
    /// Optional [`CancelToken`] for cooperative cancellation. When `Some`,
    /// the autoregressive loop checks the token between every step and
    /// returns [`crate::Error::Cancelled`] if it has been signalled.
    pub cancel: Option<CancelToken>,
}

impl Default for GenerateOptions {
    fn default() -> Self {
        Self {
            cfg_value: 2.0,
            inference_timesteps: 10,
            min_len: 2,
            max_len: 2000,
            prompt: Prompt::None,
            cancel: None,
        }
    }
}

impl GenerateOptions {
    /// Start a fluent builder for `GenerateOptions`.
    ///
    /// ```no_run
    /// use voxcpm_rs::{GenerateOptions, Prompt};
    /// let opts = GenerateOptions::builder()
    ///     .timesteps(8)
    ///     .cfg(2.0)
    ///     .prompt(Prompt::Reference { audio: "speaker.wav".into() })
    ///     .build();
    /// ```
    pub fn builder() -> GenerateOptionsBuilder {
        GenerateOptionsBuilder::default()
    }
}

/// Fluent builder for [`GenerateOptions`]. Created via
/// [`GenerateOptions::builder`].
#[derive(Debug, Clone, Default)]
pub struct GenerateOptionsBuilder {
    inner: GenerateOptions,
}

impl GenerateOptionsBuilder {
    /// Set [`GenerateOptions::cfg_value`].
    pub fn cfg(mut self, v: f32) -> Self {
        self.inner.cfg_value = v;
        self
    }
    /// Set [`GenerateOptions::inference_timesteps`].
    pub fn timesteps(mut self, n: usize) -> Self {
        self.inner.inference_timesteps = n;
        self
    }
    /// Set [`GenerateOptions::min_len`].
    pub fn min_len(mut self, n: usize) -> Self {
        self.inner.min_len = n;
        self
    }
    /// Set [`GenerateOptions::max_len`].
    pub fn max_len(mut self, n: usize) -> Self {
        self.inner.max_len = n;
        self
    }
    /// Set [`GenerateOptions::prompt`].
    pub fn prompt(mut self, p: Prompt) -> Self {
        self.inner.prompt = p;
        self
    }
    /// Set [`GenerateOptions::cancel`].
    pub fn cancel(mut self, token: CancelToken) -> Self {
        self.inner.cancel = Some(token);
        self
    }
    /// Finalize into a [`GenerateOptions`].
    pub fn build(self) -> GenerateOptions {
        self.inner
    }
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
    ///
    /// Weight-load progress is reported through the [`log`] crate (`info` for
    /// the summary, `warn` for missing/unused tensors, `error` for load
    /// errors). Wire up `env_logger`, `tracing-log`, etc. to surface them.
    pub fn from_local(path: impl AsRef<Path>, device: &B::Device) -> crate::Result<Self> {
        let path = path.as_ref();
        let config_bytes = std::fs::read_to_string(path.join("config.json"))?;
        let config: VoxCpm2Config = serde_json::from_str(&config_bytes)?;
        let tokenizer = TextTokenizer::from_local(path)?;
        let mut model = VoxCpm2Model::<B>::new(config, device);
        let result = crate::weights::load_pretrained(&mut model, path)?;
        log::info!(
            "weights loaded — applied={}, skipped={}, missing={}, unused={}, errors={}",
            result.applied.len(),
            result.skipped.len(),
            result.missing.len(),
            result.unused.len(),
            result.errors.len(),
        );
        if !result.missing.is_empty() {
            log::warn!("missing module params (first 20):");
            for (k, ctx) in result.missing.iter().take(20) {
                log::warn!("  {k} [{ctx}]");
            }
        }
        if !result.unused.is_empty() {
            log::warn!("unused checkpoint tensors (first 20):");
            for k in result.unused.iter().take(20) {
                log::warn!("  {k}");
            }
        }
        if !result.errors.is_empty() {
            log::error!("load errors (first 20):");
            for e in result.errors.iter().take(20) {
                log::error!("  {e:?}");
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
    fn encode_prompt_audio(
        &self,
        audio: &PromptAudio,
        padding_mode: PadMode,
    ) -> crate::Result<Tensor<B, 3>> {
        let encoder_sr = self.model.audio_vae.sample_rate() as u32;
        let mut samples = match audio {
            PromptAudio::File(path) => crate::audio::load_audio_as(path, encoder_sr)?,
            PromptAudio::Encoded(bytes) => crate::audio::load_audio_bytes_as(bytes, encoder_sr)?,
            PromptAudio::Pcm { samples, sample_rate } => {
                crate::audio::resample(samples, *sample_rate, encoder_sr)?
            }
        };
        let p = self.model.patch_size();
        let chunk = self.model.audio_vae.config.0.chunk_size();
        let patch_len = p * chunk;
        let n = samples.len();
        if n == 0 {
            return Err(crate::Error::AudioDecode(
                "prompt audio decoded to 0 samples".into(),
            ));
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
    /// The generation mode is selected by [`GenerateOptions::prompt`]:
    /// - [`Prompt::None`] — zero-shot, model improvises a voice.
    /// - [`Prompt::Reference`] — voice cloning via a structurally isolated
    ///   reference audio prefix (no transcript required).
    /// - [`Prompt::Continuation`] — model continues from the prompt audio in
    ///   the same speaker's voice. The prompt's transcript is prepended to
    ///   `text`.
    /// - [`Prompt::Combined`] — both a reference prefix and a continuation
    ///   suffix.
    pub fn generate(&self, text: &str, opts: GenerateOptions) -> crate::Result<Vec<f32>> {
        // Decompose the prompt into the (optional) reference + continuation
        // pieces the sequence builder consumes.
        let (ref_audio, prompt_audio, prompt_text) = match &opts.prompt {
            Prompt::None => (None, None, None),
            Prompt::Reference { audio } => (Some(audio), None, None),
            Prompt::Continuation { audio, text } => (None, Some(audio), Some(text.as_str())),
            Prompt::Combined {
                reference_audio,
                prompt_audio,
                prompt_text,
            } => (
                Some(reference_audio),
                Some(prompt_audio),
                Some(prompt_text.as_str()),
            ),
        };

        let device = &self.device;
        let p = self.model.patch_size();
        let d = self.model.latent_dim();

        // 1) Tokenize text. In continuation modes prompt_text is prepended.
        let full_text: String = match prompt_text {
            Some(pt) => format!("{pt}{text}"),
            None => text.to_string(),
        };
        let mut text_tokens = self.tokenizer.encode(&full_text)?;
        text_tokens.push(AUDIO_START_TOKEN);
        let text_len = text_tokens.len();

        // 2) Encode optional prompt audios.
        let ref_feat_opt = match ref_audio {
            Some(audio) => Some(self.encode_prompt_audio(audio, PadMode::Right)?),
            None => None,
        };
        let prompt_feat_opt = match prompt_audio {
            Some(audio) => Some(self.encode_prompt_audio(audio, PadMode::Left)?),
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
        // Wrap the cancel token (if any) into a `dyn Fn() -> bool` so the
        // model layer doesn't need to know about `CancelToken` directly.
        let cancel_fn: Option<Box<dyn Fn() -> bool>> = opts
            .cancel
            .as_ref()
            .map(|c| {
                let c = c.clone();
                Box::new(move || c.is_cancelled()) as Box<dyn Fn() -> bool>
            });
        let latent = self.model.inference(
            text_token_t,
            text_mask_t,
            feat_t,
            feat_mask_t,
            opts.min_len,
            opts.max_len,
            opts.inference_timesteps,
            opts.cfg_value as f64,
            cancel_fn.as_deref(),
        )?;

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

