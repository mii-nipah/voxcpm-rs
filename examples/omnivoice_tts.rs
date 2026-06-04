//! End-to-end OmniVoice TTS smoke test.
//!
//! Run with:
//!   cargo run --release --example omnivoice_tts --no-default-features --features cpu -- \
//!       /home/nipah/dev/ai_space/OmniVoice "Hello from OmniVoice!" /tmp/omnivoice_out.wav

#![recursion_limit = "256"]

use std::env;
use std::path::PathBuf;
use std::time::Instant;

use voxcpm_rs::omnivoice::{OmniVoice, OmniVoiceOptions, OmniVoicePrompt};
use voxcpm_rs::voxcpm2::wrapper::PromptAudio;
use voxcpm_rs::audio;

#[cfg(all(feature = "vulkan", not(feature = "wgpu")))]
type B = burn::backend::Vulkan<half::bf16, i32>;
#[cfg(all(feature = "wgpu", not(feature = "vulkan")))]
type B = burn::backend::Wgpu<f32, i32>;
#[cfg(all(feature = "wgpu", feature = "vulkan"))]
type B = burn::backend::Vulkan<half::bf16, i32>;
#[cfg(all(not(feature = "wgpu"), not(feature = "vulkan"), feature = "cpu"))]
type B = burn::backend::NdArray<f32>;

#[cfg(feature = "vulkan")]
fn backend_name() -> &'static str {
    "vulkan (bf16, SPIR-V)"
}
#[cfg(all(feature = "wgpu", not(feature = "vulkan")))]
fn backend_name() -> &'static str {
    "wgpu"
}
#[cfg(all(not(feature = "wgpu"), not(feature = "vulkan"), feature = "cpu"))]
fn backend_name() -> &'static str {
    "ndarray"
}

fn main() {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or(
            "info,wgpu_hal=error,wgpu_core=error,naga=error,cubecl_wgpu=warn",
        ),
    )
    .init();

    let mut args = env::args().skip(1);
    let model_dir = args
        .next()
        .unwrap_or_else(|| "/home/nipah/dev/ai_space/OmniVoice".to_string());
    let text = args
        .next()
        .unwrap_or_else(|| "Hello from OmniVoice, this is a pure Rust implementation!".to_string());
    let out = args.next().unwrap_or_else(|| "/tmp/omnivoice_out.wav".to_string());

    let ref_audio_path = args.next();
    let ref_text = args.next().unwrap_or_else(|| "Nice to meet you.".to_string());

    let device = Default::default();
    eprintln!("backend: {}", backend_name());
    let t0 = Instant::now();
    eprintln!("loading OmniVoice model from {} ...", model_dir);
    let omnivoice: OmniVoice<B> = OmniVoice::from_local(&model_dir, &device).expect("load omnivoice model");
    eprintln!("loaded in {:.2?}", t0.elapsed());

    eprintln!("synthesizing: {:?}", text);

    let prompt = if let Some(ref_path) = ref_audio_path {
        eprintln!("using voice clone reference: {} with text: {:?}", ref_path, ref_text);
        OmniVoicePrompt::Clone {
            audio: PromptAudio::File(PathBuf::from(ref_path)),
            text: ref_text,
        }
    } else {
        OmniVoicePrompt::None
    };

    let opts = OmniVoiceOptions::builder()
        .prompt(prompt)
        .guidance_scale(2.0)
        .num_step(32)
        .speed(1.0)
        .build();

    let t1 = Instant::now();
    let wav = omnivoice.generate(&text, opts).expect("generate speech");
    let elapsed = t1.elapsed();
    let sr = omnivoice.sample_rate();
    let audio_sec = wav.len() as f32 / sr as f32;
    eprintln!(
        "got {} samples @ {} Hz ({:.2}s of audio) in {:.2?} (RTF = {:.2})",
        wav.len(),
        sr,
        audio_sec,
        elapsed,
        elapsed.as_secs_f32() / audio_sec
    );

    eprintln!("writing to {} ...", out);
    audio::write_wav(&out, &wav, sr).expect("write wav file");
    eprintln!("done generating speech!");
}
