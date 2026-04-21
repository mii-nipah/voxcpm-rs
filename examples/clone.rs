//! Voice cloning smoke test: generate speech in the voice of a reference wav.
//!
//! Usage:
//!   cargo run --release --example clone --no-default-features --features wgpu -- \
//!       <model_dir> <ref_wav> "target text" /tmp/out.wav
//!
//! Example:
//!   cargo run --release --example clone --no-default-features --features wgpu -- \
//!       /home/nipah/dev/ai_space/VoxCPM2 \
//!       vendor/VoxCPM/examples/reference_speaker.wav \
//!       "Hello world, this is a voice cloning test." \
//!       /tmp/clone.wav

#![recursion_limit = "256"]

use std::env;
use std::path::PathBuf;
use std::time::Instant;

use voxcpm_rs::{audio, GenerateOptions, VoxCPM};

#[cfg(feature = "wgpu")]
type B = burn::backend::Wgpu<f32, i32>;
#[cfg(all(not(feature = "wgpu"), feature = "cpu"))]
type B = burn::backend::NdArray<f32>;

fn main() {
    let mut args = env::args().skip(1);
    let model_dir = args
        .next()
        .unwrap_or_else(|| "/home/nipah/dev/ai_space/VoxCPM2".to_string());
    let ref_wav = args
        .next()
        .unwrap_or_else(|| "vendor/VoxCPM/examples/reference_speaker.wav".to_string());
    let text = args
        .next()
        .unwrap_or_else(|| "Hello world, this is a voice cloning test.".to_string());
    let out = args.next().unwrap_or_else(|| "/tmp/clone.wav".to_string());

    let device = Default::default();
    let t0 = Instant::now();
    eprintln!("loading model from {model_dir} ...");
    let model: VoxCPM<B> = VoxCPM::from_local(&model_dir, &device).expect("load model");
    eprintln!("loaded in {:.2?}", t0.elapsed());
    eprintln!("reference: {ref_wav}");
    eprintln!("synthesizing: {text:?}");

    let timesteps = env::var("VOXCPM_TIMESTEPS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(10);
    let opts = GenerateOptions {
        inference_timesteps: timesteps,
        cfg_value: 2.0,
        max_len: 500,
        reference_wav: Some(PathBuf::from(&ref_wav)),
        ..GenerateOptions::default()
    };
    let t1 = Instant::now();
    let wav = model.generate(&text, opts).expect("generate");
    let elapsed = t1.elapsed();
    let sr = model.sample_rate();
    let audio_sec = wav.len() as f32 / sr as f32;
    eprintln!(
        "got {} samples @ {} Hz ({:.2}s of audio) in {:.2?} (RTF = {:.2})",
        wav.len(),
        sr,
        audio_sec,
        elapsed,
        elapsed.as_secs_f32() / audio_sec
    );
    eprintln!("writing {out}");
    audio::write_wav(&out, &wav, sr).expect("write wav");
    eprintln!("done");
}
