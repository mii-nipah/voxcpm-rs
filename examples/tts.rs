//! End-to-end TTS smoke test against a local VoxCPM2 checkpoint.
//!
//! Run with:
//!   cargo run --release --example tts --no-default-features --features cpu -- \
//!       /home/nipah/dev/ai_space/VoxCPM2 "Hello world" /tmp/out.wav

use std::env;

use voxcpm_rs::{audio, GenerateOptions, VoxCPM};

type B = burn::backend::NdArray<f32>;

fn main() {
    let mut args = env::args().skip(1);
    let model_dir = args
        .next()
        .unwrap_or_else(|| "/home/nipah/dev/ai_space/VoxCPM2".to_string());
    let text = args
        .next()
        .unwrap_or_else(|| "Hello world, this is a test.".to_string());
    let out = args.next().unwrap_or_else(|| "/tmp/out.wav".to_string());

    let device = Default::default();
    eprintln!("loading model from {model_dir} ...");
    let model: VoxCPM<B> = VoxCPM::from_local(&model_dir, &device).expect("load model");
    eprintln!("synthesizing: {text:?}");
    let opts = GenerateOptions {
        inference_timesteps: 10,
        cfg_value: 2.0,
        max_len: 500,
        ..GenerateOptions::default()
    };
    let wav = model.generate(&text, opts).expect("generate");
    let sr = model.sample_rate();
    eprintln!("got {} samples @ {} Hz; writing {out}", wav.len(), sr);
    audio::write_wav(&out, &wav, sr).expect("write wav");
    eprintln!("done");
}
