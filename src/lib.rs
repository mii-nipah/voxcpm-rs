//! # voxcpm-rs
//!
//! Pure-Rust inference for [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) built on top
//! of the [Burn](https://burn.dev) ML framework. Supports Vulkan (via `wgpu`) and a CPU
//! fallback through `ndarray`.
//!
//! ## Quick start
//!
//! ```no_run
//! # #[cfg(feature = "cpu")] {
//! use voxcpm_rs::{VoxCPM, GenerateOptions};
//!
//! type B = burn::backend::NdArray<f32>;
//! let device = Default::default();
//! let model: VoxCPM<B> = VoxCPM::from_local("./pretrained_models/VoxCPM2", &device).unwrap();
//! let wav = model.generate("Hello, world!", GenerateOptions::default()).unwrap();
//! voxcpm_rs::audio::write_wav("out.wav", &wav, model.sample_rate()).unwrap();
//! # }
//! ```
//!
//! See the [`VoxCPM`] struct for the convenience API, or the individual submodules
//! ([`minicpm4`], [`locdit`], [`locenc`], [`audiovae`]) for low-level access.

#![warn(missing_debug_implementations)]

pub mod audio;
pub mod audiovae;
pub mod config;
pub mod error;
pub mod fsq;
pub mod locdit;
pub mod locenc;
pub mod minicpm4;
pub mod tokenizer;
pub mod voxcpm2;
pub mod weights;

pub use audiovae::AudioVae;
pub use config::{
    AudioVaeConfig, CfmConfig, LoraConfig, MiniCpm4Config, RopeScalingConfig, VoxCpm2Config,
    VoxCpmDitConfig, VoxCpmEncoderConfig,
};
pub use error::{Error, Result};
pub use voxcpm2::{GenerateOptions, PromptCache, VoxCPM};
pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
