//! Top-level VoxCPM2 model and the high-level [`VoxCPM`] convenience wrapper.

pub mod model;
pub mod wrapper;

pub use model::VoxCpm2Model;
pub use wrapper::{CancelToken, GenerateOptions, GenerateOptionsBuilder, Prompt, PromptAudio, VoxCPM};
