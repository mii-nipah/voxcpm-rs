# voxcpm-rs

Pure-Rust inference for [**VoxCPM2**](https://huggingface.co/openbmb/VoxCPM2) — a zero-shot
text-to-speech model with voice cloning — built on top of the
[Burn](https://burn.dev) ML framework.

Runs locally on your machine via **Vulkan** (AMD, NVIDIA, Intel) or a **pure-CPU**
fallback. No Python, no CUDA, no ONNX runtime — just a cargo dependency.

```rust
let model: VoxCPM<B> = VoxCPM::from_local("./VoxCPM2", &device)?;
let wav = model.generate("Hello, world!", GenerateOptions::default())?;
voxcpm_rs::audio::write_wav("out.wav", &wav, model.sample_rate())?;
```

---

## Contents

- [Why](#why)
- [Quick start](#quick-start)
  - [Model files](#model-files)
- [Backends & features](#backends--features)
- [API tour](#api-tour)
  - [Zero-shot synthesis](#zero-shot-synthesis)
  - [Voice cloning](#voice-cloning)
  - [Streaming](#streaming)
  - [Tuning knobs](#tuning-knobs)
  - [Cancellation](#cancellation)
- [Architecture](#architecture)
- [Examples](#examples)
- [Contributing](#contributing)
- [Related projects](#related-projects)
- [License](#license)

---

## Why

The upstream VoxCPM2 reference is Python + PyTorch + CUDA. That is a heavy
dependency tree to ship inside a desktop app, a game, a CLI tool, or any other
Rust project where you want offline, on-device TTS.

`voxcpm-rs` is a single `cargo add` away and runs on:

- Any **Vulkan**-capable GPU (AMD, NVIDIA, Intel, Apple via MoltenVK).
- **Pure CPU** with SIMD elementwise ops, optionally with vendored OpenBLAS for
  multi-core matmul — no system libraries required.

It aims to stay faithful to the official implementation (see `vendor/VoxCPM`) while
exposing a small, idiomatic Rust API.

## Quick start

1. **Grab a checkpoint.** Download the VoxCPM2 weights from Hugging Face:

   ```bash
   huggingface-cli download openbmb/VoxCPM2 --local-dir ./VoxCPM2
   ```

   You should end up with a directory containing `config.json`, `tokenizer.json`,
   `model.safetensors`, and `audiovae.pth`. The crate consumes this layout
   **as-shipped** — no manual weight conversion step is required. See
   [Model files](#model-files) below for the full accepted layout.

2. **Add the crate:**

   ```toml
   # Cargo.toml
   [dependencies]
   voxcpm-rs = { version = "0.1", default-features = false, features = ["wgpu"] }
   ```

3. **Synthesize something:**

   ```rust
   use voxcpm_rs::{audio, GenerateOptions, VoxCPM};

   type B = burn::backend::Wgpu<f32, i32>;

   fn main() -> anyhow::Result<()> {
       let device = Default::default();
       // Load once — takes ~20–25 s for the full model on a modern GPU.
       // Subsequent `generate()` calls reuse the same loaded model.
       let model: VoxCPM<B> = VoxCPM::from_local("./VoxCPM2", &device)?;

       let wav_1 = model.generate("First sentence.",  GenerateOptions::default())?;
       let wav_2 = model.generate("Second sentence.", GenerateOptions::default())?;

       audio::write_wav("out1.wav", &wav_1, model.sample_rate())?;
       audio::write_wav("out2.wav", &wav_2, model.sample_rate())?;
       Ok(())
   }
   ```

   `VoxCPM::generate` takes `&self`, so one loaded model can serve any number
   of **sequential** synthesis calls without reloading. Note however that
   `VoxCPM<B>` is **not `Sync`** — burn's `Param<Tensor<...>>` wraps a
   `std::cell::OnceCell` for lazy device materialization, which transitively
   makes the whole model `!Sync`. To share a single loaded model across
   threads or async tasks, wrap it in `Arc<Mutex<VoxCPM<B>>>` (or
   `Arc<parking_lot::Mutex<...>>`) and serialize `generate` calls; for true
   parallel inference, load one `VoxCPM<B>` per worker.

4. **Or just run the bundled example:**

   ```bash
   cargo run --release --example tts --no-default-features --features wgpu -- \
       ./VoxCPM2 "Hello world from Rust." /tmp/out.wav
   ```

### Model files

`VoxCPM::from_local` expects a directory with:

| File                      | Purpose                               | Format accepted                      |
| ------------------------- | ------------------------------------- | ------------------------------------ |
| `config.json`             | Model architecture config             | JSON                                 |
| `tokenizer.json`          | HuggingFace tokenizer                 | JSON                                 |
| `model.safetensors`  / `model.pth`    | LM + DiT backbone weights | SafeTensors preferred, `.pth`/`.pt` fallback |
| `audiovae.safetensors` / `audiovae.pth` | AudioVAE decoder weights | SafeTensors preferred, `.pth` fallback   |

The upstream HF repo currently ships `model.safetensors` + `audiovae.pth`; both
work directly with no conversion. PyTorch `state_dict.`/`model.`/`module.`
top-level container prefixes are stripped automatically.

Weight loading takes ~20–25 s on first call (a 4.3 GB BF16 backbone is upcast
to F32 for the `wgpu` backend — WGSL has no BF16 type). The cost is paid
**once** per `from_local`; subsequent `generate()` calls are free of any I/O.
Load-phase progress is reported via the [`log`](https://crates.io/crates/log)
crate, so wiring up `env_logger` / `tracing-log` surfaces it.


## Backends & features

Pick exactly one backend:

| Feature        | Backend             | Notes                                                                 |
| -------------- | ------------------- | --------------------------------------------------------------------- |
| `cpu` *(default)* | `burn-ndarray` + SIMD | Works everywhere. Matmul is single-threaded.                          |
| `cpu-blas`     | `cpu` + vendored OpenBLAS | Multi-core matmul. Builds OpenBLAS from source (no system deps).    |
| `wgpu`         | Vulkan / Metal / DX12 | Recommended for GPUs. Fast cold start.                              |
| `wgpu-fast`    | `wgpu` + fusion + autotune | ~5–7% faster steady-state; pays a one-time autotune cost (cached). |
| `vulkan`       | Native Vulkan + **bf16** weights | ~2.6× faster than `wgpu` on AMD RDNA4. **Requires a patch** — see below. |

```bash
# CPU + BLAS
cargo run --release --example tts --no-default-features --features cpu-blas -- ...

# Vulkan, tuned
cargo run --release --example tts --no-default-features --features wgpu-fast -- ...
```

> **Tip:** with `wgpu-fast`, set `CUBECL_AUTOTUNE_LEVEL=minimal` to shrink the
> first-run autotune cost. Results are cached in `target/autotune/`.

### Bf16 Vulkan backend (opt-in, fastest path)

The `vulkan` feature uses Burn's native Vulkan backend and runs the model in
**bf16** end-to-end (the upstream weight dtype — no f32 upcast, half the VRAM,
substantially faster on bf16-capable hardware). Verified ~2.6× speedup over
`wgpu` on an AMD RX 9070 XT (RDNA4).

It needs two small patches that aren't in the released `burn-cubecl` /
`cubecl-spirv` crates yet — one fixes a conv accumulator dtype, the other
promotes a handful of bf16 SPIR-V ops that mesa's NIR translator doesn't lower
correctly. Add this to **your project's** `Cargo.toml` (alongside
`voxcpm-rs = { …, features = ["vulkan"] }`):

```toml
[patch.crates-io]
burn-cubecl  = { git = "https://github.com/mii-nipah/voxcpm-rs", branch = "main" }
cubecl-spirv = { git = "https://github.com/mii-nipah/voxcpm-rs", branch = "main" }
```

That's it — `cargo` clones the repo, finds the patched crates by name, and
rebuilds. No mesa rebuild, no environment variables, no extra steps. Pin to a
specific `rev = "…"` instead of `branch = "main"` for reproducible builds.

> **Why a patch and not a published crate?** Cargo's `[patch.crates-io]` only
> takes effect at the workspace root, so a library can't transparently pull in
> a patched dependency on its consumers' behalf — the patch block must live in
> the consumer's manifest either way. A `git = "…"` reference is the lowest-
> friction form that doesn't require maintaining renamed forks on crates.io.
> See [`patches/README.md`](patches/README.md) for the patch contents and
> rationale.

## API tour

### Zero-shot synthesis

```rust
let wav = model.generate("Good morning.", GenerateOptions::default())?;
```

### Voice cloning

Provide a short reference clip (ideally a few seconds of clean speech):

```rust
use voxcpm_rs::Prompt;

let opts = GenerateOptions::builder()
    .prompt(Prompt::Reference { audio: "speaker.wav".into() })
    .build();

let wav = model.generate("Now I sound like them.", opts)?;
```

Or continue from an existing utterance (the model picks up after `audio`):

```rust
let opts = GenerateOptions::builder()
    .prompt(Prompt::Continuation {
        audio: "intro.wav".into(),
        text:  "Once upon a time,".into(),
    })
    .build();
```

#### Audio from memory

Prompt audio doesn't have to live on disk. [`PromptAudio`](src/voxcpm2/wrapper.rs)
accepts three sources — a path, already-encoded bytes, or raw PCM samples — so
you can plug the model into an in-memory pipeline (microphone capture, HTTP
upload, another TTS stage, …):

```rust
use voxcpm_rs::{Prompt, PromptAudio};

// 1. From a file path (the default — `Into<PromptAudio>` is implemented for
//    `&str`, `&Path` and `PathBuf`):
let a = Prompt::Reference { audio: "speaker.wav".into() };

// 2. From encoded bytes in memory (any format Symphonia supports):
let bytes: Vec<u8> = std::fs::read("speaker.flac")?;
let b = Prompt::Reference { audio: PromptAudio::Encoded(bytes) };

// 3. From raw mono f32 PCM you already have:
let c = Prompt::Reference {
    audio: PromptAudio::Pcm { samples, sample_rate: 24_000 },
};
```

Symmetrically, [`audio::load_audio_bytes`](src/audio.rs) /
[`audio::load_audio_bytes_as`](src/audio.rs) let you decode encoded audio
buffers without touching the filesystem.

### Streaming

For real-time playback, network streaming, or just to start hearing audio
before the whole utterance is ready, use
[`VoxCPM::generate_stream`](src/voxcpm2/wrapper.rs). It returns an iterator
of `Result<Vec<f32>>` chunks at `model.sample_rate()`:

```rust
let opts = GenerateOptions::builder()
    .chunk_patches(5)   // ~400 ms / chunk at the default model config
    .build();

for chunk in model.generate_stream("Streaming hello!", opts)? {
    let chunk = chunk?;
    audio_sink.write(&chunk); // play / send / encode immediately
}
```

Concatenating every chunk yields exactly the same waveform `generate()`
would have returned — chunk boundaries are seamless because the AudioVAE
decoder is causal. `chunk_patches` trades latency for throughput: smaller
→ lower per-chunk latency, larger → fewer chunks. The default `5` is a
sensible balance for live playback.

See [`examples/tts_stream.rs`](examples/tts_stream.rs) for an end-to-end
run with per-chunk timing.

> **Implementation note.** The autoregressive loop (LM + DiT) runs
> incrementally with KV-cache, so streaming adds **no** AR overhead
> compared to `generate()`. The AudioVAE decoder, however, is currently
> stateless across chunks — each chunk re-decodes the cumulative latent
> and emits only the new tail samples, making total VAE work
> `O(N²/chunk_patches)` over an utterance instead of `O(N)`. AR cost
> dominates in practice, so the difference is rarely visible.

### Tuning knobs

All options flow through the fluent builder:

```rust
let opts = GenerateOptions::builder()
    .cfg(2.0)          // classifier-free guidance; 1.5–3.0 is typical
    .timesteps(10)     // diffusion Euler steps; fewer = faster, <6 degrades
    .min_len(2)
    .max_len(500)      // hard cap on generated latent patches (~80 ms each)
    .chunk_patches(5)  // patches per chunk in `generate_stream`
    .build();
```

### Cancellation

Long generations can be cancelled cooperatively from another thread via
`CancelToken`. The autoregressive loop polls the token between every
diffusion step, so cancel latency is bounded by one step
(~200 ms on `wgpu` at default `timesteps=10`).

```rust
use std::{thread, time::Duration};
use voxcpm_rs::{CancelToken, Error, GenerateOptions};

let cancel = CancelToken::new();
{
    let cancel = cancel.clone();
    thread::spawn(move || {
        thread::sleep(Duration::from_secs(5));
        cancel.cancel(); // safe to call from any thread, idempotent
    });
}

let opts = GenerateOptions::builder().cancel(cancel).build();
match model.generate("a very long passage…", opts) {
    Ok(wav) => { /* finished in time */ }
    Err(Error::Cancelled) => { /* user / watchdog bailed */ }
    Err(e) => return Err(e.into()),
}
```

`CancelToken` is `Clone + Send + Sync` (an `Arc<AtomicBool>` underneath),
so you can hand copies to as many watchers as you like.

## Architecture

VoxCPM2 is a cascade of four components — each lives in its own module:

```
text ──► tokenizer ──► minicpm4 (LM backbone) ──► locenc ──► locdit (diffusion) ──► audiovae ──► wav
```

| Module                                | Role                                                   |
| ------------------------------------- | ------------------------------------------------------ |
| [`tokenizer`](src/tokenizer.rs)       | HF `tokenizers` wrapper for the LlamaTokenizerFast vocab. |
| [`minicpm4`](src/minicpm4/)           | Decoder-only LM backbone (rotary attention + KV cache).   |
| [`locenc`](src/locenc.rs)             | Local encoder — conditions the diffusion head on LM hidden states. |
| [`locdit`](src/locdit/)               | Local DiT + conditional flow-matching sampler.         |
| [`audiovae`](src/audiovae/)           | VAE decoder that turns FSQ patches into 16 kHz audio.  |
| [`voxcpm2`](src/voxcpm2/)             | Glue + convenient [`VoxCPM`](src/voxcpm2/wrapper.rs) façade. |

Weights are loaded directly from `.safetensors` or `.pth` via
[`burn-store`](https://crates.io/crates/burn-store) with the `PyTorchToBurnAdapter`,
so HuggingFace checkpoints drop in with no manual conversion step.

## Examples

Browse [`examples/`](examples/) for standalone binaries:

- [`tts.rs`](examples/tts.rs) — end-to-end synthesis.
- [`tts_stream.rs`](examples/tts_stream.rs) — chunked streaming synthesis with per-chunk latency logging.
- [`clone.rs`](examples/clone.rs) — voice cloning from a reference wav.
- [`lm_check.rs`](examples/lm_check.rs), [`vae_check.rs`](examples/vae_check.rs),
  [`feat_check.rs`](examples/feat_check.rs) — per-component parity checks against
  the reference implementation.
- [`bench_rmsnorm.rs`](examples/bench_rmsnorm.rs) — microbench for hot kernels.

## Contributing

Contributions are very welcome — especially:

- Bug reports with a minimal repro and the backend/feature flags you used.
- Performance PRs (kernels, memory layout, KV cache, sampler).
- New backends supported by Burn (CUDA, Metal direct, etc.).

Before opening a PR:

1. `cargo fmt --all` and `cargo clippy --all-targets`.
2. `cargo test --no-default-features --features cpu`.
3. If you touched a numeric path, run the matching `*_check` example against a
   real checkpoint and include the RTF / parity numbers in the PR description.

Keep PRs focused — one feature or fix per PR makes review much easier.

## Related projects

- [**VoxCPM** (official, Python)](https://github.com/OpenBMB/VoxCPM) — the
  reference implementation this crate tracks. A copy lives under
  [`vendor/VoxCPM`](vendor/VoxCPM/) for parity testing.
- [**Burn**](https://github.com/tracel-ai/burn) — the ML framework powering all
  the tensor math here.
- [**cubecl**](https://github.com/tracel-ai/cubecl) — the GPU kernel compiler
  behind Burn's `wgpu` backend.

## License

Licensed under the [Apache License, Version 2.0](LICENSE). The vendored reference
implementation under `vendor/VoxCPM/` (kept in the repository for parity testing,
not shipped on crates.io) retains its own license — see the
[upstream LICENSE](https://github.com/OpenBMB/VoxCPM/blob/main/LICENSE).
