# Implement OmniVoice Inference in voxcpm-rs

## Context

`voxcpm-rs` is a pure-Rust inference crate for text-to-speech, built on the [Burn](https://burn.dev) ML framework. It currently implements **VoxCPM2** inference and works on Vulkan (via wgpu) and CPU (via ndarray). The crate lives at `/home/nipah/dev/voxcpm-rs`.

Your job is to add inference support for a second TTS model: **OmniVoice** (by k2-fsa). The reference Python implementation is vendored at `vendor/OmniVoice/`. The pretrained checkpoint is at [k2-fsa/OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) on HuggingFace.
You can find the downloaded HuggingFace checkpoint and related files at `/home/nipah/dev/ai_space/OmniVoice`, everything you may need regarding the model is probably there.

OmniVoice is a state-of-the-art multilingual zero-shot TTS supporting 600+ languages, voice cloning, and voice design — all from a clean "diffusion language model-style" architecture that's actually simpler than VoxCPM2.

## What Already Exists (REUSE THIS)

The existing VoxCPM2 implementation provides a complete inference stack. Study it carefully before writing anything — most infrastructure is model-agnostic:

### Fully reusable as-is
- **`src/audio.rs`** — Audio file loading (WAV/FLAC/MP3 via symphonia), resampling (rubato), WAV writing. Model-agnostic.
- **`src/tokenizer.rs`** — HuggingFace `tokenizers` crate wrapper. OmniVoice uses the same format (`tokenizer.json`).
- **`src/weights.rs`** — Safetensors + PTH weight loading with dtype conversion, weight-norm materialisation, QKV fusion, gate/up fusion. You just need to add a new key remap function for OmniVoice's weight names (like `remap_audiovae_key` does for the AudioVAE).
- **`src/error.rs`** — Error types. Add variants if needed.
- **`Cargo.toml`** — All backend features (wgpu, vulkan, cpu, etc.) work for any Burn model.

### Reusable with adaptation
- **`src/minicpm4/`** — Full transformer implementation: GQA attention with fused QKV, RoPE (with scaling), SwiGLU MLP with fused gate+up, RMSNorm, static KV cache. **You need to determine if OmniVoice's LLM backbone is architecturally compatible.** Check the `config.json` in the HF checkpoint — if it's Qwen2/LLaMA-style (which is likely), the existing `minicpm4` module can be reused or lightly adapted. The key difference is OmniVoice needs **2D attention masks** (`[B, 1, S, S]` bool masks) for its classifier-free guidance scheme, whereas the current code uses causal-only masks.
- **`src/config.rs`** — Pattern to follow for new config structs. VoxCPM2's `VoxCpm2Config` shows how to deserialize from `config.json`.
- **`src/voxcpm2/wrapper.rs`** — The high-level API design (`VoxCPM::from_local()`, `generate()`, `generate_stream()`, `Prompt` enum, `GenerateOptions` builder, `CancelToken`) is a great template for the OmniVoice wrapper.

### Not reusable (VoxCPM2-specific)
- `src/audiovae/` — VoxCPM2's continuous AudioVAE. OmniVoice uses HiGGS (discrete codebook).
- `src/locdit/`, `src/locenc.rs`, `src/fsq.rs` — VoxCPM2-specific components (diffusion decoder, local encoder, scalar quantization).
- `src/voxcpm2/model.rs` — VoxCPM2's model struct and inference loop.

## OmniVoice Architecture (What You Need to Build)

The full reference implementation is in **one file**: `vendor/OmniVoice/omnivoice/models/omnivoice.py` (~1600 lines). Read it thoroughly.

### Model Structure

OmniVoice is surprisingly simple. The entire neural net is:

```
OmniVoice:
  llm: AutoModel(llm_config)          # A generic transformer LLM (likely Qwen2-style)
  audio_embeddings: Embedding(C * V, H)  # C=8 codebooks, V=1025 vocab, H=hidden_size
  audio_heads: Linear(H, C * V, bias=False)  # Output projection
  codebook_layer_offsets: buffer [0, 1025, 2050, ...]  # Offsets per codebook
```

That's it. Three learnable components.

### Forward Pass

```python
def forward(input_ids, audio_mask, labels=None, attention_mask=None):
    # input_ids: [B, C, S] — C codebook layers, S sequence length
    # audio_mask: [B, S] — bool, True where audio tokens are
    
    # 1. Text positions: embed layer 0 of input_ids through LLM's text embeddings
    text_embeds = llm.get_input_embeddings()(input_ids[:, 0, :])
    
    # 2. Audio positions: shift IDs by codebook offset, embed through audio_embeddings, sum across codebooks
    shifted_ids = input_ids * audio_mask + codebook_layer_offsets  # [B, C, S]
    audio_embeds = audio_embeddings(shifted_ids).sum(dim=1)  # [B, S, H]
    
    # 3. Combine: use audio_mask to select
    combined = where(audio_mask, audio_embeds, text_embeds)  # [B, S, H]
    
    # 4. Run through LLM
    hidden = llm(inputs_embeds=combined, attention_mask=attention_mask)
    
    # 5. Project to audio logits
    logits = audio_heads(hidden)  # [B, S, C*V]
    logits = logits.view(B, S, C, V).permute(0, 2, 1, 3)  # [B, C, S, V]
    
    return logits
```

### Inference: Iterative Unmasking

OmniVoice generates audio via **iterative unmasking** (NOT autoregressive). The algorithm:

1. **Prepare inputs**: Build a sequence of `[style_tokens | text_tokens | ref_audio_tokens? | MASK×target_len]`. The mask token is `audio_mask_id=1024`.

2. **Duplicate for CFG**: Create two copies — **conditional** (full sequence) and **unconditional** (target audio tokens only, no text/ref context). Stack as `[2B, C, S_max]`.

3. **Schedule**: Divide `num_step` (default 32) steps across the total mask budget (`target_len × num_codebooks`). Each step unmasks `k` tokens.

4. **Loop** (`num_step` iterations):
   - Forward pass on the full `[2B, C, S]` batch → get logits `[2B, C, S, V]`
   - Apply classifier-free guidance: `log_probs = cond + scale * (cond - uncond)`
   - For each batch item, score all still-masked positions (with layer penalty favoring earlier codebooks)
   - Select top-k positions by score (with Gumbel noise for position temperature)
   - Unmask those positions with the predicted tokens
   - Update the input sequence for the next pass

5. **Decode**: Feed the final `[B, C, T]` tokens through HiGGS Audio V2 decode → waveform.

Key params: `num_step=32`, `guidance_scale=2.0`, `t_shift=0.1`, `layer_penalty_factor=5.0`, `position_temperature=5.0`, `class_temperature=0.0` (greedy).

See `_generate_iterative()` at line ~1145 and `_predict_tokens_with_scoring()` at line ~1299 in `omnivoice.py`.

### HiGGS Audio V2 Tokenizer

This is the audio codec — it converts waveforms ↔ discrete tokens across 8 codebooks.

- HuggingFace model: `eustlb/higgs-audio-v2-tokenizer`
- Class: `transformers.HiggsAudioV2TokenizerModel`
- encode: `waveform [B, 1, T]` → `audio_codes [B, C, T']` (C=8 codebooks, 1025 vocab)
- decode: `audio_codes [B, C, T']` → `audio_values [B, 1, T]`
- Sample rate: 24000 Hz
- Frame rate: available as `config.frame_rate`

**You need to inspect the actual HiGGS model architecture** from the HuggingFace repo to understand what needs porting. Download the config and check the architecture. It's likely a convolutional encoder + Finite Scalar Quantization + convolutional decoder (similar to DAC/EnCodec family).

### Input Sequence Format

```
[style_tokens] [text_tokens] [ref_audio_tokens?] [target_mask_tokens]
```

Where:
- `style_tokens` = tokenize(`<|denoise|><|lang_start|>None<|lang_end|><|instruct_start|>None<|instruct_end|>`)
  - Language is always `None` (language-agnostic) — do NOT expose language selection to users
  - Instruct is the voice design string or `None` — just a string the user can optionally pass
  - `<|denoise|>` is prepended only when ref_audio is provided
- `text_tokens` = tokenize(`<|text_start|>{ref_text} {target_text}<|text_end|>`)
- `ref_audio_tokens` = the reference audio encoded through HiGGS (shape `[C, T]`)
- `target_mask_tokens` = `mask_id` (1024) repeated `target_len` times across all `C` codebooks

Each of these is replicated across all `C=8` codebook layers for `input_ids` shape `[1, C, S]`. Text token IDs are the same across all layers. Audio token IDs differ per layer (they're the actual codebook indices).

### Duration Estimation

OmniVoice estimates how many audio tokens to generate from the text using `RuleDurationEstimator` (`vendor/OmniVoice/omnivoice/utils/duration.py`). It's a pure lookup table — Unicode character → phonetic weight, then scale by reference audio's tokens-per-weight ratio. Port it to Rust.

### Audio Post-processing

After decoding tokens → waveform:
- Remove long internal silences (>500ms mid, >100ms lead/trail)
- Adjust volume to match reference RMS
- Fade in/out + edge padding

See `_post_process_audio()` and `vendor/OmniVoice/omnivoice/utils/audio.py`.

## Implementation Strategy

### File layout

```
src/
├── omnivoice/
│   ├── mod.rs          # pub mod + re-exports
│   ├── config.rs       # OmniVoiceConfig, OmniVoiceGenerationConfig
│   ├── model.rs        # OmniVoiceModel struct + forward + iterative unmasking loop
│   ├── wrapper.rs      # High-level OmniVoice wrapper (from_local, generate, etc.)
│   └── duration.rs     # RuleDurationEstimator port
├── higgs/
│   ├── mod.rs
│   ├── model.rs        # HiGGS encoder + decoder + FSQ
│   └── config.rs       # HiGGS config
└── lib.rs              # Add `pub mod omnivoice; pub mod higgs;`
```

### Order of implementation

1. **First**: Download and inspect the HiGGS tokenizer and OmniVoice checkpoint configs from HuggingFace. Understand the exact LLM architecture used and the HiGGS model structure. This determines how much of `minicpm4/` you can reuse.

2. **Config structs**: `OmniVoiceConfig` (mirrors the HF `config.json`), `HiggsConfig`.

3. **HiGGS tokenizer**: Implement encode + decode. Test encode against the Python reference.

4. **OmniVoice model**: The thin wrapper (embeddings + LLM forward + head). If the LLM is compatible with `minicpm4`, this is fast.

5. **Iterative unmasking loop**: Port `_generate_iterative()`.

6. **Weight loading**: Write the remap function for OmniVoice's weight names. Add to `weights.rs` or keep in `omnivoice/` module.

7. **Wrapper**: `OmniVoice::from_local()`, `generate()`, handle voice cloning and voice design as text-level concerns (no special abstractions).

8. **Duration estimator**: Port the Unicode weight table.

9. **Test end-to-end**: Generate audio and compare quality to Python reference.

### Key design decisions

- **Language**: Always `None`. The model handles it. Don't expose language selection.
- **Voice design**: Just a `Option<String>` on the generate options. The user passes `"male, british accent"` or nothing. No validation, no special types — it's just text that gets tokenized.
- **Voice cloning**: Same `Prompt::Reference` pattern as VoxCPM2 — user provides audio, it gets encoded through HiGGS instead of AudioVAE.
- **Unified vs separate API**: Keep it as a **separate** top-level type (`OmniVoice<B>`) alongside `VoxCPM<B>`. Don't try to unify them behind a trait — the models are different enough that it would be a leaky abstraction.
- **2D attention masks**: The iterative unmasking loop needs explicit `[B, 1, S, S]` bool attention masks (for the cond/uncond CFG scheme where uncond rows have padding that must be masked). The current `minicpm4` attention may need a small extension for this.

## Reference Files (read these)

| File | What it contains |
|---|---|
| `vendor/OmniVoice/omnivoice/models/omnivoice.py` | **THE** reference — entire model, config, forward, inference loop, all helpers |
| `vendor/OmniVoice/omnivoice/utils/duration.py` | Duration estimator (Unicode weight table) |
| `vendor/OmniVoice/omnivoice/utils/audio.py` | Audio post-processing (silence removal, cross-fade, fade/pad) |
| `vendor/OmniVoice/omnivoice/utils/text.py` | Text chunking for long audio |
| `vendor/OmniVoice/omnivoice/utils/voice_design.py` | Instruct validation (optional — can skip validation in Rust) |
| `vendor/OmniVoice/omnivoice/utils/lang_map.py` | Language ID mapping (not needed if we always pass None) |
| `src/voxcpm2/model.rs` | Existing VoxCPM2 model — study for patterns |
| `src/voxcpm2/wrapper.rs` | Existing high-level API — study for patterns |
| `src/weights.rs` | Weight loading — you'll add to this |
| `src/minicpm4/` | Transformer backbone — likely reusable for OmniVoice's LLM |
| `src/config.rs` | Config struct patterns |

## Validation

- Generate a short English sentence with both Python OmniVoice and your Rust port
- Compare the HiGGS encode output (tokens should match exactly)
- Compare intermediate LLM hidden states at step 0 of the unmasking loop
- Audio quality doesn't need to be bit-identical (floating point differences across frameworks are expected) but should sound correct.
- If you want to be extra sure you may even use a small whisper pass on the audio to validate it's producing the text you expect.
