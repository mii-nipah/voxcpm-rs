# Upstream patches

These patches fix bugs that block bf16 inference on AMD radv (RDNA4) via
the cubecl-spirv path. Without them, voxcpm-rs in bf16 either crashes
(NaN) or emits silent audio.

## For downstream users — opting into the bf16 fast path

If you depend on voxcpm-rs from your own project and want the bf16 build
(2.6× faster than f32 on RDNA4), add this block to your **workspace root**
`Cargo.toml`:

```toml
[patch.crates-io]
burn-cubecl  = { git = "https://github.com/mii-nipah/voxcpm-rs", branch = "main" }
cubecl-spirv = { git = "https://github.com/mii-nipah/voxcpm-rs", branch = "main" }
```

Cargo will look up the named crate inside the repo and find the patched
copies under `vendor/patched/`. The patch block must live in the
**workspace root** because cargo `[patch]` directives don't propagate
transitively from dependencies — that's a hard cargo rule, not a
voxcpm-rs choice.

If you're not using the bf16 / vulkan build, **you don't need any of
this** — the default `cpu` and `wgpu` features work straight from
`cargo add voxcpm-rs` with no patches.

## `burn-cubecl-bf16-acc.patch`

Patches `burn-cubecl 0.20.1` direct conv kernels to accumulate in `f32`
instead of the element type `E` (which can be bf16). For audiovae's
transposed convs, each output sums up to ~32k bf16 multiplies; with a
bf16 accumulator the sum collapses toward zero (audio absmax ≈ 7/32768).

Files patched:

- `src/kernel/conv/direct.rs`
- `src/kernel/conv/conv_transpose2d/transpose_direct.rs`

Vendored at `vendor/patched/burn-cubecl/`. Wired into the workspace via
`[patch.crates-io]` in the root `Cargo.toml`.

## `cubecl-spirv-bf16-promote.patch`

Adds a new `Bf16PromoteTransform` IR transformer to `cubecl-spirv 0.9.0`
that, before SPIR-V emission, rewrites every bf16 arithmetic /
comparison / plane / select op as: cast operand(s) to f32 → run the op
in f32 → cast result back to bf16. This sidesteps the broken radv NIR
translation of generic `OpFAdd`/`OpFMul`/`OpFMA`/`OpFOrd*`/etc on bf16
(the SPV_KHR_bfloat16 extension is advertised but only the special
`bfmul`/`bffma`/`bfdot` intrinsics actually work — generic ops fall
through to a default arm that miscompiles).

Files patched:

- `src/compiler.rs` (one-line: register the transformer)
- `src/transformers.rs` (new transformer; appended to file)

Vendored at `vendor/patched/cubecl-spirv/`. Wired into the workspace via
`[patch.crates-io]` in the root `Cargo.toml`. With this patch applied,
**no mesa rebuild is needed**.

## `mesa-spirv-bf16-promote.patch` — OPTIONAL / REFERENCE ONLY

Patches Mesa's SPIR-V → NIR translator to do the same f32 promotion
inside the driver. Originally the only fix; superseded by the
cubecl-spirv transformer for Rust/cubecl users, but kept here because:

- It benefits *any* tool that emits bf16 SPIR-V (not just cubecl), so
  it could be useful for non-cubecl projects.
- It serves as a fallback if the cubecl-spirv approach ever needs to be
  removed.

Apply with:

```sh
cd path/to/mesa
git apply path/to/voxcpm-rs/patches/mesa-spirv-bf16-promote.patch
meson setup build -Dvulkan-drivers=amd -Dgallium-drivers= --prefix=/tmp/mesa-install
ninja -C build install
VK_DRIVER_FILES=/tmp/mesa-install/share/vulkan/icd.d/radeon_icd.x86_64.json cargo run ...
```

Tested against Mesa 25.2.8.
