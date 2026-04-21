//! Pretrained weight loading for VoxCPM2.
//!
//! Uses [`burn_store::SafetensorsStore`] with [`PyTorchToBurnAdapter`] which
//! automatically transposes Linear `weight` tensors from PyTorch's
//! `[out, in]` layout to burn's `[in, out]`.
//!
//! The reference checkpoint additionally has two wrinkles we handle here:
//!   1. The `audiovae.safetensors` file stores AudioVAE weights under their
//!      own top-level namespace. We prepend `audio_vae.` so a single combined
//!      load works against [`crate::VoxCpm2Model`].
//!   2. Convolution layers in the AudioVAE use PyTorch's `weight_norm`
//!      parameterisation (`weight_g` + `weight_v`). We materialise the
//!      effective `weight = weight_g * weight_v / ||weight_v||_per_out_channel`
//!      tensor on the fly before handing it to burn-store.

use std::any::TypeId;
use std::collections::HashMap;
use std::path::Path;

use burn::prelude::*;
use burn::tensor::TensorData;
use burn_store::{
    ApplyResult, ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore, SafetensorsStoreError,
};
use half::{bf16, f16};
use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};

use crate::{Error, Result};

/// Pick a safetensors dtype to *write* such that burn-store can hand the
/// bytes to the target backend tensors with no mismatch. burn-store does
/// not auto-cast across dtypes, so we must materialise weights in the
/// backend's native float type.
fn target_float_dtype<B: Backend>() -> Dtype {
    let id = TypeId::of::<B::FloatElem>();
    if id == TypeId::of::<f16>() {
        Dtype::F16
    } else if id == TypeId::of::<bf16>() {
        Dtype::BF16
    } else {
        Dtype::F32
    }
}

/// Load pretrained weights for a [`crate::VoxCpm2Model`] from a snapshot
/// directory containing `model.safetensors` and (optionally)
/// `audiovae.safetensors`.
pub fn load_pretrained<B: Backend, M: ModuleSnapshot<B>>(
    model: &mut M,
    snapshot_dir: impl AsRef<Path>,
) -> Result<ApplyResult> {
    let dir = snapshot_dir.as_ref();
    let model_path = dir.join("model.safetensors");
    let vae_path = dir.join("audiovae.safetensors");
    let target_dtype = target_float_dtype::<B>();

    let mut result = ApplyResult {
        applied: Vec::new(),
        skipped: Vec::new(),
        missing: Vec::new(),
        unused: Vec::new(),
        errors: Vec::new(),
    };

    if model_path.exists() {
        let r = load_single(model, &model_path, None, None, target_dtype)?;
        merge_apply_result(&mut result, r);
    } else {
        return Err(Error::NotFound(format!("{}", model_path.display())));
    }

    if vae_path.exists() {
        let r = load_single(
            model,
            &vae_path,
            Some("audio_vae."),
            Some(remap_audiovae_key),
            target_dtype,
        )?;
        merge_apply_result(&mut result, r);
    }

    Ok(result)
}

/// Translate HF AudioVAE checkpoint keys (which follow `nn.Sequential`
/// indexing, e.g. `decoder.model.2.block.4.block.1.weight`) to the named-
/// field paths used by [`crate::audiovae`]. Returns `None` for tensors that
/// have no destination on the burn side (e.g. `decoder.sr_bin_boundaries`,
/// which is an `Ignored` buffer).
fn remap_audiovae_key(name: &str) -> Option<String> {
    let parts: Vec<&str> = name.split('.').collect();

    if parts.first().copied() == Some("decoder") {
        if parts.len() >= 2 && parts[1] == "sr_bin_boundaries" {
            return None;
        }
        if parts.len() >= 3 && parts[1] == "sr_cond_model" {
            let hf_idx: usize = parts[2].parse().ok()?;
            if !(2..=7).contains(&hf_idx) {
                return None;
            }
            let i = hf_idx - 2;
            let rest = parts[3..].join(".");
            return Some(format!("decoder.sr_cond_layers.{i}.{rest}"));
        }
        if parts.len() >= 3 && parts[1] == "model" {
            let hf_idx: usize = parts[2].parse().ok()?;
            match hf_idx {
                0 => {
                    let rest = parts[3..].join(".");
                    return Some(format!("decoder.first.dw.conv.{rest}"));
                }
                1 => {
                    let rest = parts[3..].join(".");
                    return Some(format!("decoder.first.pw.conv.{rest}"));
                }
                8 => {
                    let rest = parts[3..].join(".");
                    return Some(format!("decoder.snake_out.{rest}"));
                }
                9 => {
                    let rest = parts[3..].join(".");
                    return Some(format!("decoder.last.conv.{rest}"));
                }
                2..=7 => {
                    let i = hf_idx - 2;
                    if parts.len() < 5 || parts[3] != "block" {
                        return None;
                    }
                    let sub: usize = parts[4].parse().ok()?;
                    match sub {
                        0 => {
                            let rest = parts[5..].join(".");
                            Some(format!("decoder.blocks.{i}.snake.{rest}"))
                        }
                        1 => {
                            let rest = parts[5..].join(".");
                            Some(format!("decoder.blocks.{i}.up.conv.{rest}"))
                        }
                        2 | 3 | 4 => {
                            let r = sub - 1; // res1/res2/res3
                            if parts.len() < 7 || parts[5] != "block" {
                                return None;
                            }
                            let inner: usize = parts[6].parse().ok()?;
                            let rest = parts[7..].join(".");
                            res_unit_inner(&format!("decoder.blocks.{i}.res{r}"), inner, &rest)
                        }
                        _ => None,
                    }
                }
                _ => None,
            }
        } else {
            None
        }
    } else if parts.first().copied() == Some("encoder") {
        if parts.len() >= 3 && parts[1] == "block" {
            let hf_idx: usize = parts[2].parse().ok()?;
            if hf_idx == 0 {
                let rest = parts[3..].join(".");
                return Some(format!("encoder.first.conv.{rest}"));
            }
            let i = hf_idx.checked_sub(1)?;
            if parts.len() < 5 || parts[3] != "block" {
                return None;
            }
            let sub: usize = parts[4].parse().ok()?;
            match sub {
                0 | 1 | 2 => {
                    let r = sub + 1; // res1/res2/res3
                    if parts.len() < 7 || parts[5] != "block" {
                        return None;
                    }
                    let inner: usize = parts[6].parse().ok()?;
                    let rest = parts[7..].join(".");
                    res_unit_inner(&format!("encoder.blocks.{i}.res{r}"), inner, &rest)
                }
                3 => {
                    let rest = parts[5..].join(".");
                    Some(format!("encoder.blocks.{i}.snake.{rest}"))
                }
                4 => {
                    let rest = parts[5..].join(".");
                    Some(format!("encoder.blocks.{i}.down.conv.{rest}"))
                }
                _ => None,
            }
        } else if parts.len() >= 2 && (parts[1] == "fc_mu" || parts[1] == "fc_logvar") {
            let head = parts[1];
            let rest = parts[2..].join(".");
            Some(format!("encoder.{head}.conv.{rest}"))
        } else {
            None
        }
    } else {
        None
    }
}

/// Inner mapping for a `CausalResidualUnit.block` Sequential of
/// `[Snake1d, Conv(k=7,dil), Snake1d, Conv(k=1)]`.
fn res_unit_inner(prefix: &str, inner: usize, rest: &str) -> Option<String> {
    match inner {
        0 => Some(format!("{prefix}.snake1.{rest}")),
        1 => Some(format!("{prefix}.conv1.conv.{rest}")),
        2 => Some(format!("{prefix}.snake2.{rest}")),
        3 => Some(format!("{prefix}.conv2.conv.{rest}")),
        _ => None,
    }
}

fn merge_apply_result(dst: &mut ApplyResult, src: ApplyResult) {
    dst.applied.extend(src.applied);
    dst.missing.extend(src.missing);
    dst.unused.extend(src.unused);
    dst.errors.extend(src.errors);
    dst.skipped.extend(src.skipped);
}

fn load_single<B: Backend, M: ModuleSnapshot<B>>(
    model: &mut M,
    path: &Path,
    prefix: Option<&str>,
    remap: Option<fn(&str) -> Option<String>>,
    target_float_dtype: Dtype,
) -> Result<ApplyResult> {
    // Read, materialize weight_norm, re-serialize, then hand to burn-store.
    let file = std::fs::File::open(path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&mmap)?;
    let synth_bytes = materialize_weight_norm(&st, prefix, remap, target_float_dtype)?;
    drop(st);
    drop(mmap);

    let mut store = SafetensorsStore::from_bytes(Some(synth_bytes))
        .with_from_adapter(PyTorchToBurnAdapter)
        .allow_partial(true);

    model.load_from(&mut store).map_err(map_store_err)
}

fn map_store_err(e: SafetensorsStoreError) -> Error {
    Error::Other(format!("safetensors store: {e}"))
}

/// Read a safetensors file, expanding `weight_norm` parameter pairs
/// (`X.weight_g` + `X.weight_v`) into their materialized `X.weight`. All
/// other tensors are passed through unchanged. If `prefix` is provided, it
/// is prepended to every key in the output. Returns a fresh serialized
/// safetensors buffer.
fn materialize_weight_norm(
    st: &SafeTensors<'_>,
    prefix: Option<&str>,
    remap: Option<fn(&str) -> Option<String>>,
    target_float_dtype: Dtype,
) -> Result<Vec<u8>> {
    use safetensors::serialize;

    let mut weight_g: HashMap<String, (Vec<usize>, Vec<f32>)> = HashMap::new();
    let mut weight_v: HashMap<String, (Vec<usize>, Vec<f32>)> = HashMap::new();
    let mut plain: Vec<(String, Vec<usize>, Dtype, Vec<u8>)> = Vec::new();

    for name in st.names() {
        let view = st
            .tensor(name)
            .map_err(|_| Error::MissingWeight(name.to_string()))?;
        let shape = view.shape().to_vec();
        let data = view.data().to_vec();
        if let Some(stem) = name.strip_suffix(".weight_g") {
            weight_g.insert(stem.to_string(), (shape, decode_f32(view.dtype(), &data)?));
        } else if let Some(stem) = name.strip_suffix(".weight_v") {
            weight_v.insert(stem.to_string(), (shape, decode_f32(view.dtype(), &data)?));
        } else {
            plain.push((name.to_string(), shape, view.dtype(), data));
        }
    }

    // Translate `name` (bare HF key, no prefix) through the optional remap and
    // then add the prefix. Returns `None` if the remap drops the key.
    let translate = |name: &str| -> Option<String> {
        let mapped: String = match remap {
            Some(f) => f(name)?,
            None => name.to_string(),
        };
        Some(match prefix {
            Some(p) => format!("{p}{mapped}"),
            None => mapped,
        })
    };

    let mut out: HashMap<String, (Dtype, Vec<usize>, Vec<u8>)> = HashMap::new();
    for (name, shape, dtype, data) in plain {
        let Some(key) = translate(&name) else { continue };
        // Normalise to f32 first (uniform handling), then re-encode in the
        // backend's target float dtype. burn-store does NOT auto-cast across
        // dtypes: source bytes are handed straight to the target tensor, so
        // any mismatch leaves params zero-init and the model emits silence.
        let (dtype, data) = match dtype {
            Dtype::F32 | Dtype::F16 | Dtype::BF16 => {
                let v = decode_f32(dtype, &data)?;
                (target_float_dtype, encode_float(target_float_dtype, &v))
            }
            other => (other, data),
        };
        out.insert(key, (dtype, shape, data));
    }

    for (stem, (v_shape, v_data)) in weight_v {
        let (g_shape, g_data) = weight_g
            .remove(&stem)
            .ok_or_else(|| Error::MissingWeight(format!("{stem}.weight_g")))?;
        let c_out = v_shape[0];
        let inner: usize = v_shape.iter().skip(1).product();
        if g_data.len() != c_out {
            return Err(Error::ShapeMismatch {
                name: format!("{stem}.weight_g"),
                expected: vec![c_out],
                actual: g_shape,
            });
        }
        let mut w = vec![0f32; v_data.len()];
        for i in 0..c_out {
            let off = i * inner;
            let slice = &v_data[off..off + inner];
            let norm_sq: f32 = slice.iter().map(|x| x * x).sum();
            let norm = norm_sq.sqrt().max(1e-12);
            let scale = g_data[i] / norm;
            for j in 0..inner {
                w[off + j] = slice[j] * scale;
            }
        }
        let bytes = encode_float(target_float_dtype, &w);
        let bare = format!("{stem}.weight");
        let Some(key) = translate(&bare) else { continue };
        out.insert(key, (target_float_dtype, v_shape, bytes));
    }

    if !weight_g.is_empty() {
        let leftover: Vec<String> = weight_g.keys().cloned().collect();
        return Err(Error::Other(format!(
            "weight_g without weight_v: {leftover:?}"
        )));
    }

    // Fuse Q/K/V projections. The Rust attention module uses a single
    // `qkv_proj.weight` linear (fused along the output dim) to save kernel
    // launches on GPU. Checkpoints store three separate tensors; stitch
    // them here post-remap/post-weight-norm so all sources (plain Python
    // safetensors, weight-norm materialized, etc.) land in the same map.
    //
    // `nn::Linear` weights in burn are `[out_features, in_features]`, so we
    // concat along dim 0 (row concat) in the order (q, k, v) to match the
    // `q_size + 2*kv_size` layout the attention forward expects.
    fuse_qkv(&mut out)?;
    fuse_gate_up(&mut out)?;

    let views: Vec<(String, safetensors::tensor::TensorView<'_>)> = out
        .iter()
        .map(|(k, (dtype, shape, data))| {
            let tv = safetensors::tensor::TensorView::new(*dtype, shape.clone(), data)
                .map_err(|e| Error::Other(format!("compose safetensors view `{k}`: {e}")))?;
            Ok::<_, Error>((k.clone(), tv))
        })
        .collect::<std::result::Result<_, _>>()?;
    let bytes = serialize(views.iter().map(|(k, v)| (k.clone(), v)), &None)
        .map_err(|e| Error::Other(format!("serialize synthesized safetensors: {e}")))?;
    Ok(bytes)
}

fn decode_f32(dtype: Dtype, data: &[u8]) -> Result<Vec<f32>> {
    match dtype {
        Dtype::F32 => Ok(data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()),
        Dtype::F16 => Ok(data
            .chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect()),
        Dtype::BF16 => Ok(data
            .chunks_exact(2)
            .map(|c| bf16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect()),
        other => Err(Error::Unsupported(format!(
            "safetensors dtype {other:?} for weight_norm tensor"
        ))),
    }
}

fn f32_to_le_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

/// Encode a slice of f32 values into the byte layout for a given safetensors\n/// float dtype. Used to materialise weights in the backend's native dtype\n/// because burn-store does not auto-cast across dtypes on load.
fn encode_float(dtype: Dtype, v: &[f32]) -> Vec<u8> {
    match dtype {
        Dtype::F32 => f32_to_le_bytes(v),
        Dtype::F16 => {
            let mut out = Vec::with_capacity(v.len() * 2);
            for x in v {
                out.extend_from_slice(&f16::from_f32(*x).to_le_bytes());
            }
            out
        }
        Dtype::BF16 => {
            let mut out = Vec::with_capacity(v.len() * 2);
            for x in v {
                out.extend_from_slice(&bf16::from_f32(*x).to_le_bytes());
            }
            out
        }
        other => panic!("encode_float: unsupported target dtype {other:?}"),
    }
}

/// Fuse `q_proj`/`k_proj`/`v_proj` weights into a single `qkv_proj.weight`
/// entry. The attention module stores these as one `Linear` to save kernel
/// launches at inference; the reference checkpoint stores them separately.
///
/// PyTorch `Linear.weight` is `[out_features, in_features]`, so concatenating
/// along dim 0 (the out-features axis) in row-major storage is just byte
/// append in (q, k, v) order — matching the `q_size + 2*kv_size` split the
/// attention forward expects.
///
/// The `PyTorchToBurnAdapter` downstream still does its usual `[out,in] ->
/// [in,out]` transpose, landing the fused tensor correctly in burn's layout.
fn fuse_qkv(
    out: &mut HashMap<String, (Dtype, Vec<usize>, Vec<u8>)>,
) -> Result<()> {
    // Collect all `...q_proj.weight` keys first so we can mutate `out` inside
    // the loop without aliasing.
    let q_keys: Vec<String> = out
        .keys()
        .filter(|k| k.ends_with(".q_proj.weight"))
        .cloned()
        .collect();

    for q_key in q_keys {
        let stem = q_key
            .strip_suffix(".q_proj.weight")
            .expect("filter guarantees suffix");
        let k_key = format!("{stem}.k_proj.weight");
        let v_key = format!("{stem}.v_proj.weight");
        let qkv_key = format!("{stem}.qkv_proj.weight");

        // Only fuse if all three siblings exist; otherwise this is a linear
        // that happens to be called `q_proj` in some unrelated module.
        if !out.contains_key(&k_key) || !out.contains_key(&v_key) {
            continue;
        }

        let (q_dt, q_shape, q_data) = out.remove(&q_key).unwrap();
        let (k_dt, k_shape, k_data) = out.remove(&k_key).unwrap();
        let (v_dt, v_shape, v_data) = out.remove(&v_key).unwrap();

        if q_dt != k_dt || q_dt != v_dt {
            return Err(Error::Other(format!(
                "qkv fusion dtype mismatch at {stem}: q={q_dt:?} k={k_dt:?} v={v_dt:?}"
            )));
        }
        if q_shape.len() != 2 || k_shape.len() != 2 || v_shape.len() != 2 {
            return Err(Error::Other(format!(
                "qkv fusion expects 2D weights at {stem}, got {q_shape:?}/{k_shape:?}/{v_shape:?}"
            )));
        }
        // In-features (dim 1) must match across q/k/v.
        if q_shape[1] != k_shape[1] || q_shape[1] != v_shape[1] {
            return Err(Error::Other(format!(
                "qkv fusion in-features mismatch at {stem}: q={} k={} v={}",
                q_shape[1], k_shape[1], v_shape[1]
            )));
        }

        let fused_shape = vec![q_shape[0] + k_shape[0] + v_shape[0], q_shape[1]];
        let mut fused = Vec::with_capacity(q_data.len() + k_data.len() + v_data.len());
        fused.extend_from_slice(&q_data);
        fused.extend_from_slice(&k_data);
        fused.extend_from_slice(&v_data);

        out.insert(qkv_key, (q_dt, fused_shape, fused));
    }

    Ok(())
}

/// Fuse `gate_proj` + `up_proj` into a single `gate_up_proj` weight.
///
/// Gated MLPs compute `down(silu(gate(x)) * up(x))`. `gate` and `up` share
/// the same input and output shape (`[hidden, intermediate]` with no bias),
/// so they can be packed into one matmul that produces a `2*intermediate`
/// output, then split along the last dim.
///
/// Same byte-concat trick as `fuse_qkv`: in PyTorch row-major `[out, in]`
/// layout, dim-0 concat is just byte append. The `PyTorchToBurnAdapter`
/// transposes downstream into burn's `[in, out]` layout.
fn fuse_gate_up(
    out: &mut HashMap<String, (Dtype, Vec<usize>, Vec<u8>)>,
) -> Result<()> {
    let gate_keys: Vec<String> = out
        .keys()
        .filter(|k| k.ends_with(".gate_proj.weight"))
        .cloned()
        .collect();

    for gate_key in gate_keys {
        let stem = gate_key
            .strip_suffix(".gate_proj.weight")
            .expect("filter guarantees suffix");
        let up_key = format!("{stem}.up_proj.weight");
        let fused_key = format!("{stem}.gate_up_proj.weight");

        if !out.contains_key(&up_key) {
            continue;
        }

        let (g_dt, g_shape, g_data) = out.remove(&gate_key).unwrap();
        let (u_dt, u_shape, u_data) = out.remove(&up_key).unwrap();

        if g_dt != u_dt {
            return Err(Error::Other(format!(
                "gate/up fusion dtype mismatch at {stem}: gate={g_dt:?} up={u_dt:?}"
            )));
        }
        if g_shape.len() != 2 || u_shape.len() != 2 {
            return Err(Error::Other(format!(
                "gate/up fusion expects 2D weights at {stem}, got {g_shape:?}/{u_shape:?}"
            )));
        }
        if g_shape[1] != u_shape[1] {
            return Err(Error::Other(format!(
                "gate/up fusion in-features mismatch at {stem}: gate={} up={}",
                g_shape[1], u_shape[1]
            )));
        }
        if g_shape[0] != u_shape[0] {
            return Err(Error::Other(format!(
                "gate/up fusion out-features mismatch at {stem}: gate={} up={}",
                g_shape[0], u_shape[0]
            )));
        }

        let fused_shape = vec![g_shape[0] + u_shape[0], g_shape[1]];
        let mut fused = Vec::with_capacity(g_data.len() + u_data.len());
        fused.extend_from_slice(&g_data);
        fused.extend_from_slice(&u_data);

        out.insert(fused_key, (g_dt, fused_shape, fused));
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Low-level helper kept for diagnostics
// ---------------------------------------------------------------------------

/// A memory-mapped safetensors file useful for ad-hoc tensor inspection.
#[derive(Debug)]
pub struct SafetensorsFile {
    _mmap: Mmap,
    view: *const SafeTensors<'static>,
}

impl SafetensorsFile {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file = std::fs::File::open(path.as_ref())?;
        let mmap = unsafe { Mmap::map(&file)? };
        let st = SafeTensors::deserialize(&mmap)?;
        let boxed: Box<SafeTensors<'static>> = unsafe {
            std::mem::transmute::<Box<SafeTensors<'_>>, Box<SafeTensors<'static>>>(Box::new(st))
        };
        let view = Box::into_raw(boxed) as *const SafeTensors<'static>;
        Ok(Self { _mmap: mmap, view })
    }

    fn view(&self) -> &SafeTensors<'_> {
        unsafe { &*self.view }
    }

    pub fn names(&self) -> Vec<String> {
        self.view()
            .names()
            .into_iter()
            .map(|s| s.to_string())
            .collect()
    }

    pub fn read_tensor<B: Backend, const D: usize>(
        &self,
        name: &str,
        device: &B::Device,
    ) -> Result<Tensor<B, D>> {
        let view = self
            .view()
            .tensor(name)
            .map_err(|_| Error::MissingWeight(name.to_string()))?;
        let shape = view.shape().to_vec();
        if shape.len() != D {
            return Err(Error::ShapeMismatch {
                name: name.to_string(),
                expected: vec![D; 1],
                actual: shape,
            });
        }
        let values = decode_f32(view.dtype(), view.data())?;
        let mut dims = [0usize; D];
        for (i, s) in shape.iter().enumerate() {
            dims[i] = *s;
        }
        Ok(Tensor::from_data(TensorData::new(values, dims), device))
    }
}

impl Drop for SafetensorsFile {
    fn drop(&mut self) {
        unsafe {
            let _ = Box::from_raw(self.view as *mut SafeTensors<'static>);
        }
    }
}
