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
use std::rc::Rc;

use burn::prelude::*;
use burn::tensor::{Bytes, DType, TensorData};
use burn_store::{
    ApplyResult, ModuleSnapshot, ModuleStore, PyTorchToBurnAdapter, SafetensorsStore,
    SafetensorsStoreError, TensorSnapshot, TensorSnapshotError,
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
/// directory.
///
/// Accepts the upstream [openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2)
/// HuggingFace layout as-shipped. Required files in `snapshot_dir`:
///
/// - `model.safetensors` **or** `model.pth` / `model.pt` — main model weights.
/// - `audiovae.safetensors` **or** `audiovae.pth` — AudioVAE weights.
///   (The HF repo currently ships `audiovae.pth`; `.safetensors` is preferred
///   when both are present.)
pub fn load_pretrained<B: Backend, M: ModuleSnapshot<B>>(
    model: &mut M,
    snapshot_dir: impl AsRef<Path>,
) -> Result<ApplyResult> {
    let dir = snapshot_dir.as_ref();
    let target_dtype = target_float_dtype::<B>();

    let mut result = ApplyResult {
        applied: Vec::new(),
        skipped: Vec::new(),
        missing: Vec::new(),
        unused: Vec::new(),
        errors: Vec::new(),
    };

    // Main model weights: prefer safetensors, fall back to pth/pt.
    let model_st = dir.join("model.safetensors");
    let model_pth = dir.join("model.pth");
    let model_pt = dir.join("model.pt");
    if model_st.exists() {
        let r = load_single(model, &model_st, None, None, target_dtype)?;
        merge_apply_result(&mut result, r);
    } else if model_pth.exists() {
        let r = load_single_pth(model, &model_pth, None, None, target_dtype)?;
        merge_apply_result(&mut result, r);
    } else if model_pt.exists() {
        let r = load_single_pth(model, &model_pt, None, None, target_dtype)?;
        merge_apply_result(&mut result, r);
    } else {
        return Err(Error::NotFound(format!(
            "no model weights found in {} (expected model.safetensors or model.pth)",
            dir.display()
        )));
    }

    // AudioVAE weights: same preference.
    let vae_st = dir.join("audiovae.safetensors");
    let vae_pth = dir.join("audiovae.pth");
    if vae_st.exists() {
        let r = load_single(
            model,
            &vae_st,
            Some("audio_vae."),
            Some(remap_audiovae_key),
            target_dtype,
        )?;
        merge_apply_result(&mut result, r);
    } else if vae_pth.exists() {
        let r = load_single_pth(
            model,
            &vae_pth,
            Some("audio_vae."),
            Some(remap_audiovae_key),
            target_dtype,
        )?;
        merge_apply_result(&mut result, r);
    } else {
        log::warn!(
            "no audiovae weights found in {} (expected audiovae.safetensors or audiovae.pth) — audio decoding will use random weights",
            dir.display()
        );
    }

    // burn-store reports `missing` per file (any model param not supplied
    // by *that* file). When loading both `model.safetensors` and
    // `audiovae.safetensors`, each file legitimately omits the other half
    // of the model. Dedupe `missing` against the union of `applied` so
    // the final report shows only params that were truly never loaded.
    let applied_set: std::collections::HashSet<&str> =
        result.applied.iter().map(|s| s.as_str()).collect();
    result
        .missing
        .retain(|(path, _)| !applied_set.contains(path.as_str()));
    // While we're at it, dedupe the missing list itself (a param may be
    // reported missing by every file).
    let mut seen = std::collections::HashSet::new();
    result.missing.retain(|(path, _)| seen.insert(path.clone()));

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
    // The store's snapshots own an Arc to the mmap. Cloning the snapshots
    // copies metadata only; source bytes are read when the module asks for
    // each parameter, and the mmap stays alive until application finishes.
    let mut store = SafetensorsStore::from_file(path);
    let tensors = store
        .get_all_snapshots()
        .map_err(map_store_err)?
        .iter()
        .map(|(name, snapshot)| (name.clone(), snapshot.clone()))
        .collect();
    prepare_and_apply(model, path, tensors, prefix, remap, target_float_dtype)
}

/// Load PyTorch metadata lazily, preserving the reader's backing storage in
/// the snapshots instead of materializing the entire checkpoint in RAM.
fn load_single_pth<B: Backend, M: ModuleSnapshot<B>>(
    model: &mut M,
    path: &Path,
    prefix: Option<&str>,
    remap: Option<fn(&str) -> Option<String>>,
    target_float_dtype: Dtype,
) -> Result<ApplyResult> {
    use burn_store::pytorch::PytorchReader;

    let reader = PytorchReader::new(path)
        .map_err(|e| Error::Other(format!("read pytorch file `{}`: {e}", path.display())))?;
    let tensors = reader
        .tensors()
        .iter()
        .map(|(name, snapshot)| (strip_pth_top_level(name).to_string(), snapshot.clone()))
        .collect();
    prepare_and_apply(model, path, tensors, prefix, remap, target_float_dtype)
}

fn burn_dtype_to_safetensors(dt: burn::tensor::DType) -> Result<Dtype> {
    use burn::tensor::DType as B;
    Ok(match dt {
        B::F64 => Dtype::F64,
        B::F32 | B::Flex32 => Dtype::F32,
        B::F16 => Dtype::F16,
        B::BF16 => Dtype::BF16,
        B::I64 => Dtype::I64,
        B::I32 => Dtype::I32,
        B::I16 => Dtype::I16,
        B::I8 => Dtype::I8,
        B::U64 => Dtype::U64,
        B::U32 => Dtype::U32,
        B::U16 => Dtype::U16,
        B::U8 => Dtype::U8,
        B::Bool => Dtype::BOOL,
        other => {
            return Err(Error::Unsupported(format!(
                "burn DType {other:?} has no safetensors equivalent"
            )));
        }
    })
}

fn prepare_and_apply<B: Backend, M: ModuleSnapshot<B>>(
    model: &mut M,
    path: &Path,
    tensors: HashMap<String, TensorSnapshot>,
    prefix: Option<&str>,
    remap: Option<fn(&str) -> Option<String>>,
    target_float_dtype: Dtype,
) -> Result<ApplyResult> {
    let t0 = std::time::Instant::now();
    let snapshots = prepare_snapshots(tensors, prefix, remap, target_float_dtype)?;
    let result = model.apply(snapshots, None, Some(Box::new(PyTorchToBurnAdapter)), false);
    log::debug!(
        "weights[{}] lazy load+apply: {:.2?}",
        path.display(),
        t0.elapsed()
    );
    if !result.errors.is_empty() {
        return Err(map_store_err(SafetensorsStoreError::ValidationFailed(
            format!("Import errors: {:?}", result.errors),
        )));
    }
    Ok(result)
}

fn map_store_err(e: SafetensorsStoreError) -> Error {
    Error::Other(format!("safetensors store: {e}"))
}

/// Strip a common top-level container prefix that HF-published PyTorch
/// checkpoints often use (e.g. `state_dict.`, `model.`, `model_state_dict.`)
/// so the downstream name→module remapping can operate on bare keys.
/// No-op if no known prefix matches.
fn strip_pth_top_level(name: &str) -> &str {
    for prefix in ["state_dict.", "model_state_dict.", "module."] {
        if let Some(rest) = name.strip_prefix(prefix) {
            return rest;
        }
    }
    name
}

/// Build a lazy transformation graph. This retains only checkpoint metadata:
/// dtype conversion, weight normalization and projection fusion happen for
/// one destination parameter at a time. Never serialize a whole converted
/// checkpoint: for the f32 backend that used to create a second multi-GB copy.
fn prepare_snapshots(
    mut tensors: HashMap<String, TensorSnapshot>,
    prefix: Option<&str>,
    remap: Option<fn(&str) -> Option<String>>,
    target_float_dtype: Dtype,
) -> Result<Vec<TensorSnapshot>> {
    let target_dtype = match target_float_dtype {
        Dtype::F32 => DType::F32,
        Dtype::F16 => DType::F16,
        Dtype::BF16 => DType::BF16,
        other => return Err(Error::Unsupported(format!("target float dtype {other:?}"))),
    };
    // Check actual lazy reads against metadata, including PyTorch's backing
    // storage. A malformed source must not be reinterpreted as another dtype.
    tensors = tensors
        .into_iter()
        .map(|(name, snapshot)| {
            let source = snapshot.clone_data_fn();
            let dtype = snapshot.dtype;
            let shape = snapshot.shape.clone();
            let expected_len = snapshot.data_len();
            let path = name.clone();
            let checked = TensorSnapshot::from_closure(
                Rc::new(move || {
                    let data = source()?;
                    if data.dtype != dtype
                        || data.shape != shape
                        || data.as_bytes().len() != expected_len
                    {
                        return Err(TensorSnapshotError::DataError(format!(
                            "tensor data disagrees with metadata for {path}"
                        )));
                    }
                    Ok(data)
                }),
                dtype,
                snapshot.shape,
                snapshot.path_stack.unwrap_or_default(),
                vec![],
                Default::default(),
            );
            (name, checked)
        })
        .collect();
    let v_keys: Vec<_> = tensors
        .keys()
        .filter(|k| k.ends_with(".weight_v"))
        .cloned()
        .collect();
    for v_key in v_keys {
        let stem = v_key.strip_suffix(".weight_v").unwrap();
        let v = tensors.remove(&v_key).unwrap();
        let g_key = format!("{stem}.weight_g");
        let g = tensors
            .remove(&g_key)
            .ok_or_else(|| Error::MissingWeight(g_key.clone()))?;
        let Some(&c_out) = v.shape.first() else {
            return Err(Error::Other(format!(
                "weight_norm expects a non-scalar tensor at {stem}"
            )));
        };
        if g.shape.iter().product::<usize>() != c_out {
            return Err(Error::ShapeMismatch {
                name: g_key,
                expected: vec![c_out],
                actual: g.shape,
            });
        }
        let v_dtype = burn_dtype_to_safetensors(v.dtype)?;
        let g_dtype = burn_dtype_to_safetensors(g.dtype)?;
        for dtype in [v_dtype, g_dtype] {
            if !matches!(dtype, Dtype::F32 | Dtype::F16 | Dtype::BF16) {
                return Err(Error::Unsupported(format!(
                    "safetensors dtype {dtype:?} for weight_norm tensor"
                )));
            }
        }
        let shape = v.shape.clone();
        let inner: usize = shape.iter().skip(1).product();
        let data_shape = shape.clone();
        let data_fn = Rc::new(move || {
            let v_data = decode_f32(v_dtype, v.to_data()?.as_bytes()).map_err(snapshot_error)?;
            let g_data = decode_f32(g_dtype, g.to_data()?.as_bytes()).map_err(snapshot_error)?;
            let mut w = vec![0f32; v_data.len()];
            for (i, g) in g_data.iter().enumerate() {
                let off = i * inner;
                let slice = &v_data[off..off + inner];
                let norm_sq: f32 = slice.iter().map(|x| x * x).sum();
                let scale = g / norm_sq.sqrt().max(1e-12);
                for j in 0..inner {
                    w[off + j] = slice[j] * scale;
                }
            }
            Ok(tensor_data(
                encode_float(target_float_dtype, &w),
                data_shape.clone(),
                target_dtype,
            ))
        });
        let key = format!("{stem}.weight");
        tensors.insert(
            key.clone(),
            TensorSnapshot::from_closure(
                data_fn,
                target_dtype,
                shape,
                key.split('.').map(str::to_owned).collect(),
                vec![],
                Default::default(),
            ),
        );
    }
    if let Some(key) = tensors.keys().find(|k| k.ends_with(".weight_g")) {
        return Err(Error::Other(format!("weight_g without weight_v: {key}")));
    }

    let mut out = HashMap::new();
    for (name, mut snapshot) in tensors {
        let mapped = match remap {
            Some(f) => match f(&name) {
                Some(mapped) => mapped,
                None => continue,
            },
            None => name,
        };
        let key = format!("{}{mapped}", prefix.unwrap_or_default());
        snapshot.path_stack = Some(key.split('.').map(str::to_owned).collect());
        let dtype = burn_dtype_to_safetensors(snapshot.dtype)?;
        if matches!(dtype, Dtype::F32 | Dtype::F16 | Dtype::BF16) && dtype != target_float_dtype {
            let source = snapshot.clone_data_fn();
            let shape = snapshot.shape.clone();
            let data_fn = Rc::new(move || {
                let data = source()?;
                Ok(tensor_data(
                    convert_float_bytes(dtype, target_float_dtype, data.as_bytes()),
                    shape.clone(),
                    target_dtype,
                ))
            });
            snapshot = TensorSnapshot::from_closure(
                data_fn,
                target_dtype,
                snapshot.shape,
                snapshot.path_stack.unwrap(),
                vec![],
                Default::default(),
            );
        }
        out.insert(key, snapshot);
    }
    fuse_projections(&mut out, &["q_proj", "k_proj", "v_proj"], "qkv_proj", false)?;
    fuse_projections(&mut out, &["gate_proj", "up_proj"], "gate_up_proj", true)?;
    Ok(out.into_values().collect())
}

fn snapshot_error(error: Error) -> TensorSnapshotError {
    TensorSnapshotError::DataError(error.to_string())
}

fn tensor_data(bytes: Vec<u8>, shape: Vec<usize>, dtype: DType) -> TensorData {
    TensorData {
        bytes: Bytes::from_bytes_vec(bytes),
        shape,
        dtype,
    }
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

/// Single-pass float dtype conversion that writes directly into a fresh
/// `Vec<u8>` of the target size. Avoids the intermediate `Vec<f32>` that
/// `decode_f32` + `encode_float` would allocate (~2× peak memory of the
/// destination, multi-GB for the main model).
fn convert_float_bytes(src: Dtype, dst: Dtype, data: &[u8]) -> Vec<u8> {
    let n_elems = match src {
        Dtype::F32 => data.len() / 4,
        Dtype::F16 | Dtype::BF16 => data.len() / 2,
        _ => unreachable!("convert_float_bytes called with non-float src dtype"),
    };
    let elem_size = match dst {
        Dtype::F32 => 4,
        Dtype::F16 | Dtype::BF16 => 2,
        _ => unreachable!("convert_float_bytes called with non-float dst dtype"),
    };
    let mut out = vec![0u8; n_elems * elem_size];

    macro_rules! pump {
        ($read:expr, $write:expr, $src_step:expr, $dst_step:expr) => {{
            let mut s = 0usize;
            let mut d = 0usize;
            while s < data.len() {
                let v: f32 = $read(&data[s..s + $src_step]);
                let bytes = $write(v);
                out[d..d + $dst_step].copy_from_slice(&bytes);
                s += $src_step;
                d += $dst_step;
            }
        }};
    }

    let read_f32 = |b: &[u8]| f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
    let read_f16 = |b: &[u8]| f16::from_le_bytes([b[0], b[1]]).to_f32();
    let read_bf16 = |b: &[u8]| bf16::from_le_bytes([b[0], b[1]]).to_f32();
    let write_f32 = |v: f32| v.to_le_bytes();
    let write_f16 = |v: f32| f16::from_f32(v).to_le_bytes();
    let write_bf16 = |v: f32| bf16::from_f32(v).to_le_bytes();

    match (src, dst) {
        (Dtype::F32, Dtype::F16) => pump!(read_f32, write_f16, 4, 2),
        (Dtype::F32, Dtype::BF16) => pump!(read_f32, write_bf16, 4, 2),
        (Dtype::F16, Dtype::F32) => pump!(read_f16, write_f32, 2, 4),
        (Dtype::F16, Dtype::BF16) => pump!(read_f16, write_bf16, 2, 2),
        (Dtype::BF16, Dtype::F32) => pump!(read_bf16, write_f32, 2, 4),
        (Dtype::BF16, Dtype::F16) => pump!(read_bf16, write_f16, 2, 2),
        _ => unreachable!("same-dtype conversion should have been short-circuited"),
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

/// Fuse output rows in checkpoint order, before PyTorchToBurnAdapter performs
/// the Linear [out, in] -> [in, out] transpose. Each source is materialized
/// and released separately; only the fused output survives this closure.
fn fuse_projections(
    tensors: &mut HashMap<String, TensorSnapshot>,
    projections: &[&str],
    fused_name: &str,
    equal_rows: bool,
) -> Result<()> {
    let suffix = format!(".{}.weight", projections[0]);
    let first_keys: Vec<_> = tensors
        .keys()
        .filter(|k| k.ends_with(&suffix))
        .cloned()
        .collect();
    for first_key in first_keys {
        let stem = first_key.strip_suffix(&suffix).unwrap();
        let keys: Vec<_> = projections
            .iter()
            .map(|p| format!("{stem}.{p}.weight"))
            .collect();
        // Preserve unrelated/incomplete projection groups as individual tensors.
        if !keys.iter().all(|key| tensors.contains_key(key)) {
            continue;
        }
        let sources: Vec<_> = keys
            .iter()
            .map(|key| tensors.remove(key).unwrap())
            .collect();
        let first = &sources[0];
        if sources
            .iter()
            .any(|s| s.dtype != first.dtype || s.shape.len() != 2)
            || sources.iter().any(|s| {
                s.shape[1] != first.shape[1] || (equal_rows && s.shape[0] != first.shape[0])
            })
        {
            return Err(Error::Other(format!(
                "{fused_name} fusion dtype/shape mismatch at {stem}"
            )));
        }
        let dtype = first.dtype;
        let shape = vec![sources.iter().map(|s| s.shape[0]).sum(), first.shape[1]];
        let data_shape = shape.clone();
        let byte_len = sources.iter().map(TensorSnapshot::data_len).sum();
        let data_fn = Rc::new(move || {
            let mut bytes = Vec::with_capacity(byte_len);
            for source in &sources {
                bytes.extend_from_slice(source.to_data()?.as_bytes());
            }
            Ok(tensor_data(bytes, data_shape.clone(), dtype))
        });
        let key = format!("{stem}.{fused_name}.weight");
        tensors.insert(
            key.clone(),
            TensorSnapshot::from_closure(
                data_fn,
                dtype,
                shape,
                key.split('.').map(str::to_owned).collect(),
                vec![],
                Default::default(),
            ),
        );
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    fn snapshot(name: &str, shape: &[usize], values: &[f32], dtype: DType) -> TensorSnapshot {
        TensorSnapshot::from_data(
            TensorData::new(values.to_vec(), shape.to_vec()).convert_dtype(dtype),
            name.split('.').map(str::to_owned).collect(),
            vec![],
            Default::default(),
        )
    }

    fn prepare(sources: Vec<TensorSnapshot>, dtype: Dtype) -> HashMap<String, TensorSnapshot> {
        prepare_snapshots(
            sources.into_iter().map(|s| (s.full_path(), s)).collect(),
            None,
            None,
            dtype,
        )
        .unwrap()
        .into_iter()
        .map(|s| (s.full_path(), s))
        .collect()
    }

    #[test]
    fn conversions_are_lazy_and_preserve_float_values() {
        for source_dtype in [DType::F32, DType::F16, DType::BF16] {
            for target_dtype in [Dtype::F32, Dtype::F16, Dtype::BF16] {
                let calls = Rc::new(Cell::new(0));
                let counter = calls.clone();
                let source = snapshot("weight", &[4], &[0., -2., 0.5, 16.], source_dtype);
                let s = TensorSnapshot::from_closure(
                    Rc::new(move || {
                        counter.set(counter.get() + 1);
                        source.to_data()
                    }),
                    source_dtype,
                    vec![4],
                    vec!["weight".into()],
                    vec![],
                    Default::default(),
                );
                let out = prepare(vec![s], target_dtype);
                assert_eq!(calls.get(), 0, "preparation must never read tensor bytes");
                let data = out["weight"].to_data().unwrap();
                assert_eq!(burn_dtype_to_safetensors(data.dtype).unwrap(), target_dtype);
                assert_eq!(
                    decode_f32(target_dtype, data.as_bytes()).unwrap(),
                    vec![0., -2., 0.5, 16.]
                );
                assert_eq!(calls.get(), 1);
            }
        }
    }

    #[test]
    fn fuses_qkv_and_gate_up_in_checkpoint_row_order() {
        let out = prepare(
            vec![
                snapshot("attn.v_proj.weight", &[1, 2], &[7., 8.], DType::BF16),
                snapshot("mlp.up_proj.weight", &[1, 2], &[11., 12.], DType::F16),
                snapshot(
                    "attn.q_proj.weight",
                    &[2, 2],
                    &[1., 2., 3., 4.],
                    DType::BF16,
                ),
                snapshot("attn.k_proj.weight", &[1, 2], &[5., 6.], DType::BF16),
                snapshot("mlp.gate_proj.weight", &[1, 2], &[9., 10.], DType::F16),
            ],
            Dtype::F32,
        );
        assert_eq!(out.len(), 2);
        let qkv = out["attn.qkv_proj.weight"].to_data().unwrap();
        assert_eq!(qkv.shape, vec![4, 2]);
        assert_eq!(
            qkv.to_vec::<f32>().unwrap(),
            vec![1., 2., 3., 4., 5., 6., 7., 8.]
        );
        assert_eq!(
            out["mlp.gate_up_proj.weight"]
                .to_data()
                .unwrap()
                .to_vec::<f32>()
                .unwrap(),
            vec![9., 10., 11., 12.]
        );
    }

    #[test]
    fn materializes_weight_norm_and_remaps_audio_vae() {
        let sources = [
            snapshot(
                "decoder.model.0.weight_v",
                &[2, 1, 2],
                &[3., 4., 0., 0.],
                DType::F32,
            ),
            snapshot(
                "decoder.model.0.weight_g",
                &[2, 1, 1],
                &[10., 2.],
                DType::F32,
            ),
            snapshot("decoder.sr_bin_boundaries", &[1], &[42.], DType::F32),
        ];
        let out = prepare_snapshots(
            sources.into_iter().map(|s| (s.full_path(), s)).collect(),
            Some("audio_vae."),
            Some(remap_audiovae_key),
            Dtype::BF16,
        )
        .unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].full_path(), "audio_vae.decoder.first.dw.conv.weight");
        let data = out[0].to_data().unwrap();
        assert_eq!(data.shape, vec![2, 1, 2]);
        assert_eq!(
            decode_f32(Dtype::BF16, data.as_bytes()).unwrap(),
            vec![6., 8., 0., 0.]
        );
    }

    #[test]
    fn rejects_orphan_norm_weights_and_incompatible_fusion() {
        for name in ["conv.weight_v", "conv.weight_g"] {
            let s = snapshot(name, &[1], &[1.], DType::F32);
            assert!(
                prepare_snapshots([(name.to_string(), s)].into(), None, None, Dtype::F32).is_err()
            );
        }
        for (shape, values) in [(vec![], vec![1.]), (vec![2, 1], vec![1., 2.])] {
            let sources = [
                snapshot("mlp.gate_proj.weight", &[1, 1], &[1.], DType::F32),
                snapshot("mlp.up_proj.weight", &shape, &values, DType::F32),
            ];
            assert!(
                prepare_snapshots(
                    sources.into_iter().map(|s| (s.full_path(), s)).collect(),
                    None,
                    None,
                    Dtype::F32
                )
                .is_err()
            );
        }
    }

    #[test]
    fn incomplete_projection_groups_and_integer_buffers_are_preserved() {
        let int = TensorSnapshot::from_data(
            TensorData::new(vec![1i64, 2], [2]),
            vec!["positions".into()],
            vec![],
            Default::default(),
        );
        let out = prepare(
            vec![
                int,
                snapshot("attn.q_proj.weight", &[1, 2], &[1., 2.], DType::F32),
            ],
            Dtype::F16,
        );
        assert_eq!(out.len(), 2);
        assert!(out.contains_key("attn.q_proj.weight"));
        assert_eq!(
            out["positions"].to_data().unwrap().to_vec::<i64>().unwrap(),
            vec![1, 2]
        );
    }

    #[test]
    fn strips_pytorch_container_prefixes() {
        for prefix in ["state_dict.", "model_state_dict.", "module.", ""] {
            assert_eq!(
                strip_pth_top_level(&format!("{prefix}layer.weight")),
                "layer.weight"
            );
        }
    }

    struct TempCheckpoint(std::path::PathBuf);
    impl TempCheckpoint {
        fn new() -> Self {
            static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            let path = std::env::temp_dir().join(format!(
                "voxcpm-weights-{}-{}.safetensors",
                std::process::id(),
                NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            ));
            Self(path)
        }
    }
    impl Drop for TempCheckpoint {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    #[test]
    fn file_snapshots_keep_their_mmap_alive_after_store_drop() {
        let file = TempCheckpoint::new();
        let bytes = encode_float(Dtype::BF16, &[1., 2., 3., 4.]);
        let view = safetensors::tensor::TensorView::new(Dtype::BF16, vec![2, 2], &bytes).unwrap();
        safetensors::serialize_to_file([("weight", view)], &None, &file.0).unwrap();
        let sources = {
            let mut store = SafetensorsStore::from_file(&file.0);
            store
                .get_all_snapshots()
                .unwrap()
                .iter()
                .map(|(k, s)| (k.clone(), s.clone()))
                .collect()
        };
        let snapshots = prepare_snapshots(sources, None, None, Dtype::F32).unwrap();
        assert_eq!(
            snapshots[0].to_data().unwrap().to_vec::<f32>().unwrap(),
            vec![1., 2., 3., 4.]
        );
    }

    #[cfg(feature = "cpu")]
    #[derive(Module, Debug)]
    struct ProjectionModule<B: Backend> {
        attn: AttentionModule<B>,
        untouched: burn::nn::Linear<B>,
    }
    #[cfg(feature = "cpu")]
    #[derive(Module, Debug)]
    struct AttentionModule<B: Backend> {
        qkv_proj: burn::nn::Linear<B>,
    }

    #[test]
    #[cfg(feature = "cpu")]
    fn applies_fused_linear_with_transpose_without_reading_unused_tensors() {
        type B = burn::backend::NdArray<f32>;
        let device = Default::default();
        let mut model = ProjectionModule::<B> {
            attn: AttentionModule {
                qkv_proj: burn::nn::LinearConfig::new(2, 4)
                    .with_bias(false)
                    .init(&device),
            },
            untouched: burn::nn::LinearConfig::new(2, 2)
                .with_bias(false)
                .init(&device),
        };
        let unread = TensorSnapshot::from_closure(
            Rc::new(|| panic!("unused tensor must remain lazy")),
            DType::F32,
            vec![1],
            vec!["unused".into()],
            vec![],
            Default::default(),
        );
        let sources = [
            snapshot(
                "attn.q_proj.weight",
                &[2, 2],
                &[1., 2., 3., 4.],
                DType::BF16,
            ),
            snapshot("attn.k_proj.weight", &[1, 2], &[5., 6.], DType::BF16),
            snapshot("attn.v_proj.weight", &[1, 2], &[7., 8.], DType::BF16),
            unread,
        ];
        let result = prepare_and_apply(
            &mut model,
            Path::new("synthetic"),
            sources.into_iter().map(|s| (s.full_path(), s)).collect(),
            None,
            None,
            Dtype::F32,
        )
        .unwrap();
        assert!(result.errors.is_empty(), "{:?}", result.errors);
        assert_eq!(result.applied, vec!["attn.qkv_proj.weight"]);
        assert!(
            result
                .missing
                .iter()
                .any(|(key, _)| key == "untouched.weight")
        );
        assert_eq!(result.unused, vec!["unused"]);
        let data = model.attn.qkv_proj.weight.val().to_data();
        assert_eq!(data.shape, vec![2, 4]);
        assert_eq!(
            data.to_vec::<f32>().unwrap(),
            vec![1., 3., 5., 7., 2., 4., 6., 8.]
        );
    }

    #[test]
    #[cfg(feature = "cpu")]
    fn apply_rejects_shape_mismatches_and_deferred_read_errors() {
        type B = burn::backend::NdArray<f32>;
        for malformed_shape in [true, false] {
            let mut model = burn::nn::LinearConfig::new(2, 2)
                .with_bias(false)
                .init::<B>(&Default::default());
            let source = if malformed_shape {
                snapshot("weight", &[1, 2], &[1., 2.], DType::F32)
            } else {
                TensorSnapshot::from_closure(
                    Rc::new(|| Err(TensorSnapshotError::IoError("broken storage".into()))),
                    DType::F32,
                    vec![2, 2],
                    vec!["weight".into()],
                    vec![],
                    Default::default(),
                )
            };
            assert!(
                prepare_and_apply(
                    &mut model,
                    Path::new("synthetic"),
                    [("weight".into(), source)].into(),
                    None,
                    None,
                    Dtype::F32
                )
                .is_err()
            );
        }
    }

    #[test]
    fn rejects_data_that_disagrees_with_lazy_metadata() {
        let source = TensorSnapshot::from_closure(
            Rc::new(|| Ok(TensorData::new(vec![1f32, 2.], [2]))),
            DType::BF16,
            vec![2],
            vec!["weight".into()],
            vec![],
            Default::default(),
        );
        let out = prepare(vec![source], Dtype::F32);
        assert!(out["weight"].to_data().is_err());
    }

    #[test]
    fn pytorch_snapshots_keep_storage_alive_after_reader_drop() {
        let sources = {
            let reader = burn_store::pytorch::PytorchReader::new(
                Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-state-dict.pth")
            ).unwrap();
            reader.tensors().iter().map(|(name, snapshot)|
                (strip_pth_top_level(name).to_string(), snapshot.clone())).collect()
        };
        let snapshots = prepare_snapshots(sources, None, None, Dtype::F16).unwrap();
        assert_eq!(snapshots.len(), 1);
        assert_eq!(snapshots[0].full_path(), "linear.weight");
        let data = snapshots[0].to_data().unwrap();
        assert_eq!(data.shape, vec![2, 3]);
        assert_eq!(decode_f32(Dtype::F16, data.as_bytes()).unwrap(), vec![1., 2., 3., 4., 5., 6.]);
    }

}
