//! Synthetic host-allocation regression test, independent of model downloads
//! and GPU availability. This measures heap allocation, not process RSS/VRAM.
#![cfg(feature = "cpu")]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use half::bf16;
use safetensors::{Dtype, tensor::TensorView};

struct CountingAllocator;
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

fn record_allocation(size: usize) {
    let live = LIVE.fetch_add(size, Ordering::Relaxed) + size;
    PEAK.fetch_max(live, Ordering::Relaxed);
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            record_allocation(layout.size());
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            record_allocation(layout.size());
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let ptr = unsafe { System.realloc(ptr, layout, new_size) };
        if !ptr.is_null() {
            LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
            record_allocation(new_size);
        }
        ptr
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

#[derive(Module, Debug)]
struct Model<B: Backend> {
    layers: Vec<Linear<B>>,
}

struct Checkpoint(std::path::PathBuf);
impl Drop for Checkpoint {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn load_does_not_allocate_a_second_converted_checkpoint() {
    const LAYERS: usize = 64;
    const DIM: usize = 256;
    const WEIGHT_BYTES: usize = LAYERS * DIM * DIM * size_of::<f32>();
    type B = burn::backend::NdArray<f32>;

    let directory = std::env::temp_dir().join(format!("voxcpm-memory-{}", std::process::id()));
    std::fs::create_dir(&directory).unwrap();
    let checkpoint = Checkpoint(directory);
    // Every file entry borrows the same small source buffer during streaming
    // serialization, so fixture generation doesn't set a whole-checkpoint peak.
    {
        let bytes: Vec<_> = (0..DIM * DIM)
            .flat_map(|_| bf16::from_f32(0.5).to_le_bytes())
            .collect();
        let views: Vec<_> = (0..LAYERS)
            .map(|i| {
                (
                    format!("layers.{i}.weight"),
                    TensorView::new(Dtype::BF16, vec![DIM, DIM], &bytes).unwrap(),
                )
            })
            .collect();
        safetensors::serialize_to_file(views, &None, &checkpoint.0.join("model.safetensors"))
            .unwrap();
    }
    let mut model = Model::<B> {
        layers: (0..LAYERS)
            .map(|_| {
                LinearConfig::new(DIM, DIM)
                    .with_bias(false)
                    .init(&Default::default())
            })
            .collect(),
    };
    let baseline = LIVE.load(Ordering::Relaxed);
    PEAK.store(baseline, Ordering::Relaxed);
    let result = voxcpm_rs::weights::load_pretrained(&mut model, &checkpoint.0).unwrap();
    let extra_peak = PEAK.load(Ordering::Relaxed).saturating_sub(baseline);
    assert_eq!(result.applied.len(), LAYERS);
    assert!(result.errors.is_empty());
    eprintln!("peak additional heap: {extra_peak} bytes; loaded F32 weights: {WEIGHT_BYTES} bytes");
    // Leave room for one tensor's conversion/transpose and loader metadata,
    // but fail if a complete second F32 checkpoint is retained or serialized.
    assert!(
        extra_peak < WEIGHT_BYTES * 3 / 2,
        "loading used {extra_peak} additional heap bytes for {WEIGHT_BYTES} weight bytes"
    );
    assert_eq!(
        model.layers[0]
            .weight
            .val()
            .to_data()
            .to_vec::<f32>()
            .unwrap()[0],
        0.5
    );
}
