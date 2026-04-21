//! Neural-network architecture of VoxCPM2: the top-level [`VoxCpm2Model`] that
//! composes the base and residual MiniCPM-4 LMs, the local feature encoder, the
//! diffusion decoder, the scalar-quantization layer, the projection layers, the
//! stop-prediction head, and the AudioVAE.

use crate::audiovae::AudioVae;
use crate::config::VoxCpm2Config;
use crate::fsq::ScalarQuantizationLayer;
use crate::locdit::{UnifiedCfm, VoxCpmLocDiTV2};
use crate::locenc::VoxCpmLocEnc;
use crate::minicpm4::MiniCpmModel;
use burn::module::Ignored;
use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;

/// Special audio tokens added to the text vocabulary.
pub const AUDIO_START_TOKEN: i64 = 101;
pub const AUDIO_END_TOKEN: i64 = 102;
pub const REF_AUDIO_START_TOKEN: i64 = 103;
pub const REF_AUDIO_END_TOKEN: i64 = 104;

#[derive(Module, Debug)]
pub struct VoxCpm2Model<B: Backend> {
    pub base_lm: MiniCpmModel<B>,
    pub residual_lm: MiniCpmModel<B>,
    pub feat_encoder: VoxCpmLocEnc<B>,
    pub feat_decoder: UnifiedCfm<B>,
    pub fsq_layer: ScalarQuantizationLayer<B>,

    pub enc_to_lm_proj: Linear<B>,
    pub lm_to_dit_proj: Linear<B>,
    pub res_to_dit_proj: Linear<B>,
    pub fusion_concat_proj: Linear<B>,

    pub stop_proj: Linear<B>,
    pub stop_head: Linear<B>,

    pub audio_vae: AudioVae<B>,
    pub config: Ignored<VoxCpm2Config>,
}

impl<B: Backend> VoxCpm2Model<B> {
    pub fn new(config: VoxCpm2Config, device: &B::Device) -> Self {
        let lm_hidden = config.lm_config.hidden_size;
        let enc_hidden = config.encoder_config.hidden_dim;
        let dit_hidden = config.dit_config.hidden_dim;

        let audio_vae_config = config
            .audio_vae_config
            .clone()
            .unwrap_or_default();

        let base_lm = MiniCpmModel::new(config.lm_config.clone(), device);
        let residual_lm = MiniCpmModel::new(config.residual_lm_config(), device);

        let feat_encoder = VoxCpmLocEnc::new(config.encoder_lm_config(), config.feat_dim, device);

        let estimator = VoxCpmLocDiTV2::new(config.dit_lm_config(), config.feat_dim, device);
        let feat_decoder = UnifiedCfm::new(
            config.feat_dim,
            estimator,
            config.dit_config.cfm_config.sigma_min as f64,
            config.dit_config.cfm_config.inference_cfg_rate as f64,
            config.dit_config.dit_mean_mode,
        );

        let fsq_layer = ScalarQuantizationLayer::new(
            lm_hidden,
            lm_hidden,
            config.scalar_quantization_latent_dim,
            config.scalar_quantization_scale,
            device,
        );

        let enc_to_lm_proj = LinearConfig::new(enc_hidden, lm_hidden).init(device);
        let lm_to_dit_proj = LinearConfig::new(lm_hidden, dit_hidden).init(device);
        let res_to_dit_proj = LinearConfig::new(lm_hidden, dit_hidden).init(device);
        let fusion_concat_proj = LinearConfig::new(lm_hidden * 2, lm_hidden).init(device);

        let stop_proj = LinearConfig::new(lm_hidden, lm_hidden).init(device);
        let stop_head = LinearConfig::new(lm_hidden, 2).with_bias(false).init(device);

        let audio_vae = AudioVae::new(audio_vae_config, device);

        Self {
            base_lm,
            residual_lm,
            feat_encoder,
            feat_decoder,
            fsq_layer,
            enc_to_lm_proj,
            lm_to_dit_proj,
            res_to_dit_proj,
            fusion_concat_proj,
            stop_proj,
            stop_head,
            audio_vae,
            config: Ignored(config),
        }
    }

    pub fn sample_rate(&self) -> usize {
        self.audio_vae.out_sample_rate()
    }

    pub fn patch_size(&self) -> usize {
        self.config.0.patch_size
    }

    pub fn latent_dim(&self) -> usize {
        self.config.0.audio_vae_config.as_ref().map(|c| c.latent_dim).unwrap_or(64)
    }

    fn scale_emb(&self) -> f64 {
        if self.config.0.lm_config.use_mup {
            self.config.0.lm_config.scale_emb as f64
        } else {
            1.0
        }
    }

    /// Core inference loop.
    ///
    /// Runs the text+audio-mask prefill through the base LM and residual LM,
    /// then iteratively samples audio feature patches via the diffusion
    /// decoder until the stop head fires (or `max_len` is reached).
    ///
    /// * `text_token`: `[B=1, S]` int tokens.
    /// * `text_mask`, `feat_mask`: `[1, S]` float masks (0/1) indicating which
    ///   positions are text and which are audio patches.
    /// * `feat`: `[1, S, P, D]` audio latent patches (zeros at text positions).
    ///
    /// Returns `[B=1, D, T*P]` — the concatenated latent feature sequence.
    pub fn inference(
        &self,
        text_token: Tensor<B, 2, burn::tensor::Int>,
        text_mask: Tensor<B, 2>,
        feat: Tensor<B, 4>,
        feat_mask: Tensor<B, 2>,
        min_len: usize,
        max_len: usize,
        inference_timesteps: usize,
        cfg_value: f64,
    ) -> Tensor<B, 3> {
        let device = feat.device();
        let [_b, _s, p, d] = feat.dims();
        let patch_size = self.patch_size();

        // 1) Encode audio feature patches.
        let feat_embed = self.feat_encoder.forward(feat.clone()); // [B, S, enc_h]
        let feat_embed = self.enc_to_lm_proj.forward(feat_embed); // [B, S, lm_h]

        // 2) Embed text tokens.
        let scale = self.scale_emb();
        let text_embed = self.base_lm.embed(text_token).mul_scalar(scale); // [B, S, lm_h]

        // 3) Combine via masks.
        let text_mask3: Tensor<B, 3> = text_mask.clone().unsqueeze_dim(2);
        let feat_mask3: Tensor<B, 3> = feat_mask.clone().unsqueeze_dim(2);
        let combined = text_embed * text_mask3.clone() + feat_embed.clone() * feat_mask3.clone();

        // 4) Prefix feat cond (last patch).
        let s = feat.dims()[1];
        let prefix_feat: Tensor<B, 3> = feat.clone().narrow(1, s - 1, 1).squeeze_dim::<3>(1); // [B, P, D]

        // 5) Base LM prefill.
        let (enc_outputs, base_kv) = self.base_lm.forward(combined, true);
        let enc_outputs = self.fsq_layer.forward(enc_outputs.clone()) * feat_mask3.clone()
            + enc_outputs * text_mask3;
        let lm_hidden_prefill = enc_outputs.clone();

        // 6) Residual LM prefill.
        let residual_input = self.fusion_concat_proj.forward(Tensor::cat(
            vec![enc_outputs, feat_embed.clone() * feat_mask3],
            2,
        ));
        let (residual_outputs, residual_kv) = self.residual_lm.forward(residual_input, true);

        // Seed caches with the prefill K/V.
        let s_ctx = lm_hidden_prefill.dims()[1];
        let lm_config = self.config.0.lm_config.clone();
        let max_ctx = self.config.0.max_length.max(s_ctx + max_len);
        let mut base_cache = crate::minicpm4::StaticKvCache::new(
            lm_config.num_hidden_layers,
            lm_config.num_key_value_heads,
            lm_config.head_dim(),
            1,
            max_ctx,
            &device,
        );
        base_cache.fill(base_kv);
        let res_cfg = self.config.0.residual_lm_config();
        let mut res_cache = crate::minicpm4::StaticKvCache::new(
            res_cfg.num_hidden_layers,
            res_cfg.num_key_value_heads,
            res_cfg.head_dim(),
            1,
            max_ctx,
            &device,
        );
        res_cache.fill(residual_kv);

        // Take the last position for autoregressive start.
        let mut lm_hidden: Tensor<B, 2> =
            lm_hidden_prefill.narrow(1, s_ctx - 1, 1).squeeze_dim::<2>(1);
        let mut residual_hidden: Tensor<B, 2> =
            residual_outputs.narrow(1, s_ctx - 1, 1).squeeze_dim::<2>(1);

        let mut prefix_feat_cond = prefix_feat;
        let mut pred_feats: Vec<Tensor<B, 4>> = Vec::new();

        let profile = std::env::var("VOXCPM_PROFILE").is_ok();
        let mut t_dit_ns: u128 = 0;
        let mut t_stop_ns: u128 = 0;
        let mut t_lm_ns: u128 = 0;
        let mut n_steps: usize = 0;

        // Helper closure: force a GPU→CPU sync by reading a tiny scalar.
        // Only used under VOXCPM_PROFILE to turn async kernel dispatch into
        // real per-phase wall-clock measurements.
        let sync_barrier = |t: Tensor<B, 2>| {
            let _ = t.slice([0..1, 0..1]).into_data();
        };

        for i in 0..max_len {
            // DiT inputs: concat(lm_to_dit(lm), res_to_dit(res))
            let dit1 = self.lm_to_dit_proj.forward(lm_hidden.clone()); // [B, dit_h]
            let dit2 = self.res_to_dit_proj.forward(residual_hidden.clone()); // [B, dit_h]
            let dit_hidden = Tensor::cat(vec![dit1, dit2], 1); // [B, 2*dit_h]

            let t0 = profile.then(std::time::Instant::now);

            // Diffusion sample: [B, D, P]
            let pred = self.feat_decoder.forward(
                dit_hidden,
                inference_timesteps,
                patch_size,
                prefix_feat_cond.clone().swap_dims(1, 2),
                1.0,
                cfg_value,
                1.0,
                true,
            );
            // -> [B, P, D]
            let pred_feat = pred.swap_dims(1, 2);

            // Re-encode pred_feat through feat_encoder: need shape [B, 1, P, D]
            let pred4: Tensor<B, 4> = pred_feat.clone().unsqueeze_dim(1);
            pred_feats.push(pred4.clone());
            prefix_feat_cond = pred_feat.clone();

            // Profile barrier: force GPU sync so `t_dit_ns` measures real
            // DiT wall-clock time and not just CPU-side kernel enqueue.
            if profile {
                sync_barrier(pred_feat.clone().narrow(2, 0, 1).squeeze_dim(2));
            }
            let t1 = profile.then(std::time::Instant::now);

            // Stop check.
            let stop_logits = self
                .stop_head
                .forward(burn::tensor::activation::silu(self.stop_proj.forward(lm_hidden.clone()))); // [B, 2]
            let stop_arg = stop_logits.clone().argmax(1);
            // Backend-agnostic int read: argmax tensor's elem type may be i32 (wgpu)
            // or i64 (ndarray). `into_data()` yields a TensorData; use `iter::<i64>()`
            // which converts whatever the underlying dtype is.
            //
            // NOTE: `into_data()` is itself a GPU→CPU sync; for profile we
            // measure the interval around it as the stop-check cost.
            let stop: i64 = stop_arg
                .into_data()
                .iter::<i64>()
                .next()
                .unwrap_or(0);
            let t2 = profile.then(std::time::Instant::now);
            if std::env::var("VOXCPM_DEBUG_STOP").is_ok() {
                // Dtype-agnostic: convert whatever the backend emits to f32.
                let d = stop_logits.into_data().convert::<f32>();
                let sl = d.as_slice::<f32>().unwrap_or(&[]);
                let lh = lm_hidden.clone().into_data().convert::<f32>();
                let lhs = lh.as_slice::<f32>().unwrap_or(&[]);
                let lh_abs_max = lhs.iter().fold(0f32, |a, &b| a.max(b.abs()));
                let lh_first: Vec<f32> = lhs.iter().take(4).copied().collect();
                eprintln!("step {i:4} stop={stop} logits=[{:.3}, {:.3}] lm_abs_max={:.3} lm[:4]={:?}", sl.first().copied().unwrap_or(0.0), sl.get(1).copied().unwrap_or(0.0), lh_abs_max, lh_first);
            }
            if i > min_len && stop == 1 {
                if let (Some(t0), Some(t1), Some(t2)) = (t0, t1, t2) {
                    t_dit_ns += t1.duration_since(t0).as_nanos();
                    t_stop_ns += t2.duration_since(t1).as_nanos();
                    n_steps += 1;
                }
                break;
            }

            // Encode the single predicted patch.
            let curr_embed = self.feat_encoder.forward(pred4); // [B, 1, enc_h]
            let curr_embed = self.enc_to_lm_proj.forward(curr_embed); // [B, 1, lm_h]
            let curr_embed2: Tensor<B, 2> = curr_embed.clone().squeeze_dim::<2>(1); // [B, lm_h]

            // Step base LM.
            let pos = base_cache.step();
            lm_hidden = self.base_lm.forward_step(curr_embed2.clone(), pos, &mut base_cache);
            lm_hidden = self.fsq_layer.forward(lm_hidden);

            // Step residual LM.
            let res_input2 = self
                .fusion_concat_proj
                .forward(Tensor::cat(vec![lm_hidden.clone(), curr_embed2], 1));
            let pos = res_cache.step();
            residual_hidden = self.residual_lm.forward_step(res_input2, pos, &mut res_cache);

            if let (Some(t0), Some(t1), Some(t2)) = (t0, t1, t2) {
                // Force sync so `t_lm_ns` captures actual LM step wall time.
                sync_barrier(residual_hidden.clone());
                let t3 = std::time::Instant::now();
                t_dit_ns += t1.duration_since(t0).as_nanos();
                t_stop_ns += t2.duration_since(t1).as_nanos();
                t_lm_ns += t3.duration_since(t2).as_nanos();
                n_steps += 1;
            }
        }

        if profile && n_steps > 0 {
            let ms = |ns: u128| (ns as f64) / 1e6;
            eprintln!(
                "[profile] AR steps={} dit={:.1}ms stop_sync={:.1}ms lm_tail={:.1}ms avg_per_step: dit={:.2}ms stop={:.2}ms lm={:.2}ms",
                n_steps, ms(t_dit_ns), ms(t_stop_ns), ms(t_lm_ns),
                ms(t_dit_ns) / n_steps as f64,
                ms(t_stop_ns) / n_steps as f64,
                ms(t_lm_ns) / n_steps as f64,
            );
        }

        // Stack predictions [B, T, P, D] -> [B, D, T*P] (matches Python's
        // einops `rearrange(..., "b t p d -> b d (t p)")`, where the flat
        // index `k = t * P + p`).
        let feats = Tensor::cat(pred_feats, 1);
        let [b, t, p2, d2] = feats.dims();
        debug_assert_eq!(p2, p);
        debug_assert_eq!(d2, d);
        // Permute [B,T,P,D] -> [B,D,T,P] via swap(1,3) then swap(2,3).
        feats.swap_dims(1, 3).swap_dims(2, 3).reshape([b, d, t * p])
    }
}

