use crate::minicpm4::MiniCpmModel;
use crate::omnivoice::config::{OmniVoiceConfig, OmniVoiceGenerationConfig};
use crate::omnivoice::duration::RuleDurationEstimator;
use crate::tokenizer::TextTokenizer;
use crate::Result;

use burn::module::Ignored;
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::{Int, Bool, TensorData};

// Seedable simple pseudo-random number generator for CPU Gumbel sampling
struct Xorshift {
    state: u32,
}

impl Xorshift {
    fn new(seed: u32) -> Self {
        Self {
            state: if seed == 0 { 1 } else { seed },
        }
    }

    fn next_f32(&mut self) -> f32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.state = x;
        (x as f32) / (u32::MAX as f32)
    }
}

fn get_time_steps(num_step: usize, t_shift: f32) -> Vec<f32> {
    let mut timesteps = Vec::with_capacity(num_step + 1);
    for i in 0..=num_step {
        let t = (i as f32) / (num_step as f32);
        let shifted = t_shift * t / (1.0 + (t_shift - 1.0) * t);
        timesteps.push(shifted);
    }
    timesteps
}

fn stable_log_softmax<B: Backend, const D: usize>(tensor: Tensor<B, D>, dim: usize) -> Tensor<B, D> {
    let max = tensor.clone().max_dim(dim);
    let centered = tensor - max;
    let sum_exp = centered.clone().exp().sum_dim(dim);
    let log_sum = sum_exp.log();
    centered - log_sum
}

#[derive(Module, Debug)]
pub struct OmniVoiceModel<B: Backend> {
    pub llm: MiniCpmModel<B>,
    pub audio_embeddings: Embedding<B>,
    pub audio_heads: Linear<B>,
    pub config: Ignored<OmniVoiceConfig>,
}

impl<B: Backend> OmniVoiceModel<B> {
    pub fn new(config: OmniVoiceConfig, device: &B::Device) -> Self {
        let llm = MiniCpmModel::new(config.llm_config.clone(), device);
        let audio_embeddings = EmbeddingConfig::new(
            config.num_audio_codebook * config.audio_vocab_size,
            config.llm_config.hidden_size,
        )
        .init(device);
        let audio_heads = LinearConfig::new(
            config.llm_config.hidden_size,
            config.num_audio_codebook * config.audio_vocab_size,
        )
        .with_bias(false)
        .init(device);

        Self {
            llm,
            audio_embeddings,
            audio_heads,
            config: Ignored(config),
        }
    }

    /// Forward pass of the OmniVoice model.
    /// `input_ids`: `[B, C, S]`
    /// `audio_mask`: `[B, S]`
    /// `attention_mask`: `Option<Tensor<B, 4, Bool>>`
    /// Returns logits: `[B, C, S, V]`
    pub fn forward(
        &self,
        input_ids: Tensor<B, 3, Int>,
        audio_mask: Tensor<B, 2, Bool>,
        attention_mask: Option<Tensor<B, 4, Bool>>,
    ) -> Tensor<B, 4> {
        let [b, c, s] = input_ids.dims();
        let device = input_ids.device();

        // 1. Text embeddings: layer 0 of input_ids
        let text_ids = input_ids.clone().narrow(1, 0, 1).squeeze_dim::<2>(1); // [B, S]
        let text_embeds = self.llm.embed(text_ids); // [B, S, H]

        // 2. Audio embeddings
        let audio_mask_unsqueezed = audio_mask.clone().unsqueeze_dim::<3>(1); // [B, 1, S]
        
        // Prepare offsets: [1, C, 1]
        let offsets_data = (0..c)
            .map(|i| (i * self.config.audio_vocab_size) as i64)
            .collect::<Vec<_>>();
        let offsets = Tensor::<B, 1, Int>::from_data(
            TensorData::new(offsets_data, [c]),
            &device,
        )
        .reshape([1, c, 1]);

        let shifted_ids = (input_ids * audio_mask_unsqueezed.clone().int()) + offsets.expand([b, c, s]);
        
        // Reshape to flat [B * C, S] to query standard 2D Embedding
        let shifted_ids_flat = shifted_ids.reshape([b * c, s]);
        let audio_embeds_flat = self.audio_embeddings.forward(shifted_ids_flat); // [B * C, S, H]
        
        let h = audio_embeds_flat.dims()[2];
        let audio_embeds_4d = audio_embeds_flat.reshape([b, c, s, h]);
        let audio_embeds = audio_embeds_4d.sum_dim(1).squeeze_dim::<3>(1); // [B, S, H]

        // 3. Combine embeds
        let audio_mask_bcast = audio_mask.unsqueeze_dim::<3>(2).expand([b, s, h]);
        let combined = text_embeds.mask_fill(audio_mask_bcast.clone(), 0.0)
            + audio_embeds.mask_fill(audio_mask_bcast.equal_elem(false), 0.0);

        // 4. Run LLM (not causal for iterative unmasking)
        let (hidden, _) = self.llm.forward(combined, false, attention_mask);

        // 5. Output projection
        let logits_flat = self.audio_heads.forward(hidden); // [B, S, C * V]
        let v = self.config.audio_vocab_size;
        logits_flat
            .reshape([b, s, c, v])
            .swap_dims(1, 2) // [B, C, S, V]
    }

    /// Iterative unmasking generation loop.
    pub fn generate_iterative(
        &self,
        texts: Vec<String>,
        target_lens: Vec<usize>,
        ref_texts: Vec<String>,
        ref_audio_tokens: Vec<Option<Tensor<B, 3, Int>>>,
        langs: Vec<Option<String>>,
        instructs: Vec<Option<String>>,
        gen_config: &OmniVoiceGenerationConfig,
        text_tokenizer: &TextTokenizer,
        cancel: Option<&crate::voxcpm2::wrapper::CancelToken>,
    ) -> Result<Vec<Tensor<B, 3, Int>>> {
        let bsz = texts.len();
        let device = self.llm.norm.devices()[0].clone();
        let c = self.config.num_audio_codebook;
        let v = self.config.audio_vocab_size;

        // 1. Prepare inputs for each batch item
        let mut inputs_list = Vec::with_capacity(bsz);
        for i in 0..bsz {
            let inp = self.prepare_inference_inputs(
                &texts[i],
                target_lens[i],
                ref_texts.get(i).map(|s| s.as_str()),
                ref_audio_tokens.get(i).and_then(|x| x.as_ref()),
                langs.get(i).and_then(|x| x.as_ref().map(|s| s.as_str())),
                instructs.get(i).and_then(|x| x.as_ref().map(|s| s.as_str())),
                gen_config.denoise,
                text_tokenizer,
                &device,
            )?;
            inputs_list.push(inp);
        }

        let c_lens: Vec<usize> = inputs_list.iter().map(|inp| inp.input_ids.dims()[2]).collect();
        let max_c_len = *c_lens.iter().max().unwrap_or(&0);
        let pad_id = self.config.audio_mask_id as i64;

        // Construct batched input tensors [2 * B, C, max_c_len] for CFG
        let mut batch_input_ids_cpu = vec![pad_id; 2 * bsz * c * max_c_len];
        let mut batch_audio_mask_cpu = vec![false; 2 * bsz * max_c_len];
        let mut batch_attention_mask_cpu = vec![true; 2 * bsz * max_c_len * max_c_len];

        for i in 0..bsz {
            let c_len = c_lens[i];
            let u_len = target_lens[i];
            let inp = &inputs_list[i];

            let inp_ids_data = inp.input_ids.clone().into_data().iter::<i64>().collect::<Vec<_>>();
            let inp_mask_data = inp.audio_mask.clone().into_data().iter::<bool>().collect::<Vec<_>>();

            // 1) Cond row (0..bsz)
            // copy input_ids [C, c_len]
            for codebook in 0..c {
                for seq_idx in 0..c_len {
                    let src_idx = codebook * c_len + seq_idx;
                    let dst_idx = (i * c + codebook) * max_c_len + seq_idx;
                    batch_input_ids_cpu[dst_idx] = inp_ids_data[src_idx];
                }
            }
            // copy audio_mask [c_len]
            for seq_idx in 0..c_len {
                let dst_idx = i * max_c_len + seq_idx;
                batch_audio_mask_cpu[dst_idx] = inp_mask_data[seq_idx];
            }
            // attention mask: [c_len, c_len] false (do not mask)
            for q in 0..c_len {
                for kv in 0..c_len {
                    let dst_idx = (i * max_c_len + q) * max_c_len + kv;
                    batch_attention_mask_cpu[dst_idx] = false;
                }
            }

            // 2) Uncond row (bsz..2*bsz)
            // copy input_ids last u_len tokens
            for codebook in 0..c {
                for seq_idx in 0..u_len {
                    let src_idx = codebook * c_len + (c_len - u_len + seq_idx);
                    let dst_idx = ((bsz + i) * c + codebook) * max_c_len + seq_idx;
                    batch_input_ids_cpu[dst_idx] = inp_ids_data[src_idx];
                }
            }
            // copy audio_mask last u_len
            for seq_idx in 0..u_len {
                let dst_idx = (bsz + i) * max_c_len + seq_idx;
                batch_audio_mask_cpu[dst_idx] = inp_mask_data[c_len - u_len + seq_idx];
            }
            // attention mask [u_len, u_len] false (do not mask)
            for q in 0..u_len {
                for kv in 0..u_len {
                    let dst_idx = ((bsz + i) * max_c_len + q) * max_c_len + kv;
                    batch_attention_mask_cpu[dst_idx] = false;
                }
            }
            // pad diag false for uncond padding range (do not mask diagonal of padding range to prevent NaNs)
            if max_c_len > u_len {
                for d in u_len..max_c_len {
                    let dst_idx = ((bsz + i) * max_c_len + d) * max_c_len + d;
                    batch_attention_mask_cpu[dst_idx] = false;
                }
            }
        }

        // Initialize final generated tokens tensor [B, C, max_target_len] to mask_id
        let max_target_len = *target_lens.iter().max().unwrap_or(&0);
        let mut tokens_cpu = vec![self.config.audio_mask_id as i64; bsz * c * max_target_len];

        // Prepare schedules
        let timesteps = get_time_steps(gen_config.num_step, gen_config.t_shift);
        let mut schedules = Vec::with_capacity(bsz);
        for &t_len in &target_lens {
            let total_mask = t_len * c;
            let mut rem = total_mask;
            let mut sched = Vec::with_capacity(gen_config.num_step);
            for step in 0..gen_config.num_step {
                let num = if step == gen_config.num_step - 1 {
                    rem
                } else {
                    let step_ratio = timesteps[step + 1] - timesteps[step];
                    let val = ((total_mask as f32) * step_ratio).ceil() as usize;
                    val.min(rem)
                };
                sched.push(num);
                rem -= num;
            }
            schedules.push(sched);
        }

        let mut rng = Xorshift::new(1337);

        // Upload initial batch tensors to GPU
        let mut batch_input_ids = Tensor::<B, 3, Int>::from_data(
            TensorData::new(batch_input_ids_cpu.clone(), [2 * bsz, c, max_c_len]),
            &device,
        );
        let first_codebook_ids = batch_input_ids.clone().slice([0..1, 0..1, 0..40.min(max_c_len)]).into_data().iter::<i64>().collect::<Vec<_>>();
        log::info!("Batch input ids (first 40): {:?}", first_codebook_ids);
        let batch_audio_mask = Tensor::<B, 2, Bool>::from_data(
            TensorData::new(batch_audio_mask_cpu, [2 * bsz, max_c_len]),
            &device,
        );
        let batch_attention_mask = Tensor::<B, 4, Bool>::from_data(
            TensorData::new(batch_attention_mask_cpu, [2 * bsz, 1, max_c_len, max_c_len]),
            &device,
        );

        // Iterative unmasking loop
        for step in 0..gen_config.num_step {
            if let Some(token) = cancel {
                if token.is_cancelled() {
                    return Err(crate::Error::Cancelled);
                }
            }
            let batch_logits = self.forward(
                batch_input_ids.clone(),
                batch_audio_mask.clone(),
                Some(batch_attention_mask.clone()),
            );

            // CPU updates
            let mut tokens_updated = false;

            for i in 0..bsz {
                let k = schedules[i][step];
                if k == 0 {
                    continue;
                }

                let c_len = c_lens[i];
                let t_len = target_lens[i];

                // Extract logits on GPU
                let c_logits = batch_logits.clone().slice([i..i+1, 0..c, (c_len - t_len)..c_len, 0..v]);
                let u_logits = batch_logits.clone().slice([(bsz + i)..(bsz + i + 1), 0..c, 0..t_len, 0..v]);

                // Guidance and Softmax on GPU
                let log_probs = if gen_config.guidance_scale != 0.0 {
                    let c_log = stable_log_softmax(c_logits, 3);
                    let u_log = stable_log_softmax(u_logits, 3);
                    stable_log_softmax(c_log.clone() + (c_log - u_log).mul_scalar(gen_config.guidance_scale), 3)
                } else {
                    stable_log_softmax(c_logits, 3)
                };

                // Exclude mask token from predictions by slicing vocab to 0..1024
                let log_probs_sliced = log_probs.slice([0..1, 0..c, 0..t_len, 0..1024]);

                let pred_tokens: Tensor<B, 3, Int> = log_probs_sliced.clone().argmax(3).squeeze_dim::<3>(3); // [1, C, T]
                let confidence_scores: Tensor<B, 3> = log_probs_sliced.max_dim(3).squeeze_dim::<3>(3); // [1, C, T]

                // Download predictions to CPU
                let pred_tokens_data = pred_tokens.into_data().iter::<i64>().collect::<Vec<_>>();
                let scores_data = confidence_scores.into_data().iter::<f32>().collect::<Vec<_>>();

                if step == 0 && i == 0 {
                    log::info!("Step 0 pred_tokens (first 10): {:?}", &pred_tokens_data[..10.min(pred_tokens_data.len())]);
                    log::info!("Step 0 scores_data (first 10): {:?}", &scores_data[..10.min(scores_data.len())]);
                }

                // Calculate scores for all target positions
                let mut candidate_scores = Vec::with_capacity(c * t_len);
                for codebook in 0..c {
                    for t_idx in 0..t_len {
                        let flat_idx = codebook * t_len + t_idx;
                        let token_flat_idx = (i * c + codebook) * max_target_len + t_idx;
                        
                        // Masked fill already unmasked positions with -inf
                        if tokens_cpu[token_flat_idx] != self.config.audio_mask_id as i64 {
                            candidate_scores.push((flat_idx, -f32::INFINITY));
                        } else {
                            // Apply layer penalty
                            let base_score = scores_data[flat_idx] - (codebook as f32 * gen_config.layer_penalty_factor);
                            
                            // Apply position temperature Gumbel sampling
                            let final_score = if gen_config.position_temperature > 0.0 {
                                let u_val = rng.next_f32();
                                let gumbel = -(- (u_val + 1e-10).ln() + 1e-10).ln();
                                (base_score / gen_config.position_temperature) + gumbel
                            } else {
                                base_score
                            };
                            candidate_scores.push((flat_idx, final_score));
                        }
                    }
                }

                if step == 0 && i == 0 {
                    log::info!("Step 0 candidate_scores count: {}", candidate_scores.len());
                }

                // Sort candidates descending by score
                candidate_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

                if step == 0 && i == 0 {
                    log::info!("Step 0 sorted candidates (first 10): {:?}", &candidate_scores[..10.min(candidate_scores.len())]);
                }

                // Unmask top-k
                for rank in 0..k {
                    let (flat_idx, score) = candidate_scores[rank];
                    if score == -f32::INFINITY {
                        break;
                    }
                    let codebook = flat_idx / t_len;
                    let t_idx = flat_idx % t_len;
                    let token_flat_idx = (i * c + codebook) * max_target_len + t_idx;
                    
                    tokens_cpu[token_flat_idx] = pred_tokens_data[flat_idx];
                }

                tokens_updated = true;
            }

            if tokens_updated {
                // Construct the updated token slices on GPU and update batch_input_ids
                for i in 0..bsz {
                    let c_len = c_lens[i];
                    let t_len = target_lens[i];

                    let mut item_tokens = vec![pad_id; c * t_len];
                    for codebook in 0..c {
                        for t_idx in 0..t_len {
                            let src_idx = (i * c + codebook) * max_target_len + t_idx;
                            item_tokens[codebook * t_len + t_idx] = tokens_cpu[src_idx];
                        }
                    }

                    let item_tokens_tensor = Tensor::<B, 3, Int>::from_data(
                        TensorData::new(item_tokens, [1, c, t_len]),
                        &device,
                    );

                    // Cond row slice assign
                    batch_input_ids = batch_input_ids.slice_assign(
                        [i..i+1, 0..c, (c_len - t_len)..c_len],
                        item_tokens_tensor.clone(),
                    );
                    // Uncond row slice assign
                    batch_input_ids = batch_input_ids.slice_assign(
                        [(bsz + i)..(bsz + i + 1), 0..c, 0..t_len],
                        item_tokens_tensor,
                    );
                }
            }
        }

        let mut mask_count = 0;
        for &t in &tokens_cpu {
            if t == self.config.audio_mask_id as i64 {
                mask_count += 1;
            }
        }
        log::info!("Iterative unmasking finished. Remaining mask tokens: {} out of {}", mask_count, tokens_cpu.len());

        // Return generated tokens per batch item [C, T]
        let mut results = Vec::with_capacity(bsz);
        for i in 0..bsz {
            let t_len = target_lens[i];
            let mut final_tokens = Vec::with_capacity(c * t_len);
            for codebook in 0..c {
                for t_idx in 0..t_len {
                    let src_idx = (i * c + codebook) * max_target_len + t_idx;
                    final_tokens.push(tokens_cpu[src_idx]);
                }
            }
            results.push(Tensor::<B, 3, Int>::from_data(
                TensorData::new(final_tokens, [1, c, t_len]),
                &device,
            ));
        }

        Ok(results)
    }

    /// Reference python: _prepare_inference_inputs
    pub fn prepare_inference_inputs(
        &self,
        text: &str,
        num_target_tokens: usize,
        ref_text: Option<&str>,
        ref_audio_tokens: Option<&Tensor<B, 3, Int>>,
        lang: Option<&str>,
        instruct: Option<&str>,
        denoise: bool,
        text_tokenizer: &TextTokenizer,
        device: &B::Device,
    ) -> Result<InferenceInput<B>> {
        // Build style tokens
        let mut style_text = String::new();
        if denoise && ref_audio_tokens.is_some() {
            style_text.push_str("<|denoise|>");
        }
        let lang_str = lang.unwrap_or("None");
        let instruct_str = instruct.unwrap_or("None");
        style_text.push_str(&format!("<|lang_start|>{lang_str}<|lang_end|>"));
        style_text.push_str(&format!("<|instruct_start|>{instruct_str}<|instruct_end|>"));

        let style_ids = text_tokenizer.encode(&style_text)?;
        let style_len = style_ids.len();
        let c = self.config.num_audio_codebook;

        // Replicate across C codebooks
        let mut style_ids_flat = Vec::with_capacity(c * style_len);
        for _ in 0..c {
            style_ids_flat.extend_from_slice(&style_ids);
        }
        let style_tokens = Tensor::<B, 3, Int>::from_data(
            TensorData::new(style_ids_flat, [1, c, style_len]),
            device,
        );

        // Build text tokens
        let full_text = combine_text(text, ref_text);
        let wrapped_text = format!("<|text_start|>{full_text}<|text_end|>");
        let text_ids = tokenize_with_nonverbal_tags(&wrapped_text, text_tokenizer)?;
        let text_len = text_ids.len();

        let mut text_ids_flat = Vec::with_capacity(c * text_len);
        for _ in 0..c {
            text_ids_flat.extend_from_slice(&text_ids);
        }
        let text_tokens = Tensor::<B, 3, Int>::from_data(
            TensorData::new(text_ids_flat, [1, c, text_len]),
            device,
        );

        // Target: all MASK
        let mask_id = self.config.audio_mask_id as i64;
        let target_audio_tokens = Tensor::<B, 3, Int>::from_data(
            TensorData::new(
                vec![mask_id; c * num_target_tokens],
                [1, c, num_target_tokens],
            ),
            device,
        );

        // Cat parts
        let mut parts = vec![style_tokens, text_tokens];
        if let Some(ref_tokens) = ref_audio_tokens {
            parts.push(ref_tokens.clone());
        }
        parts.push(target_audio_tokens);
        let cond_input_ids = Tensor::cat(parts, 2); // [1, C, cond_total_length]

        let cond_total_length = cond_input_ids.dims()[2];
        let mut cond_audio_start_idx = cond_total_length - num_target_tokens;
        if let Some(ref_tokens) = ref_audio_tokens {
            cond_audio_start_idx -= ref_tokens.dims()[2];
        }

        // Prepare cond_audio_mask [1, cond_total_length]
        let mut mask_data = vec![false; cond_total_length];
        for idx in cond_audio_start_idx..cond_total_length {
            mask_data[idx] = true;
        }
        let cond_audio_mask = Tensor::<B, 2, Bool>::from_data(
            TensorData::new(mask_data, [1, cond_total_length]),
            device,
        );

        Ok(InferenceInput {
            input_ids: cond_input_ids,
            audio_mask: cond_audio_mask,
        })
    }
}

#[derive(Debug)]
pub struct InferenceInput<B: Backend> {
    pub input_ids: Tensor<B, 3, Int>,
    pub audio_mask: Tensor<B, 2, Bool>,
}

const NONVERBAL_TAGS: &[&str] = &[
    "[laughter]",
    "[sigh]",
    "[confirmation-en]",
    "[question-en]",
    "[question-ah]",
    "[question-oh]",
    "[question-ei]",
    "[question-yi]",
    "[surprise-ah]",
    "[surprise-oh]",
    "[surprise-wa]",
    "[surprise-yo]",
    "[dissatisfaction-hnn]",
];

pub fn tokenize_with_nonverbal_tags(text: &str, tokenizer: &TextTokenizer) -> Result<Vec<i64>> {
    let mut ids = Vec::new();
    let mut current_idx = 0;
    
    while let Some(start_idx) = text[current_idx..].find('[') {
        let absolute_start = current_idx + start_idx;
        if let Some(end_idx) = text[absolute_start..].find(']') {
            let absolute_end = absolute_start + end_idx;
            let tag = &text[absolute_start..=absolute_end];
            if NONVERBAL_TAGS.contains(&tag) {
                // Tokenize text before tag
                if absolute_start > current_idx {
                    let segment = &text[current_idx..absolute_start];
                    ids.extend(tokenizer.encode(segment)?);
                }
                // Tokenize tag
                ids.extend(tokenizer.encode(tag)?);
                current_idx = absolute_end + 1;
                continue;
            }
        }
        // If not a valid tag, we skip the '[' and continue
        let segment = &text[current_idx..=absolute_start];
        ids.extend(tokenizer.encode(segment)?);
        current_idx = absolute_start + 1;
    }
    
    if current_idx < text.len() {
        let segment = &text[current_idx..];
        ids.extend(tokenizer.encode(segment)?);
    }
    
    Ok(ids)
}

fn combine_text(text: &str, ref_text: Option<&str>) -> String {
    let mut full_text = match ref_text {
        Some(ref_t) => format!("{} {}", ref_t.trim(), text.trim()),
        None => text.trim().to_string(),
    };

    // Filter out newline / carriage-return characters
    full_text = full_text.replace('\r', "").replace('\n', "");

    // Replace Chinese parentheses with English ones
    full_text = full_text.replace('\u{ff08}', "(").replace('\u{ff09}', ")");

    // Collapse consecutive spaces / tabs into a single space
    let mut collapsed = String::with_capacity(full_text.len());
    let mut in_space = false;
    for c in full_text.chars() {
        if c == ' ' || c == '\t' {
            if !in_space {
                collapsed.push(' ');
                in_space = true;
            }
        } else {
            collapsed.push(c);
            in_space = false;
        }
    }
    full_text = collapsed;

    // Remove spaces around Chinese characters (U+4E00..=U+9FFF)
    let is_cjk = |c: char| -> bool { ('\u{4e00}'..='\u{9fff}').contains(&c) };
    let mut chars: Vec<char> = full_text.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == ' ' {
            let prev_is_cjk = i > 0 && is_cjk(chars[i - 1]);
            let next_is_cjk = i + 1 < chars.len() && is_cjk(chars[i + 1]);
            if prev_is_cjk || next_is_cjk {
                chars.remove(i);
                continue;
            }
        }
        i += 1;
    }
    
    chars.into_iter().collect()
}

pub fn estimate_target_tokens(
    estimator: &RuleDurationEstimator,
    text: &str,
    ref_text: Option<&str>,
    num_ref_tokens: Option<usize>,
    speed: f32,
) -> usize {
    let (ref_t, ref_tokens) = match (ref_text, num_ref_tokens) {
        (Some(rt), Some(num)) if !rt.is_empty() && num > 0 => (rt, num as f32),
        _ => ("Nice to meet you.", 25.0),
    };

    let mut est = estimator.estimate_duration(
        text,
        ref_t,
        ref_tokens,
        Some(50.0),
        3.0,
    );

    if speed > 0.0 && speed != 1.0 {
        est /= speed;
    }

    (est as usize).max(1)
}
