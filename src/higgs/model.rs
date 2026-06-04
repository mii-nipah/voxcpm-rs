use crate::higgs::config::HiggsConfig;
use crate::higgs::dac::{DacDecoder, DacEncoder};
use crate::higgs::hubert::HubertModel;
use crate::higgs::quantizer::ResidualVectorQuantizer;
use burn::nn::{Linear, LinearConfig};
use burn::nn::conv::{Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig};
use burn::nn::PaddingConfig1d;
use burn::prelude::*;
use burn::tensor::Int;

fn elu<const D: usize, B: Backend>(x: Tensor<B, D>, alpha: f32) -> Tensor<B, D> {
    let mask = x.clone().lower_elem(0.0);
    let exp_part = (x.clone().exp() - 1.0).mul_scalar(alpha);
    x.mask_where(mask, exp_part)
}

// ---------------------------------------------------------------------------
// Semantic Residual Unit (uses ELU, not Snake)
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct SemanticResidualUnit<B: Backend> {
    pub conv1: Conv1d<B>,
    pub conv2: Conv1d<B>,
}

impl<B: Backend> SemanticResidualUnit<B> {
    pub fn new(in_channels: usize, out_channels: usize, dilation: usize, unit_kernel_size: usize, device: &B::Device) -> Self {
        let padding = ((unit_kernel_size - 1) / 2) * dilation;
        let conv1 = Conv1dConfig::new(in_channels, out_channels, unit_kernel_size)
            .with_dilation(dilation)
            .with_padding(PaddingConfig1d::Explicit(padding))
            .with_bias(false)
            .init(device);
        let conv2 = Conv1dConfig::new(out_channels, out_channels, 1)
            .with_bias(false)
            .init(device);
        Self { conv1, conv2 }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let y = elu(x.clone(), 1.0);
        let y = self.conv1.forward(y);
        let y = elu(y, 1.0);
        let y = self.conv2.forward(y);
        x + y
    }
}

// ---------------------------------------------------------------------------
// Semantic Encoder Block
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct SemanticEncoderBlock<B: Backend> {
    pub res_units: Vec<SemanticResidualUnit<B>>,
    pub conv: Conv1d<B>,
}

impl<B: Backend> SemanticEncoderBlock<B> {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        stride: usize,
        unit_kernel_size: usize,
        block_dilations: &[usize],
        device: &B::Device,
    ) -> Self {
        let res_units = block_dilations.iter()
            .map(|&dil| SemanticResidualUnit::new(in_channels, in_channels, dil, unit_kernel_size, device))
            .collect();
        let kernel = if stride == 1 { 3 } else { 2 * stride };
        let padding = (kernel - 1) / 2;
        let conv = Conv1dConfig::new(in_channels, out_channels, kernel)
            .with_stride(stride)
            .with_padding(PaddingConfig1d::Explicit(padding))
            .init(device);
        Self { res_units, conv }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let mut y = x;
        for unit in &self.res_units {
            y = unit.forward(y);
        }
        self.conv.forward(y)
    }
}

// ---------------------------------------------------------------------------
// Semantic Decoder Block
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub enum DecoderConv<B: Backend> {
    Conv(Conv1d<B>),
    ConvTranspose(ConvTranspose1d<B>),
}

impl<B: Backend> DecoderConv<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        match self {
            Self::Conv(c) => c.forward(x),
            Self::ConvTranspose(ct) => ct.forward(x),
        }
    }
}

#[derive(Module, Debug)]
pub struct SemanticDecoderBlock<B: Backend> {
    pub conv: DecoderConv<B>,
    pub res_units: Vec<SemanticResidualUnit<B>>,
}

impl<B: Backend> SemanticDecoderBlock<B> {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        stride: usize,
        unit_kernel_size: usize,
        block_dilations: &[usize],
        device: &B::Device,
    ) -> Self {
        let conv = if stride == 1 {
            DecoderConv::Conv(
                Conv1dConfig::new(in_channels, out_channels, 3)
                    .with_padding(PaddingConfig1d::Explicit(1))
                    .init(device)
            )
        } else {
            let kernel = 2 * stride;
            let padding = (stride + 1) / 2;
            let out_pad = stride % 2;
            DecoderConv::ConvTranspose(
                ConvTranspose1dConfig::new([in_channels, out_channels], kernel)
                    .with_stride(stride)
                    .with_padding(padding)
                    .with_padding_out(out_pad)
                    .with_bias(false)
                    .init(device)
            )
        };

        let res_units = block_dilations.iter()
            .map(|&dil| SemanticResidualUnit::new(out_channels, out_channels, dil, unit_kernel_size, device))
            .collect();

        Self { conv, res_units }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let mut y = self.conv.forward(x);
        for unit in &self.res_units {
            y = unit.forward(y);
        }
        y
    }
}

// ---------------------------------------------------------------------------
// Semantic Encoder
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct SemanticEncoder<B: Backend> {
    pub conv: Conv1d<B>,
    pub conv_blocks: Vec<SemanticEncoderBlock<B>>,
}

impl<B: Backend> SemanticEncoder<B> {
    pub fn new(
        semantic_hidden_size: usize,
        kernel_size: usize,
        strides: &[usize],
        channel_ratios: &[f32],
        unit_kernel_size: usize,
        block_dilations: &[usize],
        device: &B::Device,
    ) -> Self {
        let conv = Conv1dConfig::new(semantic_hidden_size, semantic_hidden_size, kernel_size)
            .with_padding(PaddingConfig1d::Explicit(kernel_size / 2))
            .with_bias(false)
            .init(device);

        let mut conv_blocks = Vec::with_capacity(strides.len());
        let mut in_ch = semantic_hidden_size;
        for i in 0..strides.len() {
            let out_ch = (semantic_hidden_size as f32 * channel_ratios[i]) as usize;
            conv_blocks.push(SemanticEncoderBlock::new(in_ch, out_ch, strides[i], unit_kernel_size, block_dilations, device));
            in_ch = out_ch;
        }

        Self { conv, conv_blocks }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let mut y = self.conv.forward(x);
        for block in &self.conv_blocks {
            y = block.forward(y);
        }
        y
    }
}

// ---------------------------------------------------------------------------
// Semantic Decoder
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct SemanticDecoder<B: Backend> {
    pub conv1: Conv1d<B>,
    pub conv_blocks: Vec<SemanticDecoderBlock<B>>,
    pub conv2: Conv1d<B>,
}

impl<B: Backend> SemanticDecoder<B> {
    pub fn new(
        semantic_hidden_size: usize,
        kernel_size: usize,
        strides: &[usize],
        channel_ratios: &[f32],
        unit_kernel_size: usize,
        block_dilations: &[usize],
        device: &B::Device,
    ) -> Self {
        let out_ch0 = (semantic_hidden_size as f32 * channel_ratios[0]) as usize;
        let conv1 = Conv1dConfig::new(semantic_hidden_size, out_ch0, kernel_size)
            .with_padding(PaddingConfig1d::Explicit(kernel_size / 2))
            .with_bias(false)
            .init(device);

        let mut conv_blocks = Vec::with_capacity(strides.len());
        for i in 0..strides.len() {
            let in_ch = (semantic_hidden_size as f32 * channel_ratios[i]) as usize;
            let out_ch = if i < strides.len() - 1 {
                (semantic_hidden_size as f32 * channel_ratios[i + 1]) as usize
            } else {
                semantic_hidden_size
            };
            conv_blocks.push(SemanticDecoderBlock::new(in_ch, out_ch, strides[i], unit_kernel_size, block_dilations, device));
        }

        let conv2 = Conv1dConfig::new(semantic_hidden_size, semantic_hidden_size, kernel_size)
            .with_padding(PaddingConfig1d::Explicit(kernel_size / 2))
            .with_bias(false)
            .init(device);

        Self { conv1, conv_blocks, conv2 }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let mut y = self.conv1.forward(x);
        for block in &self.conv_blocks {
            y = block.forward(y);
        }
        self.conv2.forward(y)
    }
}

// ---------------------------------------------------------------------------
// HiggsTokenizer Model
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct HiggsTokenizer<B: Backend> {
    pub acoustic_encoder: DacEncoder<B>,
    pub acoustic_decoder: DacDecoder<B>,
    pub encoder_semantic: SemanticEncoder<B>,
    pub decoder_semantic: SemanticDecoder<B>,
    pub semantic_model: HubertModel<B>,
    pub fc: Linear<B>,
    pub fc1: Linear<B>,
    pub fc2: Linear<B>,
    pub quantizer: ResidualVectorQuantizer<B>,
    pub pad: usize,
    pub sample_rate: usize,
    pub semantic_sample_rate: usize,
    pub semantic_downsample_factor: usize,
    pub hop_length: usize,
}

impl<B: Backend> HiggsTokenizer<B> {
    pub fn new(config: HiggsConfig, device: &B::Device) -> Self {
        let acoustic_encoder = DacEncoder::new(
            config.acoustic_model_config.encoder_hidden_size,
            config.acoustic_model_config.hidden_size,
            &config.acoustic_model_config.downsampling_ratios,
            device,
        );

        let acoustic_decoder = DacDecoder::new(
            config.acoustic_model_config.decoder_hidden_size,
            config.acoustic_model_config.hidden_size,
            &config.acoustic_model_config.upsampling_ratios,
            device,
        );

        let encoder_semantic = SemanticEncoder::new(
            config.semantic_model_config.hidden_size,
            config.kernel_size,
            &config.strides,
            &config.channel_ratios,
            config.unit_kernel_size,
            &config.block_dilations,
            device,
        );

        let decoder_semantic = SemanticDecoder::new(
            config.semantic_model_config.hidden_size,
            config.kernel_size,
            &config.strides,
            &config.channel_ratios,
            config.unit_kernel_size,
            &config.block_dilations,
            device,
        );

        let semantic_model = HubertModel::new(
            config.semantic_model_config.hidden_size,
            config.semantic_model_config.num_attention_heads,
            config.semantic_model_config.intermediate_size,
            config.semantic_model_config.num_hidden_layers,
            device,
        );

        let joint_hidden_size = config.semantic_model_config.hidden_size + config.acoustic_model_config.hidden_size;
        let fc = LinearConfig::new(joint_hidden_size, joint_hidden_size).init(device);
        let fc1 = LinearConfig::new(joint_hidden_size, config.semantic_model_config.hidden_size).init(device);
        let fc2 = LinearConfig::new(joint_hidden_size, config.acoustic_model_config.hidden_size).init(device);

        let hop_length = config.acoustic_model_config.downsampling_ratios.iter().product::<usize>();
        let quantizer = ResidualVectorQuantizer::new(
            joint_hidden_size,
            config.codebook_size,
            config.codebook_dim,
            8, // num_quantizers is 8 for Higgs Audio V2
            device,
        );

        // The semantic_downsample_factor is the stride used to subsample
        // HuBERT hidden states (take every Nth frame). This is 2 for HiGGS
        // Audio V2. Note: config.downsample_factor (320) is the HuBERT feature
        // extractor's effective stride, NOT the hidden-state subsampling factor.
        let semantic_downsample_factor = 2;

        Self {
            acoustic_encoder,
            acoustic_decoder,
            encoder_semantic,
            decoder_semantic,
            semantic_model,
            fc,
            fc1,
            fc2,
            quantizer,
            pad: hop_length / 2,
            sample_rate: config.sample_rate,
            semantic_sample_rate: config.semantic_sample_rate,
            semantic_downsample_factor,
            hop_length,
        }
    }

    /// Encode a raw audio waveform into discrete audio codes.
    ///
    /// `input_values`: `[B, 1, T]` at `self.sample_rate` (24 kHz).
    /// The method internally resamples to `self.semantic_sample_rate` (16 kHz)
    /// for HuBERT feature extraction, matching the Python reference.
    pub fn encode(
        &self,
        input_values: Tensor<B, 3>,
        bandwidth: Option<f32>,
    ) -> Tensor<B, 3, Int> {
        // 1) Resample to semantic_sample_rate for HuBERT if rates differ
        let semantic_input = if self.sample_rate != self.semantic_sample_rate {
            // Download to CPU, resample, re-upload
            let data = input_values.clone().into_data().iter::<f32>().collect::<Vec<_>>();
            let resampled = crate::audio::resample(
                &data,
                self.sample_rate as u32,
                self.semantic_sample_rate as u32,
            ).expect("resample for HuBERT");
            let len = resampled.len();
            Tensor::<B, 3>::from_data(
                burn::tensor::TensorData::new(resampled, [1, 1, len]),
                &input_values.device(),
            )
        } else {
            input_values.clone()
        };

        // 2) Extract semantic features from HuBERT
        // Python: input_values = input_values[:, 0, :] then pads (160, 160)
        // Our HuBERT expects [B, 1, T] so we keep the channel dim.
        let padded_semantic = semantic_input.pad((160, 160, 0, 0), 0.0);
        let all_hidden = self.semantic_model.forward(padded_semantic);
        let unsqueezed: Vec<Tensor<B, 4>> = all_hidden.into_iter().map(|t| t.unsqueeze_dim(1)).collect();
        let stacked = Tensor::cat(unsqueezed, 1);
        let mean = stacked.mean_dim(1).squeeze_dim(1); // [B, L, C]

        // 3) Downsample by semantic_downsample_factor using stride indexing
        //    Python: semantic_features[:, ::factor, :]
        let e_semantic_input = if self.semantic_downsample_factor > 1 {
            let [_b, l, _c] = mean.dims();
            let out_l = l / self.semantic_downsample_factor;
            // Gather every Nth frame by building index list
            let indices: Vec<i64> = (0..out_l)
                .map(|i| (i * self.semantic_downsample_factor) as i64)
                .collect();
            let idx_tensor = Tensor::<B, 1, Int>::from_data(
                burn::tensor::TensorData::new(indices, [out_l]),
                &mean.device(),
            );
            mean.select(1, idx_tensor)
        } else {
            mean
        };

        // 4) Semantic Encoder
        let e_semantic_input_t = e_semantic_input.swap_dims(1, 2); // [B, C, L/factor]
        let e_semantic = self.encoder_semantic.forward(e_semantic_input_t); // [B, C_s, L']

        // 5) Acoustic Encoder
        let out_len = input_values.dims()[2] / self.hop_length;
        let e_acoustic = if out_len != e_semantic.dims()[2] {
            let padded_acoustic = input_values.pad((self.pad, self.pad, 0, 0), 0.0);
            self.acoustic_encoder.forward(padded_acoustic)
        } else {
            self.acoustic_encoder.forward(input_values)
        };

        // 6) Cat embeddings
        let embeddings = Tensor::cat(vec![e_acoustic, e_semantic], 1); // [B, H, L']
        let embeddings_t = embeddings.swap_dims(1, 2); // [B, L', H]
        let projected = self.fc.forward(embeddings_t).swap_dims(1, 2); // [B, H, L']

        // 7) RVQ encode
        let frame_rate = self.sample_rate / self.hop_length;
        self.quantizer.encode(projected, bandwidth, frame_rate)
    }

    pub fn decode(&self, audio_codes: Tensor<B, 3, Int>) -> Tensor<B, 3> {
        let quantized = self.quantizer.decode(audio_codes); // [B, H, T]
        let quantized_t = quantized.swap_dims(1, 2); // [B, T, H]
        let quantized_acoustic = self.fc2.forward(quantized_t).swap_dims(1, 2); // [B, H_a, T]
        self.acoustic_decoder.forward(quantized_acoustic)
    }
}
