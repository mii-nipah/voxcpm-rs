use burn::module::Param;
use burn::nn::conv::{Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig};
use burn::nn::PaddingConfig1d;
use burn::prelude::*;

// ---------------------------------------------------------------------------
// Snake1d Activation: x + (1/(alpha+1e-9)) * sin(alpha*x)^2
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct Snake1d<B: Backend> {
    pub alpha: Param<Tensor<B, 3>>, // Shape: [1, C, 1]
}

impl<B: Backend> Snake1d<B> {
    pub fn new(channels: usize, device: &B::Device) -> Self {
        Self {
            alpha: Param::from_tensor(Tensor::ones([1, channels, 1], device)),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let alpha = self.alpha.val();
        let denom = alpha.clone().add_scalar(1e-9).recip();
        let sin_sq = (alpha * x.clone()).sin().powf_scalar(2.0);
        x + denom * sin_sq
    }
}

// ---------------------------------------------------------------------------
// DAC Residual Unit: Snake -> Conv1d(k=7, dil) -> Snake -> Conv1d(k=1)
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct DacResidualUnit<B: Backend> {
    pub snake1: Snake1d<B>,
    pub conv1: Conv1d<B>,
    pub snake2: Snake1d<B>,
    pub conv2: Conv1d<B>,
}

impl<B: Backend> DacResidualUnit<B> {
    pub fn new(dim: usize, dilation: usize, device: &B::Device) -> Self {
        let pad = (6 * dilation) / 2; // kernel_size = 7
        let conv1 = Conv1dConfig::new(dim, dim, 7)
            .with_dilation(dilation)
            .with_padding(PaddingConfig1d::Explicit(pad))
            .init(device);
        let conv2 = Conv1dConfig::new(dim, dim, 1)
            .init(device);

        Self {
            snake1: Snake1d::new(dim, device),
            conv1,
            snake2: Snake1d::new(dim, device),
            conv2,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let y = self.snake1.forward(x.clone());
        let y = self.conv1.forward(y);
        let y = self.snake2.forward(y);
        let y = self.conv2.forward(y);

        let x_len = x.dims()[2];
        let y_len = y.dims()[2];
        let x_sliced = if x_len > y_len {
            let pad = (x_len - y_len) / 2;
            x.narrow(2, pad, y_len)
        } else {
            x
        };

        x_sliced + y
    }
}

// ---------------------------------------------------------------------------
// DAC Encoder Block: downsampling block
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct DacEncoderBlock<B: Backend> {
    pub res_unit1: DacResidualUnit<B>,
    pub res_unit2: DacResidualUnit<B>,
    pub res_unit3: DacResidualUnit<B>,
    pub snake1: Snake1d<B>,
    pub conv1: Conv1d<B>,
}

impl<B: Backend> DacEncoderBlock<B> {
    pub fn new(encoder_hidden_size: usize, stride: usize, stride_index: usize, device: &B::Device) -> Self {
        let dim = encoder_hidden_size * 2usize.pow(stride_index as u32);
        let pad = (stride as f64 / 2.0).ceil() as usize;

        Self {
            res_unit1: DacResidualUnit::new(dim / 2, 1, device),
            res_unit2: DacResidualUnit::new(dim / 2, 3, device),
            res_unit3: DacResidualUnit::new(dim / 2, 9, device),
            snake1: Snake1d::new(dim / 2, device),
            conv1: Conv1dConfig::new(dim / 2, dim, 2 * stride)
                .with_stride(stride)
                .with_padding(PaddingConfig1d::Explicit(pad))
                .init(device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let y = self.res_unit1.forward(x);
        let y = self.res_unit2.forward(y);
        let y = self.res_unit3.forward(y);
        let y = self.snake1.forward(y);
        self.conv1.forward(y)
    }
}

// ---------------------------------------------------------------------------
// DAC Decoder Block: upsampling block
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct DacDecoderBlock<B: Backend> {
    pub snake1: Snake1d<B>,
    pub conv_t1: ConvTranspose1d<B>,
    pub res_unit1: DacResidualUnit<B>,
    pub res_unit2: DacResidualUnit<B>,
    pub res_unit3: DacResidualUnit<B>,
}

impl<B: Backend> DacDecoderBlock<B> {
    pub fn new(decoder_hidden_size: usize, stride: usize, stride_index: usize, device: &B::Device) -> Self {
        let input_dim = decoder_hidden_size / 2usize.pow(stride_index as u32);
        let output_dim = decoder_hidden_size / 2usize.pow((stride_index + 1) as u32);
        let pad = (stride as f64 / 2.0).ceil() as usize;
        let out_pad = stride % 2;

        let conv_t1 = ConvTranspose1dConfig::new([input_dim, output_dim], 2 * stride)
            .with_stride(stride)
            .with_padding(pad)
            .with_padding_out(out_pad)
            .init(device);

        Self {
            snake1: Snake1d::new(input_dim, device),
            conv_t1,
            res_unit1: DacResidualUnit::new(output_dim, 1, device),
            res_unit2: DacResidualUnit::new(output_dim, 3, device),
            res_unit3: DacResidualUnit::new(output_dim, 9, device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let y = self.snake1.forward(x);
        let y = self.conv_t1.forward(y);
        let y = self.res_unit1.forward(y);
        let y = self.res_unit2.forward(y);
        self.res_unit3.forward(y)
    }
}

// ---------------------------------------------------------------------------
// DAC Encoder
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct DacEncoder<B: Backend> {
    pub conv1: Conv1d<B>,
    pub block: Vec<DacEncoderBlock<B>>,
    pub snake1: Snake1d<B>,
    pub conv2: Conv1d<B>,
}

impl<B: Backend> DacEncoder<B> {
    pub fn new(
        encoder_hidden_size: usize,
        hidden_size: usize,
        downsampling_ratios: &[usize],
        device: &B::Device,
    ) -> Self {
        let conv1 = Conv1dConfig::new(1, encoder_hidden_size, 7)
            .with_padding(PaddingConfig1d::Explicit(3))
            .init(device);

        let mut block = Vec::with_capacity(downsampling_ratios.len());
        for (i, &stride) in downsampling_ratios.iter().enumerate() {
            block.push(DacEncoderBlock::new(encoder_hidden_size, stride, i + 1, device));
        }

        let last_dim = encoder_hidden_size * 2usize.pow(downsampling_ratios.len() as u32);
        let snake1 = Snake1d::new(last_dim, device);
        let conv2 = Conv1dConfig::new(last_dim, hidden_size, 3)
            .with_padding(PaddingConfig1d::Explicit(1))
            .init(device);

        Self {
            conv1,
            block,
            snake1,
            conv2,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let mut y = self.conv1.forward(x);
        for b in &self.block {
            y = b.forward(y);
        }
        y = self.snake1.forward(y);
        self.conv2.forward(y)
    }
}

// ---------------------------------------------------------------------------
// DAC Decoder
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct DacDecoder<B: Backend> {
    pub conv1: Conv1d<B>,
    pub block: Vec<DacDecoderBlock<B>>,
    pub snake1: Snake1d<B>,
    pub conv2: Conv1d<B>,
}

impl<B: Backend> DacDecoder<B> {
    pub fn new(
        decoder_hidden_size: usize,
        hidden_size: usize,
        upsampling_ratios: &[usize],
        device: &B::Device,
    ) -> Self {
        let conv1 = Conv1dConfig::new(hidden_size, decoder_hidden_size, 7)
            .with_padding(PaddingConfig1d::Explicit(3))
            .init(device);

        let mut block = Vec::with_capacity(upsampling_ratios.len());
        for (i, &stride) in upsampling_ratios.iter().enumerate() {
            block.push(DacDecoderBlock::new(decoder_hidden_size, stride, i, device));
        }

        let last_dim = decoder_hidden_size / 2usize.pow(upsampling_ratios.len() as u32);
        let snake1 = Snake1d::new(last_dim, device);
        let conv2 = Conv1dConfig::new(last_dim, 1, 7)
            .with_padding(PaddingConfig1d::Explicit(3))
            .init(device);

        Self {
            conv1,
            block,
            snake1,
            conv2,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let mut y = self.conv1.forward(x);
        for b in &self.block {
            y = b.forward(y);
        }
        y = self.snake1.forward(y);
        self.conv2.forward(y)
    }
}
