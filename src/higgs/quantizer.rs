use burn::module::Param;
use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::Int;

// ---------------------------------------------------------------------------
// Euclidean Codebook
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct EuclideanCodebook<B: Backend> {
    pub embed: Param<Tensor<B, 2>>, // Shape: [codebook_size, codebook_dim]
}

impl<B: Backend> EuclideanCodebook<B> {
    pub fn new(codebook_size: usize, codebook_dim: usize, device: &B::Device) -> Self {
        Self {
            embed: Param::from_tensor(Tensor::zeros([codebook_size, codebook_dim], device)),
        }
    }

    pub fn quantize(&self, hidden_states: Tensor<B, 2>) -> Tensor<B, 1, Int> {
        let embed = self.embed.val(); // [V, D]
        let embed_t = embed.clone().swap_dims(0, 1); // [D, V]

        let x_sq = hidden_states.clone().powf_scalar(2.0).sum_dim(1); // [N, 1]
        let y_sq = embed.powf_scalar(2.0).sum_dim(1).swap_dims(0, 1); // [1, V]

        let xy = hidden_states.matmul(embed_t); // [N, V]
        let dist = xy.mul_scalar(2.0) - x_sq - y_sq; // [N, V]

        dist.argmax(1).squeeze_dim(1)
    }

    pub fn encode(&self, hidden_states: Tensor<B, 2>) -> Tensor<B, 1, Int> {
        self.quantize(hidden_states)
    }

    pub fn decode(&self, embed_ind: Tensor<B, 1, Int>) -> Tensor<B, 2> {
        let indices_2d = embed_ind.unsqueeze_dim::<2>(1);
        let decoded_3d = burn::tensor::module::embedding(self.embed.val(), indices_2d);
        decoded_3d.squeeze_dim::<2>(1)
    }
}

// ---------------------------------------------------------------------------
// Vector Quantization Layer
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct VectorQuantization<B: Backend> {
    pub codebook: EuclideanCodebook<B>,
    pub project_in: Linear<B>,
    pub project_out: Linear<B>,
}

impl<B: Backend> VectorQuantization<B> {
    pub fn new(hidden_size: usize, codebook_size: usize, codebook_dim: usize, device: &B::Device) -> Self {
        Self {
            codebook: EuclideanCodebook::new(codebook_size, codebook_dim, device),
            project_in: LinearConfig::new(hidden_size, codebook_dim).init(device),
            project_out: LinearConfig::new(codebook_dim, hidden_size).init(device),
        }
    }

    pub fn encode(&self, x: Tensor<B, 3>) -> Tensor<B, 2, Int> {
        let x_t = x.swap_dims(1, 2);
        let projected = self.project_in.forward(x_t); // [B, T, D_c]
        let [b, t, d_c] = projected.dims();
        let flat = projected.reshape([b * t, d_c]);
        let indices = self.codebook.encode(flat); // [B * T]
        indices.reshape([b, t]) // [B, T]
    }

    pub fn decode(&self, indices: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [b, t] = indices.dims();
        let flat = indices.reshape([b * t]);
        let quantized = self.codebook.decode(flat); // [B * T, D_c]
        let projected = self.project_out.forward(quantized); // [B * T, H]
        let h = projected.dims()[1];
        let reshaped = projected.reshape([b, t, h]); // [B, T, H]
        reshaped.swap_dims(1, 2) // [B, H, T]
    }
}

// ---------------------------------------------------------------------------
// Residual Vector Quantizer (RVQ)
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct ResidualVectorQuantizer<B: Backend> {
    pub quantizers: Vec<VectorQuantization<B>>,
    pub hidden_size: usize,
}

impl<B: Backend> ResidualVectorQuantizer<B> {
    pub fn new(
        hidden_size: usize,
        codebook_size: usize,
        codebook_dim: usize,
        num_quantizers: usize,
        device: &B::Device,
    ) -> Self {
        let quantizers = (0..num_quantizers)
            .map(|_| VectorQuantization::new(hidden_size, codebook_size, codebook_dim, device))
            .collect();
        Self {
            quantizers,
            hidden_size,
        }
    }

    pub fn encode(&self, x: Tensor<B, 3>, bandwidth: Option<f32>, frame_rate: usize) -> Tensor<B, 3, Int> {
        let num_q = if let Some(bw) = bandwidth {
            // codebook_nbits = 10 (log2(1024))
            let bw_per_q = 10.0 * (frame_rate as f32) / 1000.0;
            let n = (bw / bw_per_q).floor() as usize;
            n.max(1).min(self.quantizers.len())
        } else {
            self.quantizers.len()
        };

        let mut residual = x;
        let mut all_indices = Vec::with_capacity(num_q);
        for quantizer in &self.quantizers[..num_q] {
            let indices = quantizer.encode(residual.clone()); // [B, T]
            let quantized = quantizer.decode(indices.clone()); // [B, H, T]
            residual = residual - quantized;
            all_indices.push(indices.unsqueeze_dim(1)); // [B, 1, T]
        }

        Tensor::cat(all_indices, 1) // [B, num_q, T]
    }

    pub fn decode(&self, codes: Tensor<B, 3, Int>) -> Tensor<B, 3> {
        let [b, c, t] = codes.dims();
        let device = codes.device();
        let mut quantized_out = Tensor::<B, 3>::zeros([b, self.hidden_size, t], &device);

        for i in 0..c {
            let quantizer = &self.quantizers[i];
            let indices = codes.clone().narrow(1, i, 1).squeeze_dim(1); // [B, T]
            let quantized = quantizer.decode(indices); // [B, H, T]
            quantized_out = quantized_out + quantized;
        }

        quantized_out
    }
}
