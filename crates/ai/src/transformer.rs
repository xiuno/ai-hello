use candle_core::{Module, Result, Tensor, D};
use candle_nn::{layer_norm, linear, LayerNorm, Linear, VarBuilder};

// 说 MiniLM 对齐
// const DIMS: usize = 384;

#[derive(Debug)]
struct MultiHeadAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    head_dim: usize,
    num_heads: usize,
}

impl MultiHeadAttention {
    // d_model = DIMS, num_headers = 8
    fn new(d_model: usize, num_heads: usize, vb: VarBuilder) -> Result<Self> {
        assert_eq!(d_model % num_heads, 0, "d_model must be divisible by heads");
        let head_dim = d_model / num_heads; // 64

        // 输入形状: [B, L, DIMS], 输出 [B, L, DIMS]
        // linear() 全自动随机化里面的值, 不使用 BERT 中的值
        let q_proj = linear(d_model, d_model, vb.pp("q_proj"))?;
        let k_proj = linear(d_model, d_model, vb.pp("k_proj"))?;
        let v_proj = linear(d_model, d_model, vb.pp("v_proj"))?;
        // 最后一步的投影矩阵
        let out_proj = linear(d_model, d_model, vb.pp("out_proj"))?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            head_dim: head_dim,
            num_heads,
        })
    }

    fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        // seq_len = 6, 对应维度为 L
        let (batch_size, seq_len, d_model) = x.dims3()?;

        assert_eq!(
            d_model,
            self.num_heads * self.head_dim,
            "d_model must equal num_heads * head_dim"
        );

        // 投影并重塑为多头形状
        // [B, num_heads L, head_dim] [2, 8, 10, 48]
        let q = self
            .q_proj
            .forward(x)?
            .reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?; // transpose 惰性改变, 合并以后, 一次性 contiguous() 物理移动.

        // [B, num_heads L, head_dim] [2, 8, 10, 48]
        let k = self
            .k_proj
            .forward(x)?
            .reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // [B, num_heads L, head_dim] [2, 8, 10, 48]
        let v = self
            .v_proj
            .forward(x)?
            .reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // 计算注意力分数: Q @ K^T / sqrt(d_k)
        let scale = (self.head_dim as f64).sqrt();
        // 在 candle（以及大多数深度学习框架）中，matmul 对最后两个维度执行标准的矩阵乘法，前面的维度（batch-like）进行广播。
        //  [B, num_heads L, head_dim] x [B, num_heads head_dim, L] = [B, num_heads L, L]
        // 对 k 最后两个维度进行转置. q x k^T
        let attn_scores = q.matmul(&k.transpose(D::Minus1, D::Minus2)?)?;
        // .affine(scale, bias) --> x * scale + bias
        let attn_scores = attn_scores.affine(1.0 / scale, 0.0)?; // [B, num_heads L, L]

        // 应用 mask（如果有）, 股票预测不需要.
        // 应用 padding mask（如果有）
        let attn_scores = if let Some(mask) = mask {
            // mask: [B, L], 1=valid, 0=pad
            let mask = mask.unsqueeze(1)?.unsqueeze(1)?; // [B, 1, 1, L]
            let neg_inf = Tensor::new(f32::NEG_INFINITY, x.device())?
                .broadcast_as(attn_scores.shape())?;
            // 将 mask 扩展到 [B, H, L, L]
            let mask = mask.broadcast_as(attn_scores.shape())?;
            mask.where_cond(&attn_scores, &neg_inf)?
        } else {
            attn_scores
        };

        // Softmax  [B, num_heads L, L] 形状不变
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_scores)?;

        // 应用注意力权重到 V, [B, num_heads L, L] x [B, num_heads L, head_dim] = [B, num_heads L, head_dim]
        let attn_output = attn_weights.matmul(&v)?;

        // 合并多头 [B, L, heads x head_dim]
        let attn_output = attn_output
            .transpose(1, 2)? // [B, num_heads L, head_dim] -> [B, L, heads, head_dim]
            .reshape((batch_size, seq_len, d_model))?; // [B, L, heads x head_dim]

        // 输出投影
        self.out_proj.forward(&attn_output)
    }
}

#[derive(Debug)]
struct FeedForward {
    lin1: Linear,
    lin2: Linear,
}

impl FeedForward {
    // d_ff 通常为 2048
    fn new(d_model: usize, d_ff: usize, vb: VarBuilder) -> Result<Self> {
        let lin1 = linear(d_model, d_ff, vb.pp("lin1"))?;
        let lin2 = linear(d_ff, d_model, vb.pp("lin2"))?;
        Ok(Self { lin1, lin2 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.lin1.forward(x)?.gelu()?; // [B, L, d_ff]
        self.lin2.forward(&x) // [B, L, d_model]
    }
}

#[derive(Debug)]
struct TransformerEncoderLayer {
    self_attn: MultiHeadAttention,
    ffn: FeedForward,
    norm1: LayerNorm,
    norm2: LayerNorm,
    dropout: f32,
}

// 里面连接多头 + FNN
impl TransformerEncoderLayer {
    fn new(
        d_model: usize,
        heads: usize,
        d_ff: usize,
        dropout: f32,
        vb: VarBuilder,
    ) -> Result<Self> {
        let self_attn = MultiHeadAttention::new(d_model, heads, vb.pp("self_attn"))?;
        let ffn = FeedForward::new(d_model, d_ff, vb.pp("ffn"))?;
        // todo: 这里与 MiniML 保持一致: 使用 1e-12, 而非 1e-5
        let norm1 = layer_norm(d_model, 1e-12, vb.pp("norm1"))?;
        let norm2 = layer_norm(d_model, 1e-12, vb.pp("norm2"))?;

        Ok(Self {
            self_attn,
            ffn,
            norm1,
            norm2,
            dropout,
        })
    }

    // todo: 现在先使用简单模型
    // 现在流行做法: x → LayerNorm → Self-Attention → Dropout → Add residual → LayerNorm → FCN → Dropout → Add residual
    fn forward(&self, x: &Tensor, mask: Option<&Tensor>, train: bool) -> Result<Tensor> {
        // Self-Attention with residual connection
        let attn_output = self.self_attn.forward(x, mask)?;
        // attn_output.print("attn_output", true);
        let attn_output = if train && self.dropout > 0.0 {
            candle_nn::ops::dropout(&attn_output, self.dropout)?
        } else {
            attn_output
        };
        let x = self.norm1.forward(&(x + attn_output)?)?;

        // Feed-Forward with residual connection
        let ffn_output = self.ffn.forward(&x)?;
        let ffn_output = if train && self.dropout > 0.0 {
            candle_nn::ops::dropout(&ffn_output, self.dropout)?
        } else {
            ffn_output
        };
        self.norm2.forward(&(&x + ffn_output)?)
    }
}

// 里面堆叠多层
#[derive(Debug)]
pub struct TransformerEncoder {
    layers: Vec<TransformerEncoderLayer>,
    norm: Option<LayerNorm>,
}

impl TransformerEncoder {
    pub fn new(
        num_layers: usize,
        d_model: usize,
        heads: usize,
        d_ff: usize,
        dropout: f32,
        use_final_norm: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            layers.push(TransformerEncoderLayer::new(
                d_model,
                heads,
                d_ff,
                dropout,
                vb.pp(&format!("layer_{}", i)),
            )?);
        }

        let norm = if use_final_norm {
            Some(layer_norm(d_model, 1e-5, vb.pp("final_norm"))?)
        } else {
            None
        };

        Ok(Self { layers, norm })
    }

    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>, train: bool) -> Result<Tensor> {
        let mut output = x.clone();
        for layer in &self.layers {
            output = layer.forward(&output, mask, train)?;
        }
        // output.print("output2", true);

        if let Some(norm) = &self.norm {
            output = norm.forward(&output)?;
        }

        Ok(output)
    }
}
