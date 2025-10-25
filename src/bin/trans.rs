use candle_core::{Module, Result, Tensor, Device, DType, IndexOp};
use candle_nn::{VarBuilder, VarMap, Linear, linear, Optimizer, AdamW, ParamsAdamW};
use ai::{pos, TransformerEncoder, TensorExt, TextEncoder};

const DIMS: usize = 384;
// 添加一个完整的模型结构，包含输出层
#[derive(Debug)]
struct StockPredictionModel {
    encoder: TransformerEncoder,
    output_projection: Linear,  // 将 [batch, seq_len, d_model] 映射到 [batch, seq_len, 2]
    pooling_layer: Linear,      // 可选：用于序列池化
}
impl StockPredictionModel {
    fn new(
        num_layers: usize,
        d_model: usize,
        heads: usize,
        d_ff: usize,
        dropout: f32,
        output_dim: usize,  // 2 for two stocks
        vb: VarBuilder,
    ) -> Result<Self> {
        let encoder = TransformerEncoder::new(
            num_layers,
            d_model,
            heads,
            d_ff,
            dropout,
            true,
            vb.pp("encoder"),
        )?;

        // 输出投影层：从 d_model 维度映射到 output_dim
        let output_projection = linear(d_model, output_dim, vb.pp("output_projection"))?;

        // 池化层：将序列维度压缩（可选）
        let pooling_layer = linear(d_model, d_model, vb.pp("pooling"))?;

        Ok(Self {
            encoder,
            output_projection,
            pooling_layer,
        })
    }
    fn forward(&self, x: &Tensor, mask: Option<&Tensor>, train: bool) -> Result<Tensor> {
        // Transformer编码
        let encoded = self.encoder.forward(x, mask, train)?; // [batch, seq_len, d_model]

        // 方法1: 直接对每个时间步预测
        let predictions = self.output_projection.forward(&encoded)?; // [batch, seq_len, 2]

        // 方法2: 或者使用平均池化后预测（取消注释使用）
        // let pooled = encoded.mean(1)?; // [batch, d_model] - 在seq_len维度上平均
        // let predictions = self.output_projection.forward(&pooled)?; // [batch, 2]
        // let predictions = predictions.unsqueeze(1)?; // [batch, 1, 2] 为了匹配target形状

        Ok(predictions)
    }
}

// 损失函数
fn mse_loss(predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
    let diff = (predictions - targets)?;
    let squared_diff = diff.sqr()?;
    squared_diff.mean_all()
}

// 使用示例
fn main() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let dtype = DType::F32;
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, dtype, &device);

    // 参数配置
    let d_model = DIMS;
    let num_heads = 8;
    let d_ff = 1536;
    let num_layers = 2;
    let dropout = 0.1;
    let output_dim = 2;
    let learning_rate = 5e-6;
    let num_epochs = 1000;

    // 创建模型
    let model = StockPredictionModel::new(
        num_layers,
        d_model,
        num_heads,
        d_ff,
        dropout,
        output_dim,
        vb.clone(),
    )?;

    // 创建优化器
    let params = varmap.all_vars();
    let mut optimizer = AdamW::new(
        params,
        ParamsAdamW {
            lr: learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
        }
    )?;

    let seq_len = sentences.len();

    // 准备输入数据：MiniLM 编码 + L2 归一化
    let text_encoder = TextEncoder::new("./models/minilm")?;
    let input = text_encoder.encode(sentences)?.into_iter().flatten().collect();
    let input = Tensor::from_vec(input, (seq_len, 384), &device)?;
    input.print("input", true);

    let input_2d = input; // 已是 [12, 384]
    let norms = input_2d.sqr()?.sum_keepdim(1)?.sqrt()?; // [12, 1]
    let normalized = input_2d.broadcast_div(&norms)?;   // L2 归一化

    // 计算句子0和句子6的相似度（油价涨 vs 油价跌）
    let sim = normalized.i(0)?.broadcast_mul(&normalized.i(6)?)?.sum_all()?;
    println!("句子0 与 句子6 余弦相似度: {:.4}", sim.to_scalar::<f32>()?);

    // ✅ 关键：reshape 为 [12, 1, 384] —— 12 个样本，每个 seq_len=1
    let input = normalized.unsqueeze(1)?; // [12, 1, 384]

    // 准备目标数据：[12, 2] → [12, 1, 2]
    let targets_flat: Vec<f32> = targets_data.iter().flatten().copied().collect();
    let targets_raw = Tensor::from_vec(targets_flat, (seq_len, 2), &device)?; // [12, 2]
    let targets_raw = targets_raw.unsqueeze(1)?; // [12, 1, 2]

    // 标准化：沿 batch 维度（dim=0）
    let (targets_norm, mean, std) = targets_raw.standardize(0)?;

    println!("开始训练...");

    // 训练循环
    for epoch in 0..num_epochs {
        let predictions = model.forward(&input, None, true)?; // [12, 1, 2]
        let loss = mse_loss(&predictions, &targets_norm)?;    // targets_norm: [12, 1, 2]
        let grads = loss.backward()?;
        optimizer.step(&grads)?;

        if epoch % 10 == 0 {
            let loss_val: f32 = loss.to_scalar()?;
            println!("Epoch {}: Loss = {:.6}", epoch, loss_val);

            if epoch % 50 == 0 {
                // 还原预测到原始尺度
                let pred_original = predictions.unstandardize(&mean, &std)?; // [12, 1, 2]

                // squeeze 掉 seq_len 维度用于打印
                let pred_2d = pred_original.squeeze(1)?; // [12, 2]
                let target_2d = targets_raw.squeeze(1)?; // [12, 2]

                let pred_vals = pred_2d.to_vec2::<f32>()?;
                let target_vals = target_2d.to_vec2::<f32>()?;

                println!("预测值 vs 真实值:");
                for i in 0..seq_len {
                    println!("  句子{}: [{:.4}, {:.4}] vs [{:.4}, {:.4}]",
                             i+1, pred_vals[i][0], pred_vals[i][1],
                             target_vals[i][0], target_vals[i][1]);
                }
                println!();
            }
        }
    }

    // 最终结果
    println!("\n训练完成！最终预测结果：");
    let final_predictions = model.forward(&input, None, false)?; // [12, 1, 2]
    let final_pred_original = final_predictions.unstandardize(&mean, &std)?; // 还原

    let pred_2d = final_pred_original.squeeze(1)?;
    let target_2d = targets_raw.squeeze(1)?;
    let pred_vals = pred_2d.to_vec2::<f32>()?;
    let target_vals = target_2d.to_vec2::<f32>()?;

    for i in 0..seq_len {
        println!("句子{}: 预测=[{:.4}, {:.4}], 真实=[{:.4}, {:.4}]",
                 i+1, pred_vals[i][0], pred_vals[i][1],
                 target_vals[i][0], target_vals[i][1]);
    }

    Ok(())
}