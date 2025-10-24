use candle_core::{Module, Result, Tensor, Device, DType};
use candle_nn::{VarBuilder, VarMap};
use ai::{text_encoder, candle_ext, pos, TransformerEncoder, TensorExt, TextEncoder};

const DIMS: usize = 384;

// 使用示例
fn main() -> Result<()> {
    let device = Device::Cpu;
    let dtype = DType::F32;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, dtype, &device);

    // 参数配置
    let d_model = DIMS;
    let num_heads = 8;
    let d_ff = 1536; // 2048;
    let num_layers = 1;
    let dropout = 0.1;

    // 创建模型
    let encoder = TransformerEncoder::new(
        num_layers,
        d_model,
        num_heads,
        d_ff,
        dropout,
        true, // use_final_norm
        vb.clone(),
    )?;

    // 示例输入 [batch_size, seq_len, d_model]
    let sentences = vec![
        "Apple stock rises after strong earnings report.",
        "The weather is sunny and warm today.",
        "Oil prices drop due to oversupply concerns.",
        "中国好声音",
    ];
    let seq_len = sentences.len();
    // 对应的两支股票的涨跌幅度
    let targets = vec![
        [0.6, 0.4],
        [0.8, -0.5],
        [0.2, 0.1],
        [0.1, 0.4],
        [0.9, 0.2],
    ];
    // 将 Vec<Vec<f32>> 展平, 方便张量自由操作维度
    let text_encoder = TextEncoder::new("./models/minilm").unwrap();
    let input = text_encoder.encode(sentences).unwrap().into_iter().flatten().collect();
    let input = Tensor::from_vec(input, (seq_len, 384), &device).unwrap();
    let input = input.unsqueeze(0).unwrap();
    input.print("input1", true);

    // 添加位置编码
    let pos_encoding = pos::sin_cos::encoding(seq_len, d_model, &device, dtype)?;
    // pos_encoding shape: [seq_len, d_model]
    // 广播到 [batch_size, seq_len, d_model]
    let input_with_pos = input.broadcast_add(&pos_encoding.unsqueeze(0)?)?;
    input_with_pos.print( "input_with_pos", true);

    for epoch in 0..1 {
        // shape: [4, 384]
        // 前向传播
        let output = encoder.forward(&input_with_pos, None, false)?;
        output.print("output", true);

        // 对比 target, 计算损失, 反向传播
        // 完善此处代码...
    }

    Ok(())
}
