use anyhow::{Error as E, Result};
use candle_core::{Device, IndexOp, DType, Tensor};
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use candle_nn::VarBuilder;
use tokenizers::{InputSequence, Tokenizer};
use std::path::Path;
use std::fs;

pub fn encode<S: AsRef<str>>(sentences: Vec<S>) -> Result<Vec<Vec<f32>>> {
    // 1. 设置设备(自动选择 Metal / CUDA / CPU)
    // MiniLM & Bert 内部都使用了 LayerNorm, 只支持 CPU & CUDA
    // let device = if cfg!(target_os = "macos") {  Device::new_metal(0)? }
    let device = if cfg!(feature = "cuda") {
        Device::new_cuda(0)?
    } else {
        Device::Cpu
    };
    // let device = Device::new_metal(0)?;
    println!("Using device: {:?}", device);

    // 2. 加载 tokenizer
    let tokenizer_path = Path::new("models/minilm/tokenizer.json");
    let tokenizer = Tokenizer::from_file(tokenizer_path).unwrap();

    // 3. 加载模型配置和权重
    let config_path = "models/minilm/config.json";
    let weights_path = "models/minilm/model.safetensors";

    // 读取配置文件并解析
    let config_str = fs::read_to_string(config_path)?;
    let config: BertConfig = serde_json::from_str(&config_str)?;

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)? };
    let model = BertModel::load(vb, &config)?;

    // 5. 编码并获取 embeddings
    let mut ret = vec![];
    for sentence in sentences {
        let embedding = encode_sentence(sentence.as_ref(), &tokenizer, &model, &device)?;
        println!("Sentence: {}", sentence.as_ref());
        println!("Embedding shape: {:?}", embedding.dims());
        println!("First 5 dims: {:?}", embedding.to_vec1::<f32>()?[..5].to_vec());
        ret.push(embedding.to_vec1::<f32>()?);
    }
    Ok(ret)
}

#[test]
fn test_1() {
    // 4. 输入句子
    let sentences = vec![
        "Apple stock rises after strong earnings report.",
        "The weather is sunny and warm today.",
        "Oil prices drop due to oversupply concerns.",
        "中国好声音",
    ];
    let r = encode(sentences).unwrap();
    println!("{:?}", r);
}

fn encode_sentence (
    sentence: &str,
    tokenizer: &Tokenizer,
    model: &BertModel,
    device: &Device,
) -> Result<Tensor> {
    // Tokenize
    let encoded = tokenizer
        .encode(sentence, true)
        .map_err(|e| E::msg(format!("Tokenization error: {}", e)))?;

    // [101, 7592, 2088, 102]
    let tokens = encoded.get_ids();

    // [[101, 7592, 2088, 102]]
    let token_ids = Tensor::from_slice(&tokens[..], tokens.len(), device)?
        .unsqueeze(0)?; // [1, seq_len]

    // [[0, 0, 0, 0]]
    let token_type_ids = Tensor::zeros(token_ids.shape(), DType::U32, device)?;

    // [[1, 1, 1, 1]]
    let attention_mask = Tensor::from_slice(
        &encoded.get_attention_mask()[..],
        encoded.get_attention_mask().len(),
        device,
    )?
        .unsqueeze(0)?; // [1, seq_len]

    // Forward
    // shape: [1, 4, 384]
    /*
        [
          [
            [0.12, -0.34, ..., 0.56],   // [CLS]
            [0.21,  0.45, ..., -0.12],  // Hello
            [-0.05, 0.67, ..., 0.33],   // world
            [0.01, -0.22, ..., 0.89] ]  // [SEP]
     */
    let hidden_states = model.forward(&token_ids, &token_type_ids, Some(&attention_mask))?;

    let hidden_size = hidden_states.dim(2)?; // 384
    let seq_len = hidden_states.dim(1)?;    // 4

    // Get attention mask as f32 for masking
    // 原始 mask: [[1, 1, 1, 1]] → 转为 f32 → [[[1.0], [1.0], [1.0], [1.0]]]
    let attention_mask_f32 = attention_mask
        .to_dtype(DType::F32)?
        .unsqueeze(2)? // [1, seq_len, 1]
        .broadcast_as((1, seq_len, hidden_size))?;

    // Apply mask and compute mean (skip [PAD] tokens)
    // 均值池化, 丢失了位置信息, 但保留了高维空间的语义.
    // [1, 4, 384]
    let masked_hidden = hidden_states.broadcast_mul(&attention_mask_f32)?;
    let sum = masked_hidden.sum(1)?; // [1, 384(hidden_size)]
    // [4.0, 4.0, ..., 4.0]
    let mask_sum = attention_mask_f32.sum(1)?; // [1, 384(hidden_size)]
    // shape: [1, 384]
    // [0.29/4, ..., 1.66/4] ≈ [0.0725, ..., 0.415]
    let mean_pooled = sum.broadcast_div(&mask_sum)?;

    // L2 normalize (替代 broadcast_norm_l2)
    // norm：标量（但形状 [1, 1]），例如 sqrt(0.0725² + ... + 0.415²) ≈ 1.23
    // normalized：每个元素除以该 norm
    // 形状仍为 [1, 384]
    // 结果是单位向量（L2 norm = 1）
    let norm = mean_pooled.sqr()?.sum_keepdim(1)?.sqrt()?;
    let normalized = mean_pooled.broadcast_div(&norm)?;

    // Remove batch dim -> [hidden_size]
    let tensor = normalized.squeeze(0)?;
    Ok(tensor)
}