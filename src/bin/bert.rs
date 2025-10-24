use candle_core::{DType, Device, Tensor, IndexOp, Module};
use safetensors::tensor::SafeTensors;
use candle_nn::VarBuilder;
use tokenizers::tokenizer::Tokenizer;
use anyhow::Result;
use std::fs::File;
use std::io::Read;
// 使用标准的 BERT 实现而不是 Jina BERT
use candle_transformers::models::bert::{BertModel, Config};

fn main() -> Result<()> {
    // 1. 设置设备
    // let device = Device::new_metal(0).unwrap_or(Device::Cpu);
    let device = Device::Cpu;
    println!("Using device: {:?}", device);

    // 2. 加载分词器
    let tokenizer = Tokenizer::from_file("./data/bert/tokenizer.json").map_err(|e| anyhow::anyhow!(e))?;

    // 3. 加载模型配置
    let config_file = File::open("./data/bert/config.json")?;
    let config: Config = serde_json::from_reader(config_file)?;

    // 4. 加载模型权重
    let mut weights_buf = Vec::new();
    File::open("./data/bert/model.safetensors")?.read_to_end(&mut weights_buf)?;
    let safetensors = SafeTensors::deserialize(&weights_buf)?;

    println!("Available tensors in model:");
    for (name, tensor) in safetensors.tensors() {
        println!("  {} - shape: {:?}", name, tensor.shape());
    }

    let vb = VarBuilder::from_buffered_safetensors(weights_buf, DType::F32, &device)?;

    // 5. 构建标准 BERT 模型
    let model = BertModel::load(vb, &config)?;

    // 6. 准备输入文本
    let texts = ["你好，世界！", "Rust Candle 很棒。"];
    println!("Processing texts: {:?}", texts);

    for text in texts.iter() {
        println!("\nInput: {}", text);
        let encoding = tokenizer.encode(*text, true).map_err(|e| anyhow::anyhow!(e))?;

        let input_ids = Tensor::new(encoding.get_ids(), &device)?.unsqueeze(0)?;
        let token_type_ids = Tensor::new(encoding.get_type_ids(), &device)?.unsqueeze(0)?;
        let attention_mask = Tensor::new(encoding.get_attention_mask(), &device)?.unsqueeze(0)?;

        println!("Input IDs shape: {:?}", input_ids.shape());
        println!("Attention Mask shape: {:?}", attention_mask.shape());

        // 7. 模型推理
        let output = model.forward(&input_ids, &token_type_ids, Some(&attention_mask))?;

        // 8. 处理输出:取[CLS]标记的嵌入
        let cls_embedding = output.i((0, 0))?;
        let embedding_slice = cls_embedding.to_vec1::<f32>()?;
        println!("[CLS] Embedding (first 10 dims): {:?}",
                 embedding_slice.iter().take(10).collect::<Vec<_>>());
        println!("Embedding dimension: {}", embedding_slice.len());
    }

    Ok(())
}