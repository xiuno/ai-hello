use anyhow::{Context, Result};
use safetensors::tensor::{TensorView, View};
use std::collections::HashMap;
use std::fs;

fn main() -> Result<()> {
    let args: Vec<String> = vec!["".to_string(), "./data/bert/model.safetensors.old".to_string(), "./data/bert/model.safetensors".to_string()];
    if args.len() != 3 {
        eprintln!("Usage: {} <input.safetensors> <output.safetensors>", args[0]);
        std::process::exit(1);
    }
    let input_path = &args[1];
    let output_path = &args[2];

    // 1. Read and deserialize input
    let data = fs::read(input_path)
        .with_context(|| format!("Failed to read {}", input_path))?;
    let safe_tensors = safetensors::SafeTensors::deserialize(&data)
        .with_context(|| "Failed to deserialize safetensors")?;

    // 2. 打印原始张量名称用于调试
    println!("Original tensors:");
    for (key, _) in safe_tensors.tensors() {
        println!("  {}", key);
    }

    // 3. 更保守的键重映射
    let mut tensors: HashMap<String, TensorView> = HashMap::new();
    for (key, tensor) in safe_tensors.tensors() {
        let new_key = if key.ends_with(".gamma") {
            key.replace(".gamma", ".weight")
        } else if key.ends_with(".beta") {
            key.replace(".beta", ".bias")
        } else {
            key.to_string()
        };

        // 移除 "bert." 前缀，但保留其他结构
        let new_key = if new_key.starts_with("bert.") {
            new_key.strip_prefix("bert.").unwrap().to_string()
        } else {
            new_key
        };

        // 对于 Jina BERT，可能需要特殊处理 MLP 层
        // 暂时不跳过任何张量，先看看所有张量
        println!("Mapping: {} -> {}", key, new_key);
        tensors.insert(new_key, tensor);
    }

    // 4. 打印转换后的张量名称
    println!("Converted tensors:");
    for key in tensors.keys() {
        println!("  {}", key);
    }

    // 5. Serialize to new safetensors file
    let serialized = safetensors::serialize(&tensors, &Some(HashMap::new()))
        .with_context(|| "Failed to serialize safetensors")?;

    fs::write(output_path, serialized)
        .with_context(|| format!("Failed to write {}", output_path))?;

    println!("✅ Successfully converted: {} -> {}", input_path, output_path);
    println!("Total tensors converted: {}", tensors.len());
    Ok(())
}