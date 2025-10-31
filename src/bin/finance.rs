use candle_core::{Device, Result, Tensor, DType, Var};
use candle_nn::{Linear, Adam, loss, ops::conv1d, Module};

const KERNEL_SIZE: usize = 30;
const SEQ_LEN: usize = 365;
const STOCKS: usize = 2;

fn main() -> Result<()> {
    // === 1. 设备选择 ===
    let device = if cfg!(feature = "cuda") {
        Device::new_cuda(0)?
    } else if cfg!(feature = "metal") {
        Device::new_metal(0)?
    } else {
        Device::Cpu
    };
    println!("Using device: {:?}", device);

    // === 2. 模拟数据 ===
    let mut fed_op = vec![0.0f32; SEQ_LEN];
    fed_op[99] = 0.05;   // 第100天加息
    fed_op[199] = -0.25; // 第200天降息

    let mut stock_data = vec![[0.0f32; STOCKS]; SEQ_LEN];
    // 简单模拟：加息后几天股票下跌
    for i in 0..SEQ_LEN {
        let mut noise = (i as f32 * 0.01).sin() * 0.01;
        if i > 99 && i <= 110 { noise -= 0.02; } // 加息后回调
        if i > 199 && i <= 210 { noise += 0.015; } // 降息后反弹
        stock_data[i] = [noise + 0.001, noise - 0.001];
    }

    // 转为张量
    let fed_tensor = Tensor::from_slice(&fed_op, (1, SEQ_LEN, 1), &device)?;
    let stock_flat: Vec<f32> = stock_data.iter().flat_map(|x| x.iter()).copied().collect();
    let stock_tensor = Tensor::from_slice(&stock_flat, (1, SEQ_LEN, STOCKS as usize), &device)?;

    // === 3. 可训练衰减核 ===
    let initial_kernel: Vec<f32> = (0..KERNEL_SIZE)
        .map(|i| (-0.07 * i as f32).exp()) // 初始指数衰减
        .collect();
    let kernel_var = Var::from_slice(&initial_kernel, &device)?;
    println!("Initial decay kernel (first 10): {:?}", &initial_kernel[..10]);

    // === 4. 模型层 ===
    let linear1 = Linear::new(1 + STOCKS, 32, &device)?;
    let linear2 = Linear::new(32, STOCKS, &device)?;

    // === 5. 训练设置 ===
    let mut vars = kernel_var.clone().all_vars();
    vars.extend(linear1.all_vars());
    vars.extend(linear2.all_vars());
    let mut optimizer = Adam::new(vars, 1e-2)?;

    // === 6. 训练循环 ===
    for epoch in 0..200 {
        // 构造因果卷积输入
        let padding = KERNEL_SIZE - 1;
        let zeros = Tensor::zeros((1, padding, 1), DType::F32, &device)?;
        let fed_padded = Tensor::cat(&[&zeros, &fed_tensor], 1)?;
        let kernel_t = kernel_var.as_tensor().unsqueeze(0)?.unsqueeze(0)?; // [1,1,K]
        let fed_impact = conv1d(&fed_padded, &kernel_t, 1, 0, 1, 1)?; // [1,365,1]

        // 拼接特征
        let features = Tensor::cat(&[&stock_tensor, &fed_impact], 2)?; // [1,365,3]

        // 预测下一日（t -> t+1）
        let mut preds = Vec::new();
        let mut targets = Vec::new();
        for t in 0..(SEQ_LEN - 1) {
            let x = features.i(t)?.unsqueeze(0)?; // [1, 3]
            let h = linear1.forward(&x)?.relu()?;
            let y = linear2.forward(&h)?;
            preds.push(y);
            targets.push(stock_tensor.i(t + 1)?);
        }
        let pred_all = Tensor::cat(&preds, 0)?;
        let true_all = Tensor::cat(&targets, 0)?;

        let loss = loss::mse(&pred_all, &true_all)?;
        if epoch % 50 == 0 {
            println!("Epoch {}: loss = {:.6}", epoch, loss.to_vec0::<f32>()?);
        }

        optimizer.backward_step(&loss)?;
    }

    // === 7. 打印可解释的衰减核 ===
    let final_kernel = kernel_var.to_vec1::<f32>()?;
    println!("\n=== Learned Fed Impact Decay Kernel (first 15 days) ===");
    for (day, weight) in final_kernel.iter().take(15).enumerate() {
        println!("Day {}: {:.4}", day, weight);
    }
    println!("(Higher weight = stronger market reaction on that lag day)");

    Ok(())
}