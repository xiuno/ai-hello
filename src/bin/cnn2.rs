use candle_core::{DType, Device, Module, ModuleT, Result, Tensor};
use candle_nn::{loss, ops, Conv2d, Linear, BatchNorm, Optimizer, VarBuilder, VarMap, ParamsAdamW, Dropout};

fn load_mnist_tensors(device: &Device) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
    let mnist = mnist::MnistBuilder::new()
        .base_path("data/")
        //.download_and_extract()
        .finalize();

    // 添加数据标准化（更好的做法）
    let train_images: Vec<f32> = mnist.trn_img.iter().map(|&x| (x as f32 / 255.0 - 0.1307) / 0.3081).collect();
    let test_images: Vec<f32> = mnist.tst_img.iter().map(|&x| (x as f32 / 255.0 - 0.1307) / 0.3081).collect();

    // 训练数据：60000 x 28 x 28 → 转为 f32 并归一化到 [0, 1]
    // let train_images: Vec<f32> = mnist.trn_img.iter().map(|&x| x as f32 / 255.0).collect();
    // let test_images: Vec<f32> = mnist.tst_img.iter().map(|&x| x as f32 / 255.0).collect();

    let train_labels: Vec<u32> = mnist.trn_lbl.iter().map(|&x| x as u32).collect();
    let test_labels: Vec<u32> = mnist.tst_lbl.iter().map(|&x| x as u32).collect();

    let train_images = Tensor::from_vec(train_images, (60000, 1, 28, 28), device)?; // CNN
    let test_images = Tensor::from_vec(test_images, (10000, 1, 28, 28), device)?;

    let train_labels = Tensor::from_vec(train_labels, (60000,), device)?;
    let test_labels = Tensor::from_vec(test_labels, (10000,), device)?;

    Ok((train_images, train_labels, test_images, test_labels))
}

struct NeuralNetwork {
    conv1: Conv2d, // 输入: [B, 1, 28, 28] -> 输出: [B, 32, 28, 28]
    bn1: BatchNorm,
    conv2: Conv2d, // 输入: [B, 32, 28, 28] -> 输出: [B, 64, 28, 28]
    bn2: BatchNorm,
    // 池化  [B, 64, 14, 14]
    dropout1: Dropout,
    dropout2: Dropout,
    fc1: Linear, // 64 * 14 * 14
    fc2: Linear, // 10个类别(MNIST数字0-9)
}

impl NeuralNetwork {
    fn new(vs: VarBuilder) -> Result<Self> {
        let conv_cfg = candle_nn::Conv2dConfig { padding: 1, ..Default::default() };
        let conv1 = candle_nn::conv2d(1, 32, 3, conv_cfg, vs.pp("conv1"))?;
        let conv2 = candle_nn::conv2d(32, 64, 3, conv_cfg, vs.pp("conv2"))?; //
        let bn1 = candle_nn::batch_norm(32, 1e-5, vs.pp("bn1"))?;
        let bn2 = candle_nn::batch_norm(64, 1e-5, vs.pp("bn2"))?;

        let dropout1 = Dropout::new(0.25);
        let dropout2 = Dropout::new(0.3);
        let fc1 = candle_nn::linear( 64*7*7, 128, vs.pp("fc1"))?;
        let fc2 = candle_nn::linear(128, 10, vs.pp("fc2"))?;
        Ok(Self { conv1, bn1, conv2, bn2, dropout1, dropout2, fc1, fc2 })
    }
    fn forward(&self, input: &Tensor, train: bool) -> Result<Tensor> {
        // 卷积1
        let x = self.conv1.forward(&input)?;
        let x = self.bn1.forward_t(&x, train)?.relu()?;
        let x = x.max_pool2d(2)?;

        // 卷积2
        let x = self.conv2.forward(&x)?;
        let x = self.bn2.forward_t(&x, train)?.relu()?;
        let x = x.max_pool2d(2)?;
        let x = self.dropout1.forward(&x, train)?;

        // 全连接
        let x = x.flatten_from(1)?;
        let x = self.fc1.forward(&x)?.relu()?;
        let x = self.dropout2.forward(&x, train)?;
        let x = self.fc2.forward(&x)?;

        Ok(x)
    }
    fn predict(&self, input: &Tensor) -> Result<Vec<u32>> {
        let logits = self.forward(input, false)?;
        let probs = ops::softmax(&logits, 1)?;
        let pred = probs.argmax(1)?;
        pred.to_vec1()
    }

}

fn compute_agreement_accuracy(predicted_labels: &[u32], reference_labels: &[u32]) -> f32 {
    assert_eq!( predicted_labels.len(), reference_labels.len());
    let matching_count = predicted_labels
        .iter()
        .zip(reference_labels)
        .filter(|(&pred, &label)| pred == label)
        .count();
    (matching_count as f32 / reference_labels.len() as f32) * 100.0
}

/// 优化: 同时打乱多个张量（保持对应关系）
use rand::seq::SliceRandom;
use rand::thread_rng;
fn shuffle_tensors(tensors: &[&Tensor]) -> Result<Vec<Tensor>> {
    if tensors.is_empty() {
        return Ok(vec![]);
    }

    let size = tensors[0].dims()[0];
    let device = tensors[0].device();

    // 生成随机排列
    let mut indices: Vec<u32> = (0..size as u32).collect();
    indices.shuffle(&mut thread_rng());
    let permutation = Tensor::from_vec(indices, (size,), device)?;

    // 用相同的排列打乱所有张量
    tensors.iter()
        .map(|t| t.index_select(&permutation, 0))
        .collect()
}

fn main() -> Result<()> {
    //let device = &Device::Cpu;
    let device = &Device::new_metal(0)?;
    // let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    // let device_info = device.location(); // 新增设备信息查询

    // 加载 MNIST 数据
    let (train_images, train_labels, test_images, test_labels) = load_mnist_tensors(device)?;

    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let model = NeuralNetwork::new(vs)?;
    // let mut optimizer = candle_nn::SGD::new(varmap.all_vars(), 0.01)?;
    // todo: 更快的收敛方案, 现代 AI 的做法
    let mut optimizer = candle_nn::AdamW::new(
        varmap.all_vars(),
        ParamsAdamW {
            lr: 3e-4,          // 学习率
            weight_decay: 1e-4, // 添加权重衰减
            beta1: 0.9,        // 明确设置动量参数
            beta2: 0.999,
            eps: 1e-8,
            ..Default::default()
        }
    )?;

    println!("开始训练...");
    const BATCH_SIZE: usize = 256; // 每批处理的图片数
    const CHUNKS: usize = 60000 / BATCH_SIZE;

    // 第1轮: 96-97%
    // 第5轮: 98%
    // 第10轮: 98.5-99%
    // 最终: 99.2%
    for epoch in 0..20 {
        let shuffled = shuffle_tensors(&[&train_images, &train_labels])?;
        let shuffled_images = &shuffled[0];
        let shuffled_labels = &shuffled[1];
        let image_batches = shuffled_images.chunk(CHUNKS, 0)?; // 60000/CHUNKS
        let label_batches = shuffled_labels.chunk(CHUNKS, 0)?; // 60000/CHUNKS

        let mut epoch_loss = 0.0;
        for(batch_idx, (images, lables)) in image_batches.iter().zip(&label_batches).enumerate() {
            let logits = model.forward(&images, true)?;
            let loss = loss::cross_entropy(&logits, &lables)?;
            optimizer.backward_step(&loss)?;
            epoch_loss += loss.to_vec0::<f32>()?;
            if batch_idx % 10 == 0 {
                println!("Epoch {}, Batch {}/{}, Loss: {:.4}", epoch, batch_idx, CHUNKS, loss.to_vec0::<f32>()?);
            }
        }
        let avg_loss = epoch_loss / CHUNKS as f32;
        println!("Epoch {} 完成, 平均损失: {:.4}", epoch, avg_loss);
        let predictions = model.predict(&test_images)?;
        let epoch_test_label = test_labels.to_vec1::<u32>()?;
        let accuracy = compute_agreement_accuracy(&predictions, &epoch_test_label);
        println!("准确率: {}%", accuracy);

        // todo: 优化, 此段代码可删除
        if (epoch + 1) % 5 == 0 {
            let new_lr = optimizer.learning_rate() * 0.9;
            optimizer.set_learning_rate(new_lr);
            println!("学习率降低至: {:.6}", new_lr);
        }
    }

    Ok(())
}