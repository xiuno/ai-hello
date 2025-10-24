use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{loss, ops, Conv2d, Linear, Optimizer, VarBuilder, VarMap, ParamsAdamW};

fn load_mnist_tensors(device: &Device) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
    let mnist = mnist::MnistBuilder::new()
        .base_path("data/")
        //.download_and_extract()
        .finalize();

    // 训练数据：60000 x 28 x 28 → 转为 f32 并归一化到 [0, 1]
    let train_images: Vec<f32> = mnist.trn_img.iter().map(|&x| x as f32 / 255.0).collect();
    let train_labels: Vec<u32> = mnist.trn_lbl.iter().map(|&x| x as u32).collect();
    let test_images: Vec<f32> = mnist.tst_img.iter().map(|&x| x as f32 / 255.0).collect();
    let test_labels: Vec<u32> = mnist.tst_lbl.iter().map(|&x| x as u32).collect();

    let train_images = Tensor::from_vec(train_images, (60000, 1, 28, 28), device)?; // CNN
    let train_labels = Tensor::from_vec(train_labels, (60000,), device)?;

    let test_images = Tensor::from_vec(test_images, (10000, 1, 28, 28), device)?;
    let test_labels = Tensor::from_vec(test_labels, (10000,), device)?;

    Ok((train_images, train_labels, test_images, test_labels))
}

// 30 Epoch: 正确率达到 97.6%
struct NerualNetwork {
    conv1: Conv2d, // 输入: [B, 1, 28, 28] -> 输出: [B, 8, 24, 24]
    conv2: Conv2d, // 输入: [B, 8, 24, 24] -> 输出: [B, 16, 20, 20]
    fc1: Linear, // 输入: 16 * 20 * 20 = 6400, 输出: 128
    fc2: Linear, // 输入: 128个特征 -> 10个类别(MNIST数字0-9)
}

impl NerualNetwork {
    fn new(vs: VarBuilder) -> Result<Self> {
        // 卷积层1: 输入通道1(灰度图, RGB图为3), 输出通道8, 卷积核5x5
        // 输入: [B, 1, 28, 28] -> 输出: [B, 8, 24, 24]
        // 计算: H_out = (28 - 5) / 1 + 1 = 24 (默认stride=1, padding=0)
        let conv1 = candle_nn::conv2d(1, 8, 5, std::default::Default::default(), vs.pp("conv1"))?;
        // 卷积层2: 输入通道8, 输出通道16, 卷积核5x5
        // 输入: [B, 8, 24, 24] -> 输出: [B, 16, 20, 20]
        // 计算: H_out = (24 - 5) / 1 + 1 = 20
        let conv2 = candle_nn::conv2d(8, 16, 5, std::default::Default::default(), vs.pp("conv2"))?; // [B, 16, 20, 20]
        // 16 * 20 * 20 = 6400
        let fc1 = candle_nn::linear(16*20*20, 128, vs.pp("fc1"))?;
        // 输出层: 128个特征 -> 10个类别(MNIST数字0-9)
        let fc2 = candle_nn::linear(128, 10, vs.pp("fc2"))?;
        Ok(Self { conv1, conv2, fc1, fc2 })
    }
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // 第一层卷积 + ReLU 激活
        let x = self.conv1.forward(&input)?.relu()?;

        // 第二层卷积 + ReLU 激活
        let x = self.conv2.forward(&x)?.relu()?;

        // 展平操作: [B, 16, 20, 20] -> [B, 16*20*20]
        let x = x.flatten_from(1)?;

        // 第一层全连接 + ReLU 激活
        let x = self.fc1.forward(&x)?.relu()?;

        // 输出层, 这里不能使用 relu() !
        let x = self.fc2.forward(&x)?;

        Ok(x)
    }
    fn predict(&self, input: &Tensor) -> Result<Vec<u32>> {
        let logits = self.forward(input)?;
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

fn main() -> Result<()> {
    //let device = &Device::Cpu;
    let device = &Device::new_metal(0)?;

    // 加载 MNIST 数据
    let (train_images, train_labels, test_images, test_labels) = load_mnist_tensors(device)?;

    println!("训练集图片形状: {:?}", train_images.shape());
    println!("训练集标签形状: {:?}", train_labels.shape());
    println!("测试集图片形状: {:?}", test_images.shape());
    println!("测试集标签形状: {:?}", test_labels.shape());

    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let model = NerualNetwork::new(vs)?;
    // let mut optimizer = candle_nn::SGD::new(varmap.all_vars(), 0.01)?;
    // todo: 更快的收敛方案, 现代 AI 的做法
    let mut optimizer = candle_nn::AdamW::new(
        varmap.all_vars(),
        ParamsAdamW {
            lr: 5e-4,          // 降低学习率
            weight_decay: 1e-4, // 添加权重衰减
            beta1: 0.9,        // 明确设置动量参数
            beta2: 0.999,
            eps: 1e-8,
            ..Default::default()
        }
    )?;

    println!("网络结构:");
    println!("  输入层: 784 节点 (28x28图像)");
    println!("  隐藏层1: 128 节点 (ReLU激活)");
    println!("  隐藏层2: 64 节点 (ReLU激活)");
    println!("  输出层: 10 节点 (Softmax激活)");
    println!("  学习率: 0.01\n");

    println!("开始训练...");
    const BATCH_SIZE: usize = 120;

    let image_batches = train_images.chunk(BATCH_SIZE, 0)?; // 60000/BATCH_SIZE
    let label_batches = train_labels.chunk(BATCH_SIZE, 0)?; // 60000/BATCH_SIZE
    // let epoch_test_images = test_images.narrow(0, 0, 100)?;
    // let epoch_test_label = test_labels.narrow(0, 0, 100)?;
    for epoch in 0..100 {
        let mut epoch_loss = 0.0;
        for(batch_idx, (images, lables)) in image_batches.iter().zip(&label_batches).enumerate() {
            let logits = model.forward(&images)?;
            let loss = loss::cross_entropy(&logits, &lables)?;
            optimizer.backward_step(&loss)?;
            epoch_loss += loss.to_vec0::<f32>()?;
            if batch_idx % 10 == 0 {
                println!("Epoch {}, Batch {}/{}, Loss: {:.4}", epoch, batch_idx, BATCH_SIZE, loss.to_vec0::<f32>()?);
            }
        }
        let avg_loss = epoch_loss / BATCH_SIZE as f32;
        println!("Epoch {} 完成, 平均损失: {:.4}", epoch, avg_loss);
        let predictions = model.predict(&test_images)?;
        let epoch_test_label = test_labels.to_vec1::<u32>()?;
        let accuracy = compute_agreement_accuracy(&predictions, &epoch_test_label);
        println!("准确率: {}%", accuracy);
    }

    Ok(())
}