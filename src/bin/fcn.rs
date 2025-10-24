use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{loss, ops, Linear, Optimizer, VarBuilder, VarMap};

/*
curl -LO https://raw.githubusercontent.com/fgnt/mnist/master/train-images-idx3-ubyte.gz
curl -LO https://raw.githubusercontent.com/fgnt/mnist/master/train-labels-idx1-ubyte.gz
curl -LO https://raw.githubusercontent.com/fgnt/mnist/master/t10k-images-idx3-ubyte.gz
curl -LO https://raw.githubusercontent.com/fgnt/mnist/master/t10k-labels-idx1-ubyte.gz
 */
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

    // let train_images = Tensor::from_vec(train_images, (60000, 28, 28), dev)?; // CNN
    let train_images = Tensor::from_vec(train_images, (60000, 784), device)?; // 全连接
    let train_labels = Tensor::from_vec(train_labels, (60000,), device)?;

    let test_images = Tensor::from_vec(test_images, (10000, 784), device)?;
    let test_labels = Tensor::from_vec(test_labels, (10000,), device)?;

    Ok((train_images, train_labels, test_images, test_labels))
}

struct NerualNetwork {
    layer1: Linear, // 输入层 -> 隐藏层 (784 -> 128)
    layer2: Linear, // 隐藏层 -> 隐藏层 (128 -> 64)
    layer3: Linear, // 隐藏层 -> 输出层 (64 -> 10)
}

impl NerualNetwork {
    fn new(vs: VarBuilder) -> Result<Self> {
        let layer1 = candle_nn::linear(784, 128, vs.pp("layer1"))?;
        let layer2 = candle_nn::linear(128, 64, vs.pp("layer2"))?;
        let layer3 = candle_nn::linear(64, 10, vs.pp("layer3"))?;
        Ok(Self { layer1, layer2, layer3 })
    }
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let x = self.layer1.forward(&input)?;
        let x = x.relu()?;
        let x = self.layer2.forward(&x)?;
        let x = x.relu()?;
        let x = self.layer3.forward(&x)?;
        Ok(x)
    }
    fn predict(&self, input: &Tensor) -> Result<Vec<u32>> {
        // input: [B, 784]
        let logits = self.forward(input)?; // logits: [B, 10]
        let probs = ops::softmax(&logits, 1)?; // probs: [B, 10]
        let pred = probs.argmax(1)?; // pred: [B]
        pred.to_vec1()
    }

}

fn compute_agreement_accuracy(predicted_labels: &[u32], reference_labels: &[u32]) -> f32 {
    assert_eq!(
        predicted_labels.len(),
        reference_labels.len(),
        "Label vectors must have the same length"
    );
    let matching_count = predicted_labels
        .iter()
        .zip(reference_labels)
        .filter(|(&pred, &label)| pred == label)
        .count();

    (matching_count as f32 / reference_labels.len() as f32) * 100.0
}

fn main() -> Result<()> {
    let device = &Device::Cpu;
    //let device = &Device::new_metal(0)?;

    // 加载 MNIST 数据
    let (train_images, train_labels, test_images, test_labels) = load_mnist_tensors(device)?;

    println!("训练集图片形状: {:?}", train_images.shape());
    println!("训练集标签形状: {:?}", train_labels.shape());
    println!("测试集图片形状: {:?}", test_images.shape());
    println!("测试集标签形状: {:?}", test_labels.shape());

    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let model = NerualNetwork::new(vs)?;
    let mut optimizer = candle_nn::SGD::new(varmap.all_vars(), 0.01)?;

    println!("网络结构:");
    println!("  输入层: 784 节点 (28x28图像)");
    println!("  隐藏层1: 128 节点 (ReLU激活)");
    println!("  隐藏层2: 64 节点 (ReLU激活)");
    println!("  输出层: 10 节点 (Softmax激活)");
    println!("  学习率: 0.01\n");

    println!("开始训练...");
    const BATCH_SIZE: usize = 60;

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