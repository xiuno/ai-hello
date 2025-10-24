use rand::Rng;

// ============ 神经网络结构 ============
struct NeuralNetwork {
    // 权重矩阵
    w1: Vec<Vec<f64>>, // 输入层 -> 隐藏层 (64 x 128)
    w2: Vec<Vec<f64>>, // 隐藏层 -> 输出层 (128 x 10)

    // 偏置
    b1: Vec<f64>, // 隐藏层偏置 (128)
    b2: Vec<f64>, // 输出层偏置 (10)

    learning_rate: f64,
}

impl NeuralNetwork {
    fn new(input_size: usize, hidden_size: usize, output_size: usize, learning_rate: f64) -> Self {
        use rand::thread_rng;
        use rand_distr::{Normal, Distribution};

        let mut rng = thread_rng();

        // He Normal initialization: std = sqrt(2 / fan_in)
        // Optimal for ReLU activation functions
        let w1_std = (2.0 / input_size as f64).sqrt();
        let w2_std = (2.0 / hidden_size as f64).sqrt();
        let w1_dist = Normal::new(0.0, w1_std).unwrap();
        let w2_dist = Normal::new(0.0, w2_std).unwrap();

        // w1: [hidden_size × input_size] for efficient computation
        let w1: Vec<Vec<f64>> = (0..input_size)
            .map(|_| {
                (0..hidden_size)
                    .map(|_| w1_dist.sample(&mut rng))  // 注意这里
                    .collect()
            })
            .collect();

        // w2: [output_size × hidden_size]
        let w2: Vec<Vec<f64>> = (0..hidden_size)
            .map(|_| {
                (0..output_size)
                    .map(|_| w2_dist.sample(&mut rng))  // 注意这里
                    .collect()
            })
            .collect();

        NeuralNetwork {
            w1,
            w2,
            b1: vec![0.0; hidden_size],
            b2: vec![0.0; output_size],
            learning_rate,
        }
    }

    fn softmax(z: &[f64]) -> Vec<f64> {
        let max_z = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        // 对 z2 里的每个元素求 e^(zi-max(z))
        let exp_z: Vec<f64> = z.iter().map(|&x| (x - max_z).exp()).collect();
        // 求和, 作为 softmax 计算时的分母
        let sum_exp: f64 = exp_z.iter().sum();
        // 将 z2 通过 softmax(改进版, 指数做了最大化处理) 归一化, 得到 a2
        let a: Vec<f64> = exp_z.iter().map(|&x| x / sum_exp).collect();
        a
    }

    // 前向传播, 返回 (z1, a1, z2, a2)
    fn forward(&self, input: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        // 隐藏层：z1 = W1 * input + b1
        let mut z1 = vec![0.0; self.b1.len()];
        for i in 0..self.b1.len() {
            z1[i] = self.b1[i];
            for j in 0..input.len() {
                z1[i] += self.w1[j][i] * &input[j];
            }
        }

        // ReLU激活
        let a1: Vec<f64> = z1.iter().map(|&x| x.max(0.0)).collect();

        // 输出层：z2 = W2 * a1 + b2
        let mut z2 = vec![0.0; self.b2.len()];
        for i in 0..self.b2.len() {
            z2[i] = self.b2[i];
            for j in 0..a1.len() {
                z2[i] += self.w2[j][i] * a1[j];
            }
        }
        // Softmax 激活
        let a2 = Self::softmax(&z2);
        (z1, a1, z2, a2)
    }

    // 反向传播
    // target: 第几个是正确答案
    // 反向传播
    // target: 正确答案的索引 (0-9), 在数学里表示为 k. 对应 onehot 所在元素下标.
    fn backward(&mut self, input: &[f64], target: usize, z1: &[f64], a1: &[f64], a2: &[f64]) {
        // ============ 输出层梯度 ============
        // Softmax + 交叉熵的组合求导：dL/dz2 = a2 - one_hot(target)
        let mut dz2 = a2.to_vec();
        dz2[target] -= 1.0;

        // ============ 更新 W2 和 b2 ============
        // dL/dw2[j][k] = dz2[k] * a1[j]
        // w2[j][k]: 隐藏层j -> 输出层k
        for j in 0..self.w2.len() {  // 遍历隐藏层
            for k in 0..self.w2[j].len() {  // 遍历输出层
                let grad = a1[j] * dz2[k];
                self.w2[j][k] -= self.learning_rate * grad;
            }
        }

        // dL/db2[k] = dz2[k]
        for k in 0..self.b2.len() {
            self.b2[k] -= self.learning_rate * dz2[k];
        }

        // ============ 隐藏层梯度 ============
        // dL/da1[j] = Σ_k (w2[j][k] * dz2[k])
        // dL/dz1[j] = dL/da1[j] * ReLU'(z1[j])
        let mut dz1 = vec![0.0; a1.len()];
        for j in 0..a1.len() {  // 遍历隐藏层
            let mut da1_j = 0.0;
            for k in 0..dz2.len() {  // 遍历输出层
                da1_j += self.w2[j][k] * dz2[k];
            }
            // ReLU 的导数: z > 0 时为 1, z <= 0 时为 0
            dz1[j] = if z1[j] > 0.0 { da1_j } else { 0.0 };
        }

        // ============ 更新 W1 和 b1 ============
        // dL/dw1[i][j] = dz1[j] * input[i]
        // w1[i][j]: 输入层i -> 隐藏层j
        for i in 0..self.w1.len() {  // 遍历输入层
            for j in 0..self.w1[i].len() {  // 遍历隐藏层
                let grad = input[i] * dz1[j];
                self.w1[i][j] -= self.learning_rate * grad;
            }
        }

        // dL/db1[j] = dz1[j]
        for j in 0..self.b1.len() {
            self.b1[j] -= self.learning_rate * dz1[j];
        }
    }

    fn cross_entropy_loss(x: f64) -> f64 {
        -x.ln()
    }

    // 训练一个epoch
    fn train_epoch(&mut self, data: &[(Vec<f64>, usize)]) -> f64 {
        let mut total_loss = 0.0;

        for (input, target) in data {
            let (z1, a1, _z2, a2) = self.forward(input);

            // 交叉熵损失 L = -log(a2[k])
            // let loss = -a2[*target].ln();
            let loss = Self::cross_entropy_loss(a2[*target]);

            total_loss += loss;
            // println!("loss: {}", -a2[*target]);

            // 反向传播
            self.backward(input, *target, &z1, &a1, &a2);
        }

        total_loss / data.len() as f64
    }

    // 预测
    fn predict(&self, input: &[f64]) -> usize {
        let (_, _, _, a2) = self.forward(input);
        a2.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                if a.is_nan() && b.is_nan() {
                    std::cmp::Ordering::Equal
                } else if a.is_nan() {
                    std::cmp::Ordering::Less
                } else if b.is_nan() {
                    std::cmp::Ordering::Greater
                } else {
                    a.partial_cmp(b).unwrap()
                }
            })
            .unwrap()
            .0
        //
        // a2.iter()
        //     .enumerate()
        //     .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        //     .unwrap()
        //     .0
    }
}

// ============ 模拟数据生成 ============
fn generate_digit_samples() -> Vec<(Vec<f64>, usize)> {
    // 8x8 简化图像（64个像素）
    // 1 代表黑色，0 代表白色
    let patterns = vec![
        // 数字 0
        vec![
            0,1,1,1,1,1,1,0,
            1,1,0,0,0,0,1,1,
            1,0,0,0,0,0,0,1,
            1,0,0,0,0,0,0,1,
            1,0,0,0,0,0,0,1,
            1,0,0,0,0,0,0,1,
            1,1,0,0,0,0,1,1,
            0,1,1,1,1,1,1,0,
        ],
        // 数字 1
        vec![
            0,0,0,1,1,0,0,0,
            0,0,1,1,1,0,0,0,
            0,1,1,1,1,0,0,0,
            0,0,0,1,1,0,0,0,
            0,0,0,1,1,0,0,0,
            0,0,0,1,1,0,0,0,
            0,0,0,1,1,0,0,0,
            1,1,1,1,1,1,1,1,
        ],
        // 数字 2
        vec![
            0,1,1,1,1,1,1,0,
            1,1,0,0,0,0,1,1,
            0,0,0,0,0,0,1,1,
            0,0,0,0,0,1,1,0,
            0,0,0,1,1,1,0,0,
            0,1,1,1,0,0,0,0,
            1,1,0,0,0,0,0,0,
            1,1,1,1,1,1,1,1,
        ],
        // 数字 3
        vec![
            1,1,1,1,1,1,1,0,
            0,0,0,0,0,0,1,1,
            0,0,0,0,0,1,1,0,
            0,0,1,1,1,1,0,0,
            0,0,0,0,0,1,1,0,
            0,0,0,0,0,0,1,1,
            1,1,0,0,0,0,1,1,
            0,1,1,1,1,1,1,0,
        ],
        // 数字 4
        vec![
            0,0,0,0,1,1,0,0,
            0,0,0,1,1,1,0,0,
            0,0,1,1,1,1,0,0,
            0,1,1,0,1,1,0,0,
            1,1,0,0,1,1,0,0,
            1,1,1,1,1,1,1,1,
            0,0,0,0,1,1,0,0,
            0,0,0,0,1,1,0,0,
        ],
        // 数字 5
        vec![
            1,1,1,1,1,1,1,1,
            1,1,0,0,0,0,0,0,
            1,1,0,0,0,0,0,0,
            1,1,1,1,1,1,1,0,
            0,0,0,0,0,0,1,1,
            0,0,0,0,0,0,1,1,
            1,1,0,0,0,0,1,1,
            0,1,1,1,1,1,1,0,
        ],
        // 数字 6
        vec![
            0,0,1,1,1,1,1,0,
            0,1,1,0,0,0,0,0,
            1,1,0,0,0,0,0,0,
            1,1,1,1,1,1,1,0,
            1,1,0,0,0,0,1,1,
            1,1,0,0,0,0,1,1,
            1,1,0,0,0,0,1,1,
            0,1,1,1,1,1,1,0,
        ],
        // 数字 7
        vec![
            1,1,1,1,1,1,1,1,
            0,0,0,0,0,0,1,1,
            0,0,0,0,0,1,1,0,
            0,0,0,0,1,1,0,0,
            0,0,0,1,1,0,0,0,
            0,0,1,1,0,0,0,0,
            0,0,1,1,0,0,0,0,
            0,0,1,1,0,0,0,0,
        ],
        // 数字 8
        vec![
            0,1,1,1,1,1,1,0,
            1,1,0,0,0,0,1,1,
            1,1,0,0,0,0,1,1,
            0,1,1,1,1,1,1,0,
            1,1,0,0,0,0,1,1,
            1,1,0,0,0,0,1,1,
            1,1,0,0,0,0,1,1,
            0,1,1,1,1,1,1,0,
        ],
        // 数字 9
        vec![
            0,1,1,1,1,1,1,0,
            1,1,0,0,0,0,1,1,
            1,1,0,0,0,0,1,1,
            0,1,1,1,1,1,1,1,
            0,0,0,0,0,0,1,1,
            0,0,0,0,0,0,1,1,
            0,0,0,0,0,1,1,0,
            0,1,1,1,1,1,0,0,
        ],
    ];

    // 转换为 f64 并归一化
    patterns
        .into_iter()
        .enumerate()
        .map(|(label, pattern)| {
            let input: Vec<f64> = pattern.iter().map(|&x| x as f64).collect();
            (input, label)
        })
        .collect()
}

// ============ 主函数 ============
fn main() {
    println!("=== 手写数字识别神经网络 ===\n");

    // 生成训练数据
    let training_data = generate_digit_samples();

    // 创建神经网络
    let mut nn = NeuralNetwork::new(64, 128, 10, 0.01);

    println!("网络结构:");
    println!("  输入层: 64 节点 (8x8图像)");
    println!("  隐藏层: 128 节点 (ReLU激活)");
    println!("  输出层: 10 节点 (Softmax激活)");
    println!("  学习率: 0.1\n");

    // 训练
    println!("开始训练...");
    for epoch in 0..50 {
        let loss = nn.train_epoch(&training_data);

        if epoch % 10 == 0 {
            // 计算准确率
            let mut correct = 0;
            for (input, target) in &training_data {
                if nn.predict(input) == *target {
                    correct += 1;
                }
            }
            let accuracy = correct as f64 / training_data.len() as f64 * 100.0;

            println!("Epoch {:4} | Loss: {:.4} | Accuracy: {:.1}%",
                     epoch, loss, accuracy);
        }
    }

    // 测试
    println!("\n=== 测试结果 ===");
    for (input, target) in &training_data {
        let prediction = nn.predict(input);
        let (_, _, _, a2) = nn.forward(input);

        println!("真实标签: {} | 预测: {} | 置信度: {:.2}%",
                 target, prediction, a2[prediction] * 100.0);
    }

    println!("\n训练完成！");


    // 测试单个:
    let input: Vec<f64> = vec![
        0,0,1,1,1,1,0,0,
        0,0,0,0,0,0,0,0,
        0,0,0,0,0,1,1,0,
        0,0,0,0,1,1,0,0,
        0,0,0,1,1,0,0,0,
        0,0,0,1,0,0,0,0,
        0,0,1,1,0,0,0,0,
        0,0,1,1,0,0,0,0,
    ].iter().map(|&x| x as f64).collect();

    let prediction = nn.predict(&input);
    println!("prediction: {}", prediction);
    let (_, _, _, a2) = nn.forward(&input);
    println!("a2: {:?}", &a2);
    println!("置信度: {:.2}%", a2[prediction] * 100.0);

}