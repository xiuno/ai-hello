use candle_core::{DType, Device, Result, Tensor};

// 启发了 ChatGPT 位置编码的核心方法!!! SinCos 位置编码法! 再怎么强调它的重要性都不为过, 简直天才的想法!
/// ![架构图](https://raw.githubusercontent.com/xiuno/static/refs/heads/main/images/ai/sin-cos.png)
///  https://www.bilibili.com/video/BV1u34y1H7cd
pub mod sin_cos {
    use super::*;
    pub fn encoding(seq_len: usize, dims: usize, device: &Device, dtype: DType) -> Result<Tensor> {
        if dims % 2 != 0 {
            return Err(candle_core::Error::Msg(
                "dims must be even for sin_cos encoding".into(),
            ));
        }
        let half_dims = dims / 2;

        // cos(ωi)cos(ωj)+sin(ωi)sin(ωj)=cos(ω(i−j))
        // 相对位置需要相减
        // 创建位置索引 [seq_len, 1],
        // 如: [0., 1., 2., 3., 4., ... seq_len-1]
        // 升维以后: [[0], [1], [2]... [seq_len-1]]
        let positions = Tensor::arange(0f32, seq_len as f32, device)?
            .to_dtype(dtype)?
            .unsqueeze(1)?;

        // 创建频率项 [1, half_dims]
        let div_term = {
            // 注意这里是 half_dims, 最后要交错拼接的!
            let i = Tensor::arange(0f32, half_dims as f32, device)?.to_dtype(dtype)?;

            // 计算 1 / (10000^(2i/dims))
            // 避免标量张量，直接创建向量
            let i_vec = i.to_vec1::<f32>()?;
            let power_vec: Vec<f32> = i_vec.iter().map(|&x| 2.0 * x / dims as f32).collect();

            let base_vec: Vec<f32> = power_vec.iter().map(|&p| 10000f32.powf(p)).collect();

            // 计算 1/denominator, 最后和 positions 张量相乘, 这样速度快. (AI 还是厉害, 工程性很好!)
            let ones_vec: Vec<f32> = base_vec
                .iter()
                .map(|&d| 1.0 / d) // 这里是 1 / d, 不是 pos/d, 张量相乘更快!
                .collect();
            let div_term = Tensor::new(ones_vec.as_slice(), device)?.to_dtype(dtype)?;

            div_term.unsqueeze(0)?
        };

        // 计算角度 [seq_len, half_dims]
        let angles = positions.broadcast_mul(&div_term)?;

        // 计算 sin 和 cos
        let sin_vals = angles.sin()?; // 注意, 这里形状是 [seq_len, half_dims]
        let cos_vals = angles.cos()?; // 注意, 这里形状是 [seq_len, half_dims]

        // 位置权值设置的相当巧妙, 相对位置表示为 (i-j), 所以 cos(i-j) = cos(i)cos(j) + sin(i)sin(j) .
        // ![](https://raw.githubusercontent.com/xiuno/static/refs/heads/main/images/ai/sin-cos-pos.png)
        // 高效的操作方法
        // 交错排列：[cos0, sin0, cos1, sin1, ...]
        // 在最后一个维度上堆叠得到形状 [seq_len, half_dims, 2]
        /*
           cos_vals = [[c00, c01], [c10, c11], [c20, c21]]
           sin_vals = [[s00, s01], [s10, s11], [s20, s21]]
           Tensor::stack(&[cos_vals, sin_vals], 2) 堆叠之后:
           stacked[i][j][0] = cos_vals[i][j]
           stacked[i][j][1] = sin_vals[i][j]
           stacked = [
             [ [c00, s00], [c01, s01] ],
             [ [c10, s10], [c11, s11] ],
             [ [c20, s20], [c21, s21] ]
           ]
        */
        let stacked = Tensor::stack(&[cos_vals, sin_vals], 2)?; // [seq_len, half_dims, 2]
        let pe = stacked.reshape((seq_len, dims))?;

        Ok(pe)
    }

    mod test {
        use super::*;

        /// Helper: 模拟 PyTorch allclose
        // fn allclose(a: &Tensor, b: &Tensor, atol: f32) -> Result<bool> {
        //     let diff = a.sub(b)?.abs()?;
        //     let max_diff = diff.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        //     Ok(max_diff <= atol)
        // }

        #[test]
        fn test_sin_cos_encoding() -> Result<()> {
            let device = Device::Cpu;
            let seq_len = 16;
            let dims = 64;
            let pe = sin_cos::encoding(seq_len, dims, &device, DType::F32)?;
            assert_eq!(pe.dims(), &[seq_len, dims]);

            // 验证第一个位置的编码
            //let first_pos = pe.get(12)?;
            //println!("First position encoding: {:?}", first_pos.to_vec1::<f32>()?);

            // 验证形状
            //println!("SinCos Position Encoding shape: {:?}", pe.dims());
            Ok(())
        }

        #[test]
        fn test_sin_cos_properties() -> Result<()> {
            let device = Device::Cpu;
            let seq_len = 10;
            let dims = 16;
            let pe = sin_cos::encoding(seq_len, dims, &device, DType::F32)?;

            // 验证每个位置向量的范数
            for i in 0..std::cmp::min(5, seq_len) {
                // 只打印前5个
                let pos_vec = pe.get(i)?;
                let norm = pos_vec.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
                println!("Position {} norm: {:.4}", i, norm);
            }

            Ok(())
        }
    }
}

// SinCos 位置编码的改进版本, 旋转编码, 更加天才的想法!
pub mod rope {
    use candle_core::{DType, Device, Result, Tensor};

    /// 旋转位置编码 (RoPE) 实现
    pub fn encoding(x: &Tensor, start_pos: usize, theta: Option<f64>) -> Result<Tensor> {
        let theta = theta.unwrap_or(10000.0);
        let device = x.device();
        let dtype = x.dtype();

        // 获取输入张量的形状 [batch_size, seq_len, num_heads, head_dim]
        let shape = x.dims();
        let seq_len = shape[1];
        let head_dim = shape[shape.len() - 1];

        // 确保 head_dim 是偶数（RoPE 要求）
        if head_dim % 2 != 0 {
            return Err(candle_core::Error::Msg(
                "Head dimension must be even for RoPE".to_string(),
            ));
        }

        // 创建频率张量
        let freqs = create_frequencies(head_dim, theta, device, dtype)?;

        // 创建位置张量
        let positions = create_positions(start_pos, seq_len, device, dtype)?;

        // 计算角度：positions * freqs
        let angles = compute_angles(&positions, &freqs)?;

        // 应用旋转
        apply_rotary_embedding(x, &angles)
    }

    /// 创建频率张量
    fn create_frequencies(
        head_dim: usize,
        theta: f64,
        device: &Device,
        dtype: DType,
    ) -> Result<Tensor> {
        let half_dim = head_dim / 2;

        // 创建维度索引 [0, 1, 2, ..., half_dim-1]
        let dim_indices: Vec<f64> = (0..half_dim).map(|i| i as f64).collect();

        let dim_tensor = Tensor::from_vec(dim_indices, (half_dim,), device)?.to_dtype(dtype)?;

        // 创建常数张量，确保有正确的形状
        let two_tensor = Tensor::from_vec(vec![2.0f64], (1,), device)?
            .to_dtype(dtype)?
            .broadcast_as((half_dim,))?;

        let head_dim_tensor = Tensor::from_vec(vec![head_dim as f64], (1,), device)?
            .to_dtype(dtype)?
            .broadcast_as((half_dim,))?;

        let theta_tensor = Tensor::from_vec(vec![theta], (1,), device)?
            .to_dtype(dtype)?
            .broadcast_as((half_dim,))?;

        // 计算频率：1.0 / (theta ^ (2 * i / head_dim))
        let exponent = dim_tensor.mul(&two_tensor)?.div(&head_dim_tensor)?;

        let freqs = theta_tensor.pow(&exponent)?.recip()?;

        Ok(freqs)
    }

    /// 创建位置张量
    fn create_positions(
        start_pos: usize,
        seq_len: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Tensor> {
        let positions: Vec<f64> = (start_pos..start_pos + seq_len).map(|i| i as f64).collect();

        Tensor::from_vec(positions, (seq_len,), device)?.to_dtype(dtype)
    }

    /// 计算角度矩阵
    fn compute_angles(positions: &Tensor, freqs: &Tensor) -> Result<Tensor> {
        // positions: [seq_len], freqs: [head_dim/2]
        // 结果: [seq_len, head_dim/2]
        let positions = positions.unsqueeze(1)?; // [seq_len, 1]
        let freqs = freqs.unsqueeze(0)?; // [1, head_dim/2]

        positions.broadcast_mul(&freqs)
    }

    /// 应用旋转嵌入
    fn apply_rotary_embedding(x: &Tensor, angles: &Tensor) -> Result<Tensor> {
        let shape = x.dims();
        let batch_size = shape[0];
        let seq_len = shape[1];
        let num_heads = shape[2];
        let head_dim = shape[3];
        let half_dim = head_dim / 2;

        // 计算 cos 和 sin
        let cos_angles = angles.cos()?;
        let sin_angles = angles.sin()?;

        // 扩展维度以匹配输入张量
        let cos_angles = cos_angles
            .unsqueeze(0)? // [1, seq_len, head_dim/2]
            .unsqueeze(2)? // [1, seq_len, 1, head_dim/2]
            .broadcast_as((batch_size, seq_len, num_heads, half_dim))?;

        let sin_angles = sin_angles
            .unsqueeze(0)?
            .unsqueeze(2)?
            .broadcast_as((batch_size, seq_len, num_heads, half_dim))?;

        // 分离 x 的前半部分和后半部分
        let x1 = x.narrow(3, 0, half_dim)?; // 前半部分
        let x2 = x.narrow(3, half_dim, half_dim)?; // 后半部分

        // 应用旋转公式
        // rotated_x1 = x1 * cos - x2 * sin
        // rotated_x2 = x1 * sin + x2 * cos
        let rotated_x1 = x1.mul(&cos_angles)?.sub(&x2.mul(&sin_angles)?)?;
        let rotated_x2 = x1.mul(&sin_angles)?.add(&x2.mul(&cos_angles)?)?;

        // 连接结果
        Tensor::cat(&[rotated_x1, rotated_x2], 3)
    }

    /// 预计算旋转矩阵（可选的优化版本）
    pub fn precompute_rope_cache(
        max_seq_len: usize,
        head_dim: usize,
        theta: f64,
        device: &Device,
        dtype: DType,
    ) -> Result<(Tensor, Tensor)> {
        let freqs = create_frequencies(head_dim, theta, device, dtype)?;
        let positions = create_positions(0, max_seq_len, device, dtype)?;
        let angles = compute_angles(&positions, &freqs)?;

        let cos_cache = angles.cos()?;
        let sin_cache = angles.sin()?;

        Ok((cos_cache, sin_cache))
    }

    /// 使用预计算缓存的 RoPE 编码
    pub fn encoding_with_cache(
        x: &Tensor,
        start_pos: usize,
        cos_cache: &Tensor,
        sin_cache: &Tensor,
    ) -> Result<Tensor> {
        let shape = x.dims();
        let batch_size = shape[0];
        let seq_len = shape[1];
        let num_heads = shape[2];
        let head_dim = shape[3];
        let half_dim = head_dim / 2;

        // 从缓存中提取对应位置的 cos 和 sin
        let cos_angles = cos_cache
            .narrow(0, start_pos, seq_len)?
            .unsqueeze(0)?
            .unsqueeze(2)?
            .broadcast_as((batch_size, seq_len, num_heads, half_dim))?;

        let sin_angles = sin_cache
            .narrow(0, start_pos, seq_len)?
            .unsqueeze(0)?
            .unsqueeze(2)?
            .broadcast_as((batch_size, seq_len, num_heads, half_dim))?;

        // 分离 x 的前半部分和后半部分
        let x1 = x.narrow(3, 0, half_dim)?;
        let x2 = x.narrow(3, half_dim, half_dim)?;

        // 应用旋转
        let rotated_x1 = x1.mul(&cos_angles)?.sub(&x2.mul(&sin_angles)?)?;
        let rotated_x2 = x1.mul(&sin_angles)?.add(&x2.mul(&cos_angles)?)?;

        Tensor::cat(&[rotated_x1, rotated_x2], 3)
    }

    // 简化版本的频率创建（更直接的方法）
    #[warn(dead_code)]
    pub fn create_frequencies_simple(
        head_dim: usize,
        theta: f64,
        device: &Device,
        dtype: DType,
    ) -> Result<Tensor> {
        let half_dim = head_dim / 2;
        let mut freqs = Vec::with_capacity(half_dim);

        for i in 0..half_dim {
            let freq = 1.0 / theta.powf(2.0 * i as f64 / head_dim as f64);
            freqs.push(freq);
        }

        Tensor::from_vec(freqs, (half_dim,), device)?.to_dtype(dtype)
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use candle_core::{DType, Device};

        #[test]
        fn test_rope_encoding() -> Result<()> {
            let device = Device::Cpu;
            let dtype = DType::F32;

            // 创建测试输入 [batch_size=1, seq_len=4, num_heads=2, head_dim=8]
            let x = Tensor::randn(0f32, 1f32, (1, 4, 2, 8), &device)?.to_dtype(dtype)?;

            //println!("Input shape: {:?}", x.dims());

            // 应用 RoPE
            let result = super::encoding(&x, 0, Some(10000.0))?;

            // 检查输出形状
            assert_eq!(result.dims(), x.dims());
            // println!("Output shape: {:?}", result);
            // println!("Output shape: {:?}", result.dims());
            //
            // println!("RoPE encoding test passed!");
            Ok(())
        }

        #[test]
        fn test_rope_with_cache() -> Result<()> {
            let device = Device::Cpu;
            let dtype = DType::F32;

            // 预计算缓存
            let (cos_cache, sin_cache) =
                super::precompute_rope_cache(100, 8, 10000.0, &device, dtype)?;

            // println!(
            //     "Cache shapes - cos: {:?}, sin: {:?}",
            //     cos_cache.dims(),
            //     sin_cache.dims()
            // );

            // 创建测试输入
            let x = Tensor::randn(0f32, 1f32, (1, 4, 2, 8), &device)?.to_dtype(dtype)?;

            // 使用缓存应用 RoPE
            let result = super::encoding_with_cache(&x, 0, &cos_cache, &sin_cache)?;

            assert_eq!(result.dims(), x.dims());

            //println!("RoPE with cache test passed!");
            Ok(())
        }

        // #[test]
        // fn test_frequency_creation() -> Result<()> {
        //     let device = Device::Cpu;
        //     let dtype = DType::F32;
        //
        //     //et freqs = super::create_frequencies_simple(8, 10000.0, &device, dtype)?;
        //     //println!("Frequencies shape: {:?}", freqs.dims());
        //     //println!("Frequencies: {:?}", freqs.to_vec1::<f32>()?);
        //
        //     Ok(())
        // }
    }
}
