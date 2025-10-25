use candle_core::{Tensor, IndexOp, Result};
const MAX_ROWS: usize = 3;
const MAX_COLS: usize = 8;

pub trait TensorExt {
    // 标准方差
    fn std_all(&self, dim: Option<usize>, keepdim: bool) -> Result<Tensor>;
    // 打印向量维度, 数据, 平均方差, 标准方差, 余弦相似度
    fn print(&self, name: &str, stats: bool) -> anyhow::Result<()>;
    // ✨ 新增：沿指定维度标准化（Z-score）
    fn standardize(&self, dim: usize) -> Result<(Tensor, Tensor, Tensor)>;

    // ✨ 新增：用给定 mean/std 还原
    fn unstandardize(&self, mean: &Tensor, std: &Tensor) -> Result<Tensor>;
}

// 辅助函数：递归打印高维张量
fn print_high_dim_sample(
    tensor: &Tensor,
    name: &str,
    current_indices: &[usize],
    dims: &[usize],
    max_elements: usize,
    depth: usize,
) -> anyhow::Result<()> {
    if depth == dims.len() - 1 {
        // 到达最后一维，打印数据
        let mut indices = current_indices.to_vec();
        indices.push(0); // 临时索引，会被替换

        // 构建索引访问
        let slice = match current_indices.len() {
            0 => tensor.clone(),
            1 => tensor.i(current_indices[0])?,
            2 => tensor.i((current_indices[0], current_indices[1]))?,
            3 => tensor.i((current_indices[0], current_indices[1], current_indices[2]))?,
            4 => tensor.i((current_indices[0], current_indices[1], current_indices[2], current_indices[3]))?,
            _ => return Err(anyhow::anyhow!("Unsupported tensor rank > 5")),
        };

        let data = slice.to_vec1::<f32>()?;
        let display_count = data.len().min(max_elements);
        let display_data: Vec<f32> = data.iter().take(display_count).copied().collect();

        let indent = "  ".repeat(depth);
        let index_str = current_indices.iter()
            .map(|i| format!("[{}]", i))
            .collect::<String>();

        if data.len() > max_elements {
            println!("{}{}[..{}]: {:?}...", indent, index_str, display_count, display_data);
        } else {
            println!("{}{}: {:?}", indent, index_str, display_data);
        }
    } else {
        // 还没到最后一维，继续递归
        let elements_to_show = dims[depth].min(MAX_ROWS);

        for i in 0..elements_to_show {
            let mut new_indices = current_indices.to_vec();
            new_indices.push(i);
            print_high_dim_sample(tensor, name, &new_indices, dims, max_elements, depth + 1)?;
        }

        if dims[depth] > 2 {
            let indent = "    ".repeat(depth);
            //println!("{}... ({} more elements at this level)", indent, dims[depth] - 2);
            println!("{indent}...");
        }
    }

    Ok(())
}


fn cosine_sim_str(x: &Tensor) -> anyhow::Result<String> {
    // 处理 [1, M, N] → [M, N]
    let x = if let Ok((1, _, _)) = x.dims3() {
        x.squeeze(0)?
    } else {
        x.clone()
    };

    // 必须是 2D 且至少两行
    let (n_rows, _n_cols) = x.dims2().map_err(|_| anyhow::anyhow!("not 2D"))?;

    if n_rows < 2 {
        return Ok("".to_string());
    }

    let norms = x.sqr()?.sum_keepdim(1)?.sqrt()?;
    let eps = Tensor::full(1e-8f64, norms.shape(), x.device())?.to_dtype(x.dtype())?;
    let normalized = x.broadcast_div(&norms.maximum(&eps)?)?;
    let sim_matrix = normalized.matmul(&normalized.t()?)?;

    let sims = sim_matrix.to_vec2::<f32>()?;
    let mut sum = 0.0f32;
    let mut count = 0;
    for i in 0..n_rows {
        for j in (i + 1)..n_rows {
            sum += sims[i][j];
            count += 1;
        }
    }

    if count == 0 {
        Ok("".to_string())
    } else {
        Ok(format!("{:.4}", sum / count as f32))
    }
}

impl TensorExt for Tensor {
    fn std_all(&self, dim: Option<usize>, keepdim: bool) -> Result<Tensor> {
        // 计算均值
        let mean = if let Some(d) = dim {
            self.mean_keepdim(d)?
        } else {
            self.mean_all()?
        };

        // 计算方差 (x - mean)^2 的均值
        let diff = self.broadcast_sub(&mean)?;
        let squared_diff = diff.sqr()?;
        let variance = if let Some(d) = dim {
            if keepdim {
                squared_diff.mean_keepdim(d)?
            } else {
                squared_diff.mean(d)?
            }
        } else {
            squared_diff.mean_all()?
        };

        // 计算标准差 (方差的平方根)
        variance.sqrt()
    }

    fn print(&self, name: &str, stats: bool) -> anyhow::Result<()> {
        let stats_info = if stats {
            let mean_all = self.mean_all()?.to_vec0::<f32>()?;
            let std_all = self.std_all(None, true)?.to_vec0::<f32>()?;
            let cosim = cosine_sim_str(self)?;
            format!(" (mean: {:.4}, std: {:.4}{})",mean_all, std_all,
                    (!cosim.is_empty()).then(||format!(", cossim: {cosim}")).unwrap_or_default())
        } else {
            "".to_string()
        };
        let shape = self.shape();

        // 只处理 F32 类型
        if self.dtype() != candle_core::DType::F32 {
            println!("{name}: [unsupported dtype for display]");
            return Ok(());
        }

        match shape.rank() {
            0 => {
                // 标量
                let val = self.to_scalar::<f32>()?;
                println!("{name}: {}", val);
            }
            1 => {
                // 1维张量：最多打印 max_elements 个元素
                let data = self.to_vec1::<f32>()?;
                let display_count = data.len().min(MAX_COLS);
                let display_data: Vec<f32> = data.iter().take(display_count).copied().collect();
                if data.len() > MAX_COLS {
                    println!("{name}[..{}]:{stats_info} {:?}...", display_count, display_data);
                } else {
                    println!("{name}:{stats_info} {:?}", display_data);
                }
            }
            2 => {
                // 2维张量：第一维最多打印3个，最后一维最多打印 MAX_COLS 个
                let dims = shape.dims();
                let rows_to_show = dims[0].min(MAX_ROWS);

                println!("{name}{:?}:{stats_info}", dims);
                for i in 0..rows_to_show {
                    let row = self.i(i)?;
                    let row_data = row.to_vec1::<f32>()?;
                    let display_count = row_data.len().min(MAX_COLS);
                    let display_data: Vec<f32> = row_data.iter().take(display_count).copied().collect();

                    if row_data.len() > MAX_COLS {
                        println!("  [{i}][..{}]: {:?}...", display_count, display_data);
                    } else {
                        println!("  [{i}]: {:?}", display_data);
                    }
                }

                if dims[0] > 3 {
                    println!("  ... ({} more rows)", dims[0] - 3);
                }
            }
            _ => {
                // 高维张量（>2维）：前面几维最多打印2个，最后一维最多打印 MAX_COLS 个
                let dims = shape.dims();
                println!("{name}{:?}:{stats_info}", dims);

                // 递归打印高维张量的前2个元素
                print_high_dim_sample(self, name, &[], dims, MAX_COLS, 0)?;
            }
        }

        Ok(())
    }

    // 🔧 标准化：返回 (normalized, mean, std)
    // let (targets_norm, mean, std) = targets.standardize(0)?;
    fn standardize(&self, dim: usize) -> Result<(Tensor, Tensor, Tensor)> {
        let mean = self.mean(dim)?;
        let centered = self.broadcast_sub(&mean)?;
        let var = centered.sqr()?.mean(dim)?;
        let std = var.sqrt()?;

        // 防止除零
        let eps = 1e-8f32;
        let device = std.device();
        let eps_tensor = Tensor::new(eps, device)?.broadcast_as(std.shape())?;
        let std = std.maximum(&eps_tensor)?;

        let normalized = centered.broadcast_div(&std)?;
        Ok((normalized, mean, std))
    }

    // 🔧 还原标准化
    // let final_pred = final_pred_norm.unstandardize(&mean, &std)?;
    fn unstandardize(&self, mean: &Tensor, std: &Tensor) -> Result<Tensor> {
        let scaled = self.broadcast_mul(std)?;
        scaled.broadcast_add(mean)
    }
}


// use candle_core::{Device, DType};
/// 对 [N, D] 张量沿第0维（样本维度）做 Z-score 标准化
/// 返回 (normalized_tensor, mean, std)

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    #[test]
    fn test_print_tensor() {
        let tensor1 = Tensor::zeros((5, 10, 10), DType::F32, &Device::Cpu).unwrap();
        tensor1.print("test", false).unwrap();
    }
    #[test]
    fn test_print_sim() -> anyhow::Result<()> {
        let device = Device::Cpu;

        // 构造两个明显不同的向量：一个偏正，一个偏负
        let row1 = vec![1.0f32, 22.0, 13.0, 5.0];
        let row2 = vec![-1.0, -22.0, -13.0, -5.0];
        // let row2 = vec![-11.0f32, -21.0, -3.0, 400.0];
        let data = [row1, row2].concat(); // [1,2,3,4,-1,-2,-3,-4]

        // 测试 2D: [2, 4]
        let tensor_2d = Tensor::from_vec(data.clone(), (2, 4), &device)?;
        //println!("--- Testing 2D input ---");
        tensor_2d.print("2D", true)?;

        // 测试 3D with batch=1: [1, 2, 4]
        let tensor_3d = Tensor::from_vec(data, (1, 2, 4), &device)?;
        //println!("\n--- Testing [1,2,4] input ---");
        tensor_3d.print("3D", true)?;

        Ok(())
    }
    #[test]
    fn test_std_all() -> Result<()> {
        let device = &Device::Cpu;

        let data = [[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let flat: Vec<f32> = data.iter().flatten().copied().collect();
        let tensor = Tensor::from_slice(&flat, (2, 3), device)?;

        // 1. 全局标准差 → scalar (rank 0)
        let std_global = tensor.std_all(None, false)?;
        let std_global_val = std_global.to_scalar::<f32>()?;
        assert!((std_global_val - 1.7078251).abs() < 1e-5);

        // 2. 沿 dim=1，keepdim=true → shape [2, 1]
        let std_dim1_keepdim = tensor.std_all(Some(1), true)?;
        assert_eq!(std_dim1_keepdim.dims(), &[2, 1]);
        let vals = std_dim1_keepdim.to_vec2::<f32>()?;
        assert!((vals[0][0] - 0.8164966).abs() < 1e-5);
        assert!((vals[1][0] - 0.8164966).abs() < 1e-5);

        // 3. 沿 dim=0，keepdim=false → shape [3]
        let std_dim0_no_keepdim = tensor.std_all(Some(0), false)?;
        assert_eq!(std_dim0_no_keepdim.dims(), &[3]);
        let vals = std_dim0_no_keepdim.to_vec1::<f32>()?;
        for v in vals {
            assert!((v - 1.5).abs() < 1e-5);
        }

        Ok(())
    }
}