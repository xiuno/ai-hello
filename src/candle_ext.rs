use candle_core::{Tensor, IndexOp, Result};
pub trait TensorExt {
    fn std_all(&self, dim: Option<usize>, keepdim: bool) -> Result<Tensor>;
    fn print(&self, name: &str, stats: bool) -> anyhow::Result<()>;
}
/// let device = Device::Cpu;
///
///
///     // 创建测试张量
///     let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
///     let tensor = Tensor::from_vec(data, (2, 3), &device)?;
///
///     // 计算整个张量的标准差
///     let std_all = std_dev(&tensor, None, false)?;
///     println!("Standard deviation (all): {:?}", std_all);
///
///     // 计算沿着维度0的标准差
///     let std_dim0 = std_dev(&tensor, Some(0), false)?;
///     println!("Standard deviation (dim 0): {:?}", std_dim0);
///
//     // 计算沿着维度1的标准差
//     let std_dim1 = std_dev(&tensor, Some(1), false)?;
//     println!("Standard deviation (dim 1): {:?}", std_dim1);


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
        let elements_to_show = dims[depth].min(2);

        for i in 0..elements_to_show {
            let mut new_indices = current_indices.to_vec();
            new_indices.push(i);
            print_high_dim_sample(tensor, name, &new_indices, dims, max_elements, depth + 1)?;
        }

        if dims[depth] > 2 {
            let indent = "  ".repeat(depth);
            //println!("{}... ({} more elements at this level)", indent, dims[depth] - 2);
            println!("    ...");
        }
    }

    Ok(())
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
        let max_elements = 10;
        let stats_info = if stats {
            let mean_all = self.mean_all()?.to_vec0::<f32>()?;
            let std_all = self.std_all(None, true)?.to_vec0::<f32>()?;
            format!(" (mean: {mean_all}, std: {std_all})")
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
                let display_count = data.len().min(max_elements);
                let display_data: Vec<f32> = data.iter().take(display_count).copied().collect();
                if data.len() > max_elements {
                    println!("{name}[..{}]:{stats_info} {:?}...", display_count, display_data);
                } else {
                    println!("{name}:{stats_info} {:?}", display_data);
                }
            }
            2 => {
                // 2维张量：第一维最多打印3个，最后一维最多打印 max_elements 个
                let dims = shape.dims();
                let rows_to_show = dims[0].min(3);

                println!("{name}{:?}:{stats_info}", dims);
                for i in 0..rows_to_show {
                    let row = self.i(i)?;
                    let row_data = row.to_vec1::<f32>()?;
                    let display_count = row_data.len().min(max_elements);
                    let display_data: Vec<f32> = row_data.iter().take(display_count).copied().collect();

                    if row_data.len() > max_elements {
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
                // 高维张量（>2维）：前面几维最多打印2个，最后一维最多打印 max_elements 个
                let dims = shape.dims();
                println!("{name}{:?}:{stats_info}", dims);

                // 递归打印高维张量的前2个元素
                print_high_dim_sample(self, name, &[], dims, max_elements, 0)?;
            }
        }

        Ok(())
    }


}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device};

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