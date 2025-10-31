use polars::prelude::*;
use std::fs;
use std::path::{Path, PathBuf};

// 给 Polars 进行扩展, 让它变得好用
pub enum Format {
    Csv,
    Parquet,
    Json,
}
trait PolarsEasy {
    /// 简单保存到文件，自动识别 CSV/Parquet 格式
    fn easy_sink<P: AsRef<Path>>(self, path: P) -> PolarsResult<()>;

    /// 分区保存
    fn easy_sink_partition<P: AsRef<Path>>(
        self,
        dir: P,
        partition_cols: &[&str],
        format: Option<Format>,
    ) -> PolarsResult<()>;
}

impl PolarsEasy for LazyFrame {
    fn easy_sink<P: AsRef<Path>>(self, path: P) -> PolarsResult<()> {
        let path_ref = path.as_ref();
        let ext = path_ref
            .extension()
            .and_then(|e| e.to_str())
            .ok_or_else(|| {
                PolarsError::ComputeError(
                    "Cannot determine file type from path. Use .csv or .parquet".into(),
                )
            })?;
        let sink_target = SinkTarget::Path(PlPath::Local(Arc::from(path_ref)));
        match ext.to_lowercase().as_str() {
            "csv" => {
                let mut opt = CsvWriterOptions::default();
                opt.include_header = true;
                self.sink_csv(sink_target, opt, None, SinkOptions::default())?.collect()?
            }
            "parquet" => self.sink_parquet(
                sink_target,
                ParquetWriteOptions::default(),
                None,
                SinkOptions::default(),
            )?.collect()?,
            "json" => {
                self.sink_json(
                    sink_target,
                    JsonWriterOptions::default(),
                    None,
                    SinkOptions::default(),
                )?.collect()?
            }
            _ => unreachable!(),
        };
        Ok(())
    }

    fn easy_sink_partition<P: AsRef<Path>>(
        self,
        dir: P,
        partition_cols: &[&str],
        format: Option<Format>,
    ) -> PolarsResult<()> {
        let dir_path = dir.as_ref();

        if dir_path.is_file() {
            return Err(PolarsError::ComputeError(
                "Path is a file, not a directory".into(),
            ));
        }

        // 先创建目录
        fs::create_dir_all(dir_path).map_err(|e| {
            PolarsError::ComputeError(format!("Failed to create directory: {}", e).into())
        })?;

        // 🔥 获取规范化的绝对路径（在目录存在后）
        let canonical_path = dir_path.canonicalize().map_err(|e| {
            PolarsError::ComputeError(
                format!(
                    "Failed to canonicalize path '{}': {}",
                    dir_path.display(),
                    e
                )
                .into(),
            )
        })?;

        eprintln!("📁 Canonical path: {}", canonical_path.display());

        if partition_cols.is_empty() {
            return Err(PolarsError::ComputeError(
                "partition_cols cannot be empty".into(),
            ));
        }

        match format {
            Some(Format::Csv) => {
                write_partitioned(self, canonical_path, partition_cols, Format::Csv)?;
            }
            _ => {
                write_partitioned(self, canonical_path, partition_cols, Format::Parquet)?;
                eprintln!("✅ Parquet partition written");
            }
        }

        Ok(())
    }
}

// 🔥 核心函数：手动分区并写入
/// 手动实现分区写入（支持 Parquet 和 CSV）
fn write_partitioned<P: AsRef<Path>>(
    lf: LazyFrame,
    base_path: P,
    partition_cols: &[&str],
    format: Format,
) -> PolarsResult<()> {
    let dir_path = base_path.as_ref();

    // 创建输出目录
    fs::create_dir_all(dir_path).map_err(|e| {
        PolarsError::ComputeError(format!("Failed to create directory: {}", e).into())
    })?;

    // 收集数据
    eprintln!("📊 Collecting data...");
    let df = lf.collect()?;
    eprintln!("✅ Collected {} rows", df.height());

    if df.height() == 0 {
        eprintln!("⚠️  DataFrame is empty, nothing to write");
        return Ok(());
    }

    // 按分区列分组
    eprintln!("🔍 Grouping by partition columns: {:?}", partition_cols);
    let grouped = df.partition_by_stable(partition_cols.iter().cloned(), true)?;
    eprintln!("📦 Found {} partitions", grouped.len());

    // 为每个分组写入文件
    for (idx, mut partition_df) in grouped.into_iter().enumerate() {
        // 构建文件名：col1=value1_col2=value2
        let mut file_name_parts = Vec::new();

        for &col_name in partition_cols {
            let value = partition_df
                .column(col_name)?
                .get(0)?
                .to_string()
                .replace("\"", "") // 移除引号
                .replace("/", "-") // 替换路径分隔符
                .replace("\\", "-")
                .replace(" ", "_"); // 替换空格

            file_name_parts.push(format!("{}={}", col_name, value));
        }

        // 文件扩展名
        let extension = match format {
            Format::Parquet => "parquet",
            Format::Csv => "csv",
            Format::Json => "json",
        };

        let file_name = format!("{}.{}", file_name_parts.join("_"), extension);
        let file_path = dir_path.join(&file_name);

        eprintln!("💾 Writing file: {}", file_path.display());

        // 根据格式写入文件
        let mut file = fs::File::create(&file_path).map_err(|e| {
            PolarsError::ComputeError(
                format!("Failed to create file '{}': {}", file_path.display(), e).into(),
            )
        })?;

        match format {
            Format::Parquet => {
                ParquetWriter::new(&mut file).finish(&mut partition_df)?;
            }
            Format::Csv => {
                CsvWriter::new(&mut file)
                    .include_header(true)
                    .finish(&mut partition_df)?;
            }
            Format::Json => {
                // 🔥 新增 JSON 支持
                JsonWriter::new(&mut file)
                    .with_json_format(JsonFormat::JsonLines) // 使用 JSON Lines 格式
                    .finish(&mut partition_df)?;
            }
        }

        eprintln!("✅ Written {} rows to {}", partition_df.height(), file_name);
    }

    eprintln!("🎉 All partitions written successfully!");
    Ok(())
}

pub fn load_csv_vec<P: AsRef<str>>(paths: Vec<P>) -> PolarsResult<LazyFrame> {
    let paths: Arc<[PlPath]> = paths
        .into_iter()
        .map(|v| PlPath::from_str(v.as_ref()))
        .collect();
    // 使用最通用的参数, 避免复杂, 如果不满足需求, 自行调底层 API 处理
    let lf = LazyCsvReader::new_paths(paths)
        .with_separator(b',')
        .with_has_header(true)
        .with_glob(false)
        .finish()?;
    Ok(lf)
}

// load_csv("data/*.csv");
// load_csv("data/***/*.csv");
pub fn load_csv<P: AsRef<Path>>(path: P) -> PolarsResult<LazyFrame> {
    let path = PlPath::Local(Arc::from(path.as_ref()));
    let lf = LazyCsvReader::new(path)
        .with_separator(b',')
        .with_has_header(true)
        .with_glob(true)
        .finish()?;
    Ok(lf)
}

// 🔥 新增：加载 JSON 文件
/// 加载 JSON 文件，支持 glob 模式
/// 例如: load_json("data/*.json")
/// 例如: load_json("data/**/*.json")
pub fn load_json(path: &str) -> PolarsResult<LazyFrame> {
    // let args = ScanArgsAnonymous {
    //     infer_schema_length: Some(100),
    //     n_rows: None,
    //     row_index: None,
    //     ..Default::default()
    // };

    let lf = LazyJsonLineReader::new(PlPath::from_str(path))
        .finish()?;

    Ok(lf)
}

// 强制打开 glob
// 支持 load_parquet("data/*.parquet");
// 支持 load_parquet("data/***/*.parquet");
fn load_parquet(path: &str, arg: Option<ScanArgsParquet>) -> PolarsResult<LazyFrame> {
    let mut arg = arg.unwrap_or_default();
    arg.glob = true;
    let lf = LazyFrame::scan_parquet(PlPath::from_str(path), arg)?;
    Ok(lf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_csv() -> PolarsResult<()> {
        //let a = list_csv_files("../../data");
        //println!("{:?}", a);
        //
        let lf = load_csv_vec(vec![
            "../../data/stocks/002475.csv",
            "../../data/stocks/600000.csv",
            "../../data/stocks/603893.csv",
            "../../data/stocks/603986.csv",
        ])
        .unwrap();
        println!("{:?}", lf.collect().unwrap().height());

        let lf = load_csv("../../data/stocks/*.csv").unwrap();
        println!("{:?}", lf.collect().unwrap().height());
        Ok(())
    }

    #[test]
    fn test_parquet() -> PolarsResult<()> {
        let lf = load_parquet("../../tools/sina_news_parquet/2018.parquet", None).unwrap();
        // println!("2018 height: {:?}", lf.collect().unwrap().height());
        let cleaned = lf
            .with_column(
                col("pub_time")
                    .cast(DataType::Datetime(TimeUnit::Milliseconds, None))
                    .alias("datetime"),
            )
            .with_column(col("datetime").dt().strftime("%Y-%m").alias("year_month"))
            .with_column(col("datetime").dt().year().alias("year"))
            .with_column(col("datetime").dt().month().alias("month"));
        let lf_cleaned = cleaned.unique(Some(cols(["title"])), UniqueKeepStrategy::First);
        let result_df = lf_cleaned.collect().unwrap();
        // let r = result_df.lazy().easy_sink("./1.parquet").unwrap();
        result_df
            .lazy()
            .easy_sink_partition("./test-parquet-partition", &["year_month"], None)
            .unwrap();
        // result_df.lazy().easy_sink_partition("./test-csv-partition", &["year_month"], Some(PartitionFormat::Csv)).unwrap();
        Ok(())
    }

    #[test]
    fn test_parquet2() -> PolarsResult<()> {
        let df = df! {
            "year_month" => ["2018-07", "2018-08", "2018-07"],
            "value" => [1, 2, 3],
        }?;

        println!("DataFrame:\n{}", df);
        // 流式
        df.lazy().with_new_streaming(true).easy_sink_partition(
            "./test-parquet-partition",
            &["year_month"],
            None, // 默认是 Parquet
        )?;

        // 验证文件是否创建
        assert!(Path::new("./test-parquet-partition").exists());

        Ok(())
    }

    #[test]
    fn test_json() -> PolarsResult<()> {
        // 创建测试数据
        let df = df! {
            "name" => ["Alice", "Bob", "Charlie"],
            "age" => [25, 30, 35],
            "city" => ["NY", "LA", "SF"],
        }?;

        // 保存为 JSON
        df.clone().lazy().easy_sink("./test_output.json").unwrap();

        // 读取 JSON
        let lf = load_json("./test_output.json").unwrap();
        let loaded_df = lf.collect().unwrap();

        println!("Loaded JSON DataFrame:\n{}", loaded_df);

        // 清理测试文件
        std::fs::remove_file("./test_output.json").ok();

        Ok(())
    }

    #[test]
    fn test_json_partition() -> PolarsResult<()> {
        let df = df! {
            "year_month" => ["2018-07", "2018-08", "2018-07"],
            "category" => ["A", "B", "A"],
            "value" => [1, 2, 3],
        }?;

        println!("DataFrame:\n{}", df);

        // 分区保存为 JSON
        df.lazy().easy_sink_partition(
            "./test-json-partition",
            &["year_month"],
            Some(Format::Json),
        )?;

        // 验证文件是否创建
        assert!(Path::new("./test-json-partition").exists());

        // 清理测试目录
        std::fs::remove_dir_all("./test-json-partition").ok();

        Ok(())
    }
}

// https://docs.pola.rs/user-guide/getting-started/#group_by
