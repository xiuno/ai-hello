use polars::prelude::*;

fn main() -> PolarsResult<()> {
    {
        let path = "tools/sina_news_parquet/2018.parquet"; // 注意你写的是 .parquest，但正确扩展名是 .parquet

        // 读取 Parquet 文件（惰性加载，不立即读入内存）
        let df = LazyFrame::scan_parquet(PlPath::from_str(path), Default::default())?
            .collect()?; // 实际加载数据

        println!("origin rows: {}", df.height());
        let mut df_dedup = df.unique_stable(Some(&["title".to_string()]), UniqueKeepStrategy::First, None)?;

        println!("dedup rows: {}", df_dedup.height());

        ParquetWriter::new(std::fs::File::create("./1.parquet")?).finish(&mut df_dedup)?;
    }
    {
        let path = "1.csv";
        let a = LazyCsvReader::new(path).finish()?;


    }
    // // 打印前10条
    // println!("=== 前10条记录 ===");
    // let head = df.head(Some(10));
    // println!("{:?}", head);

    // // 打印后10条
    // println!("\n=== 后10条记录 ===");
    // let tail = df.tail(Some(10));
    // println!("{:?}", tail);

    Ok(())
}