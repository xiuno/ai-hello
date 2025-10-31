pub mod ffd {
    fn load() -> anyhow::Result<Vec<u8>> {
        use polars::prelude::*;

            // 读取 CSV（假设文件为 "fed.csv"）
            let mut df = CsvReader::from_path("fed.csv")?
                .has_header(true)
                .finish()?;

            // 确保按日期排序
            df.sort_in_place(["observation_date"], SortOptions::default())?;

            // 计算 DFF 日变动：DFF - DFF.shift(1)
            let dff_chg = df.column("DFF")?
                .f64()?
                .shift_and_fill(-1, 0.0)  // 向上移一行（前一天的值），空值填 0.0
                .unwrap();

            let current_dff = df.column("DFF")?.f64()?;
            let change = current_dff - &dff_chg;

            // 第一天无前值，设为 0.0（或 NaN，这里用 0.0 简洁）
            let mut change_vec = change.into_no_null_iter().collect::<Vec<f64>>();
            if !change_vec.is_empty() {
                change_vec[0] = 0.0;
            }

            // 添加新列
            df.with_column(Series::new("fed_change", &change_vec))?;

            // 打印结果
            println!("{}", df);

            Ok(())
    }
}