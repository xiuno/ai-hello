// 日期相关的特性, features

pub mod holiday_feature {
    use chrono::{Datelike, NaiveDate, Weekday};

    #[allow(dead_code)]
    #[derive(Debug, Default, Clone)]
    pub struct DateFeature {
        pub is_holiday: bool,           // 普通假日, 周末
        pub is_holiday_first_day: bool, // 假期第一天
        pub is_holiday_last_day: bool,  // 假期最后一天
        pub is_holiday_before_1day: bool, // 假期前一天
        pub is_holiday_after_1day: bool,  // 假期后一天
        pub is_national_day: bool,      // 国庆节
        pub is_spring_festival: bool,   // 春节
        pub is_christmas: bool,         // 圣诞节
    }

    impl TryFrom<&str> for DateFeature {
        type Error = Box<dyn std::error::Error>;
        fn try_from(date: &str) -> Result<Self, Self::Error> {
            from_date(date)
        }
    }

    pub fn from_date(date: &str) -> Result<DateFeature, Box<dyn std::error::Error>> {
        let current_date = NaiveDate::parse_from_str(date, "%Y-%m-%d").map_err(|_|"expect date format: %Y-%m-%d")?;

        let is_national_day = is_national_day(&current_date);
        let is_spring_festival = is_spring_festival(&current_date);

        // 计算节假日相关特征
        let is_public_holiday = is_national_day || is_spring_festival || is_weekend(&current_date);

        let (is_first_day, is_last_day) = if is_national_day {
            (current_date.day() == 1, current_date.day() == 7)
        } else if is_spring_festival {
            let spring_festival_dates = get_spring_festival_dates(current_date.year());
            (
                spring_festival_dates.first().map_or(false, |&d| d == current_date),
                spring_festival_dates.last().map_or(false, |&d| d == current_date)
            )
        } else {
            (false, false)
        };

        // 检查前一天和后一天
        let is_day_before = if let Some(yesterday) = current_date.pred_opt() {
            !is_public_holiday_type(&yesterday) && !is_weekend(&yesterday)
        } else {
            false
        };

        let is_day_after = if let Some(tomorrow) = current_date.succ_opt() {
            !is_public_holiday_type(&tomorrow) && !is_weekend(&tomorrow)
        } else {
            false
        };

        Ok(DateFeature {
            is_holiday: is_public_holiday,
            is_holiday_first_day: is_first_day,
            is_holiday_last_day: is_last_day,
            is_holiday_before_1day: is_day_before,
            is_holiday_after_1day: is_day_after,
            is_national_day,
            is_spring_festival,
            is_christmas: is_christmas(&current_date),
        })
    }

    // 判断是否是公共假日类型（用于前后一天判断）
    fn is_public_holiday_type(date: &NaiveDate) -> bool {
        is_national_day(date) || is_spring_festival(date)
    }

    // 判断是否是周末
    fn is_weekend(date: &NaiveDate) -> bool {
        matches!(date.weekday(), Weekday::Sat | Weekday::Sun)
    }

    // 判断是否是国庆节（10月1日-10月7日）
    fn is_national_day(date: &NaiveDate) -> bool {
        date.month() == 10 && date.day() >= 1 && date.day() <= 7
    }

    // 判断是否是春节（农历正月初一，这里用近似公历日期）
    fn is_spring_festival(date: &NaiveDate) -> bool {
        let year = date.year();
        let spring_festival_dates = get_spring_festival_dates(year);
        spring_festival_dates.contains(date)
    }

    // 判断是否是圣诞节
    fn is_christmas(date: &NaiveDate) -> bool {
        date.month() == 12 && date.day() == 25
    }

    // 获取春节日期范围（简化版，实际应该使用农历计算）
    fn get_spring_festival_dates(year: i32) -> Vec<NaiveDate> {
        let spring_festival_approx = match year {
            2020 => NaiveDate::from_ymd_opt(2020, 1, 25).unwrap(),
            2021 => NaiveDate::from_ymd_opt(2021, 2, 12).unwrap(),
            2022 => NaiveDate::from_ymd_opt(2022, 2, 1).unwrap(),
            2023 => NaiveDate::from_ymd_opt(2023, 1, 22).unwrap(),
            2024 => NaiveDate::from_ymd_opt(2024, 2, 10).unwrap(),
            2025 => NaiveDate::from_ymd_opt(2025, 1, 29).unwrap(),
            _ => NaiveDate::from_ymd_opt(year, 1, 25).unwrap(), // 默认
        };

        (-7..=7)
            .filter_map(|offset| spring_festival_approx.checked_add_signed(chrono::Duration::days(offset)))
            .collect()
    }

    // 测试函数
    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_national_day() {
            let features = from_date("2023-10-01").unwrap();
            assert!(features.is_national_day);
            assert!(features.is_holiday);
            assert!(features.is_holiday_first_day);

            let features = from_date("2023-10-02").unwrap();
            assert!(features.is_national_day);
            assert!(features.is_holiday);

            let features = from_date("2023-10-07").unwrap();
            assert!(features.is_national_day);
            assert!(features.is_holiday);
            assert!(features.is_holiday_last_day);
        }

        #[test]
        fn test_weekend() {
            let features = from_date("2023-10-14").unwrap(); // 周六
            assert!(features.is_holiday);

            let features = from_date("2023-10-15").unwrap(); // 周日
            assert!(features.is_holiday);

            let features = from_date("2023-10-16").unwrap(); // 周一
            assert!(!features.is_holiday);
        }

        #[test]
        fn test_christmas() {
            let features = from_date("2023-12-25").unwrap();
            assert!(features.is_christmas);
        }
    }
}

// 将日期转成周期性值
pub mod cycle_feature {
    use chrono::{NaiveDate, Datelike};

    /// 将日期转换为周期性编码的6维向量
    /// 返回格式: [dow_sin, dow_cos, dom_sin, dom_cos, month_sin, month_cos]
    pub struct DateFeature {
        pub dow_sin: f32,
        pub dow_cos: f32,
        pub dom_sin: f32,
        pub dom_cos: f32,
        pub month_sin: f32,
        pub month_cos: f32,
    }
    pub fn from_date(date: NaiveDate) -> DateFeature {
        let day_of_week = date.weekday().num_days_from_monday() as f32; // 0-6 (周一至周日)
        let day_of_month = (date.day() - 1) as f32; // 0-30 (0-based)
        let month = (date.month() - 1) as f32;      // 0-11 (0-based)

        // 周期编码
        let dow_sin = (2.0 * std::f32::consts::PI * day_of_week / 7.0).sin();
        let dow_cos = (2.0 * std::f32::consts::PI * day_of_week / 7.0).cos();

        let dom_sin = (2.0 * std::f32::consts::PI * day_of_month / 31.0).sin();
        let dom_cos = (2.0 * std::f32::consts::PI * day_of_month / 31.0).cos();

        let month_sin = (2.0 * std::f32::consts::PI * month / 12.0).sin();
        let month_cos = (2.0 * std::f32::consts::PI * month / 12.0).cos();

        DateFeature {
            dow_sin,
            dow_cos,
            dom_sin,
            dom_cos,
            month_sin,
            month_cos,
        }
    }

    /// 批量日期编码，便于Tensor构建
    pub fn from_date_vec(dates: &[NaiveDate]) -> Vec<DateFeature> {
        dates.iter().map(|&date| from_date(date)).collect()
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_from_date() {
            // 测试用例1: 2025-10-24 (周五)
            let date = NaiveDate::from_ymd_opt(2025, 10, 24).unwrap();
            let feature = from_date(date);

            // 验证周期性特征的范围 (-1.0 到 1.0)
            assert!(feature.dom_cos >= -1.0 && feature.dom_cos <= 1.0);
            assert!(feature.dom_sin >= -1.0 && feature.dom_sin <= 1.0);
            assert!(feature.dow_cos >= -1.0 && feature.dow_cos <= 1.0);
            assert!(feature.dow_sin >= -1.0 && feature.dow_sin <= 1.0);
            assert!(feature.month_cos >= -1.0 && feature.month_cos <= 1.0);
            assert!(feature.month_sin >= -1.0 && feature.month_sin <= 1.0);

            // 验证周五的编码 (day_of_week = 4)
            let expected_dow_sin = (2.0 * std::f32::consts::PI * 4.0 / 7.0).sin();
            let expected_dow_cos = (2.0 * std::f32::consts::PI * 4.0 / 7.0).cos();
            assert!((feature.dow_sin - expected_dow_sin).abs() < 1e-6);
            assert!((feature.dow_cos - expected_dow_cos).abs() < 1e-6);

            // 验证10月的编码 (month = 9)
            let expected_month_sin = (2.0 * std::f32::consts::PI * 9.0 / 12.0).sin();
            let expected_month_cos = (2.0 * std::f32::consts::PI * 9.0 / 12.0).cos();
            assert!((feature.month_sin - expected_month_sin).abs() < 1e-6);
            assert!((feature.month_cos - expected_month_cos).abs() < 1e-6);
        }

        #[test]
        fn test_from_date_vec() {
            let dates = vec![
                NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(),   // 元旦
                NaiveDate::from_ymd_opt(2025, 12, 31).unwrap(), // 年末
            ];

            let features = from_date_vec(&dates);

            assert_eq!(features.len(), 2);
        }

        #[test]
        fn test_cyclic_properties() {
            // 测试周期性：相隔一周的同一天应该有相同的星期编码
            let date1 = NaiveDate::from_ymd_opt(2025, 10, 24).unwrap(); // 周五
            let date2 = NaiveDate::from_ymd_opt(2025, 10, 31).unwrap(); // 下一个周五

            let features1 = from_date(date1);
            let features2 = from_date(date2);

            // 星期编码应该相同
            assert!(features1.dow_sin - features2.dow_sin.abs() < 1e-6); // dow_sin
            assert!(features1.dow_cos - features2.dow_cos.abs() < 1e-6); // dow_cos

            // 日期和月份编码应该不同
            assert!((features1.dom_sin - features2.dom_sin).abs() > 1e-6); // dom_sin
            assert!((features1.dom_cos - features2.dom_cos).abs() > 1e-6); // dom_cos
        }
    }
}