use serde::{Deserialize, Serialize};
/*
    处理数据
 */
#[derive(Serialize, Deserialize, Default)]
struct Story {
    text: String,
    date: String,
    petrochina: f32,
    sinopec: f32,
}

impl Story {
    fn load_json() -> Vec<Self> {
        let datasets_json = serde_json::json!([
            {
                "text": "国际原油价格大幅上涨，布伦特原油突破90美元/桶。",
                "date": "2025/10/24 00:00:00",
                "source": "新华社",
                "author": "张三",
                "stocks": {
                    "petrochina": 0.75,
                    "sinopec": 0.65
                }
            },
            {
                "text": "国家发改委宣布上调国内汽柴油价格，每吨上调300元。",
                "date": "2025/10/24 00:00:00",
                "source": "人民日报",
                "author": "李四",
                "petrochina": 0.40,
                "sinopec": 0.55
            },
            {
                "text": "中东地缘冲突升级，市场担忧原油供应中断。",
                "date": "2025/10/24 00:00:00",
                "source": "华商报",
                "author": "王五",
                "petrochina": 0.80,
                "sinopec": 0.70
            },
            {
                "text": "新能源汽车销量激增，传统燃油需求预期下调。",
                "date": "2025/10/24 00:00:00",
                "source": "财经新闻",
                "author": "赵六",
                "petrochina": -0.30,
                "sinopec": -0.45
            },
            {
                "text": "中石化旗下炼油厂开工率创历史新高，利润显著改善。",
                "date": "2025/10/24 00:00:00",
                "source": "新浪新闻",
                "author": "Jack",
                "petrochina": 0.10,
                "sinopec": 0.60
            },
            {
                "text": "中石油在塔里木盆地发现大型油气田，储量超亿吨。",
                "date": "2025/10/24 00:00:00",
                "source": "搜狐新闻",
                "author": "FoxZhang",
                "petrochina": 0.90,
                "sinopec": 0.20
            },
            {
                "text": "全球经济增长放缓，国际油价连续三周下跌。",
                "date": "2025/10/24 00:00:00",
                "source": "搜狐新闻",
                "author": "FoxZhang",
                "petrochina": -0.65,
                "sinopec": -0.55
            },
            {
                "text": "环保政策加码，多地限制高硫燃料油使用，利好清洁炼化企业。",
                "date": "2025/10/24 00:00:00",
                "source": "搜狐新闻",
                "author": "FoxZhang",
                "petrochina": 0.05,
                "sinopec": 0.40
            },
            {
                "text": "人民币汇率大幅贬值，进口原油成本上升。",
                "date": "2025/10/24 00:00:00",
                "source": "搜狐新闻",
                "author": "FoxZhang",
                "petrochina": 0.50,
                "sinopec": 0.30
            },
            {
                "text": "化工产品价格暴跌，中石化下游业务承压。",
                "date": "2025/10/24 00:00:00",
                "source": "搜狐新闻",
                "author": "FoxZhang",
                "petrochina": -0.20,
                "sinopec": -0.60
            },
            {
                "text": "冬季用油高峰来临，国内成品油库存降至低位。",
                "date": "2025/10/24 00:00:00",
                "source": "搜狐新闻",
                "author": "FoxZhang",
                "petrochina": 0.35,
                "sinopec": 0.50
            },
            {
                "text": "OPEC+宣布延长减产协议至明年，支撑油价上行。",
                "date": "2025/10/24 00:00:00",
                "source": "股吧中石油版块",
                "author": "追高的神",
                "petrochina": 0.70,
                "sinopec": 0.60
            }
        ]);
        let stories = serde_json::from_value::<Story>(datasets_json)?;
        return stories;
    }

}
