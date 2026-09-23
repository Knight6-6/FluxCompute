#pragma once

#include <flux/flux.hpp>
#include <cstdint>
#include <string>
#include <vector>
#include <cmath>
#include <random>
#include <limits>
#include <stdexcept>
#include <iostream>

namespace flux::xlt {

/**
 * @brief 降采样引擎 XltWinState 的微观结构聚合记录 (对齐 xlt_winstate.h 核心字段与口径)
 *
 * 价格以 int64_t × 10000 存储 (定点精度)
 * 量以 int64_t (股) 存储
 * 金额以 double (元) 存储
 */
struct XltWinRecord {
    char     code[16]{0};
    int64_t  window_seq{0};
    int64_t  window_close_sec{0};

    // 持久/价格状态
    int64_t  last_price_x{0};         // 现价 × 10000
    int64_t  limit_up_x{0};           // 涨停价 × 10000

    // 撤单
    int64_t  cxl_vol{0};
    int64_t  cxl_buy_vol{0};
    int64_t  cxl_sell_vol{0};

    // 大单委托新增
    int64_t  new_large_buy_vol{0};
    double   new_large_buy_amt{0.0};
    int64_t  new_large_sell_vol{0};
    double   new_large_sell_amt{0.0};

    // 大单主动成交 (4拆核心)
    int64_t  large_act_buy_vol{0};
    double   large_act_buy_amt{0.0};
    int64_t  large_act_sell_vol{0};
    double   large_act_sell_amt{0.0};

    // 全量成交 (按主动方向)
    int64_t  act_buy_vol{0};
    double   act_buy_amt{0.0};
    int64_t  act_sell_vol{0};
    double   act_sell_amt{0.0};

    // v3 集合竞价成交 (中性成交, 必须三项相加才守恒)
    int64_t  act_neu_vol{0};
    double   act_neu_amt{0.0};
    uint32_t act_neu_cnt{0};

    // v3 全量委托 (意图金额/量)
    int64_t  ord_buy_vol{0};
    double   ord_buy_amt{0.0};
    int64_t  ord_sell_vol{0};
    double   ord_sell_amt{0.0};

    // 笔数
    uint32_t trd_cnt{0};
    uint32_t cxl_cnt{0};
};

/**
 * @brief Flux 计算图的批量输入张量集 (Shape 统一为 [T, N])
 *
 * 维度约定:
 *   T (axis=0): 时间维 (时间窗口序列, 如每3秒一窗)
 *   N (axis=1): 标的维 (全市场股票截面)
 */
struct XltTensorBatch {
    std::size_t T = 0;                     // 时间窗口数
    std::size_t N = 0;                     // 股票标的数
    std::vector<std::string> codes;        // [N] 股票代码
    std::vector<int64_t> window_secs;      // [T] 市场墙钟秒 (秒-of-day)

    // 输入张量 (每个张量 Shape 均为 {T, N})
    tensor::Tensor<float> last_price;         // 最新成交价 (元, 由 last_price_x / 10000 转化)
    tensor::Tensor<float> limit_up;           // 涨停价 (元)
    tensor::Tensor<float> act_buy_amt;        // 主动买入成交额 (元)
    tensor::Tensor<float> act_sell_amt;       // 主动卖出成交额 (元)
    tensor::Tensor<float> act_neu_amt;        // 集合竞价中性成交额 (元)
    tensor::Tensor<float> large_act_buy_amt;  // 大单主动买入额 (元)
    tensor::Tensor<float> large_act_sell_amt; // 大单主动卖出额 (元)
    tensor::Tensor<float> ord_buy_amt;        // 全量买单委托额 (元)
    tensor::Tensor<float> ord_sell_amt;       // 全量卖单委托额 (元)
    tensor::Tensor<float> cxl_vol;            // 全量撤单量 (股)
    tensor::Tensor<float> ord_buy_vol;        // 全量买单委托量 (股)
    tensor::Tensor<float> ord_sell_vol;       // 全量卖单委托量 (股)
    tensor::Tensor<float> trd_cnt;            // 成交笔数

    XltTensorBatch() = default;

    XltTensorBatch(std::size_t t, std::size_t n, std::vector<std::string> c, std::vector<int64_t> w_secs)
        : T(t), N(n), codes(std::move(c)), window_secs(std::move(w_secs)),
          last_price(tensor::Shape({t, n})),
          limit_up(tensor::Shape({t, n})),
          act_buy_amt(tensor::Shape({t, n})),
          act_sell_amt(tensor::Shape({t, n})),
          act_neu_amt(tensor::Shape({t, n})),
          large_act_buy_amt(tensor::Shape({t, n})),
          large_act_sell_amt(tensor::Shape({t, n})),
          ord_buy_amt(tensor::Shape({t, n})),
          ord_sell_amt(tensor::Shape({t, n})),
          cxl_vol(tensor::Shape({t, n})),
          ord_buy_vol(tensor::Shape({t, n})),
          ord_sell_vol(tensor::Shape({t, n})),
          trd_cnt(tensor::Shape({t, n})) {}
};

/**
 * @brief 数据加载器与模拟器
 */
class XltDataLoader {
public:
    /**
     * @brief 构造仿真高频微观结构聚合数据 (严格符合 xltwin 行为)
     *
     * 特性设计:
     * 1. 窗口 0 (09:25 开盘集合竞价): 仅产生中性成交 (act_neu_amt > 0, 主买主卖为 0)
     * 2. 窗口 1 起 (连续竞价): 模拟买卖力量博弈、大单推波助澜、撤单与价格随机游走
     * 3. 注入边界用例: 设置部分标的处于停牌/无委托状态 (分母为0)，以验证除零保护 (NaN 传递)
     */
    static XltTensorBatch generate_mock_batch(std::size_t T = 20, std::size_t N = 5) {
        if (T < 2 || N < 2) {
            throw std::invalid_argument("T and N must be >= 2");
        }

        std::vector<std::string> codes;
        for (std::size_t i = 0; i < N; ++i) {
            char buf[32];
            if (i % 2 == 0) {
                std::snprintf(buf, sizeof(buf), "%06zu.SZ", i + 1);
            } else {
                std::snprintf(buf, sizeof(buf), "60%04zu.SH", i + 1);
            }
            codes.push_back(buf);
        }

        std::vector<int64_t> w_secs;
        int64_t sec0 = 9 * 3600 + 25 * 60; // 09:25:00
        for (std::size_t t = 0; t < T; ++t) {
            w_secs.push_back(sec0 + static_cast<int64_t>(t * 3));
        }

        XltTensorBatch batch(T, N, codes, w_secs);

        std::mt19937 rng(42);
        std::normal_distribution<float> ret_dist(0.0002f, 0.0015f);   // 价格微幅随机游走
        std::uniform_real_distribution<float> vol_dist(100.0f, 500.0f);

        std::vector<float> base_prices(N);
        for (std::size_t i = 0; i < N; ++i) {
            base_prices[i] = 10.0f + static_cast<float>(i * 12.5f);
        }

        for (std::size_t t = 0; t < T; ++t) {
            for (std::size_t n = 0; n < N; ++n) {
                std::size_t idx = t * N + n;

                // 涨停价 (设定为基准价格 + 10%)
                batch.limit_up[idx] = base_prices[n] * 1.10f;

                if (t == 0) {
                    // Window 0: 开盘集合竞价撮合
                    batch.last_price[idx] = base_prices[n];
                    batch.act_buy_amt[idx] = 0.0f;
                    batch.act_sell_amt[idx] = 0.0f;
                    // 中性成交: 开盘集合竞价集中释放
                    batch.act_neu_amt[idx] = base_prices[n] * (1000.0f + n * 200.0f);
                    batch.large_act_buy_amt[idx] = 0.0f;
                    batch.large_act_sell_amt[idx] = 0.0f;
                    batch.ord_buy_amt[idx] = batch.act_neu_amt[idx] * 1.5f;
                    batch.ord_sell_amt[idx] = batch.act_neu_amt[idx] * 1.2f;
                    batch.cxl_vol[idx] = 50.0f;
                    batch.ord_buy_vol[idx] = 2000.0f;
                    batch.ord_sell_vol[idx] = 1800.0f;
                    batch.trd_cnt[idx] = 10;
                } else {
                    // 连续竞价
                    float prev_px = batch.last_price[(t - 1) * N + n];
                    float r = ret_dist(rng);
                    float cur_px = prev_px * (1.0f + r);
                    batch.last_price[idx] = cur_px;

                    // 制造一个特例：倒数第 1 支股票在第 3 个窗口完全无交易/无委托 (测试除零保护)
                    if (n == N - 1 && t == 3) {
                        batch.act_buy_amt[idx] = 0.0f;
                        batch.act_sell_amt[idx] = 0.0f;
                        batch.act_neu_amt[idx] = 0.0f;
                        batch.large_act_buy_amt[idx] = 0.0f;
                        batch.large_act_sell_amt[idx] = 0.0f;
                        batch.ord_buy_amt[idx] = 0.0f;
                        batch.ord_sell_amt[idx] = 0.0f;
                        batch.cxl_vol[idx] = 0.0f;
                        batch.ord_buy_vol[idx] = 0.0f;
                        batch.ord_sell_vol[idx] = 0.0f;
                        batch.trd_cnt[idx] = 0;
                        continue;
                    }

                    // 正常连续竞价
                    float buy_power = 1.0f + r * 100.0f;
                    float sell_power = 1.0f - r * 100.0f;
                    if (buy_power < 0.2f) buy_power = 0.2f;
                    if (sell_power < 0.2f) sell_power = 0.2f;

                    float base_amt = cur_px * vol_dist(rng);
                    float buy_amt = base_amt * buy_power;
                    float sell_amt = base_amt * sell_power;

                    batch.act_buy_amt[idx] = buy_amt;
                    batch.act_sell_amt[idx] = sell_amt;
                    batch.act_neu_amt[idx] = 0.0f; // 连续竞价期间中性为 0

                    // 大单成交 (约为普通成交的 30% ~ 50%)
                    batch.large_act_buy_amt[idx] = buy_amt * (0.3f + 0.1f * (n % 3));
                    batch.large_act_sell_amt[idx] = sell_amt * (0.25f + 0.1f * ((n + 1) % 3));

                    // 委托申报
                    batch.ord_buy_amt[idx] = buy_amt * 2.0f;
                    batch.ord_sell_amt[idx] = sell_amt * 1.8f;
                    batch.ord_buy_vol[idx] = batch.ord_buy_amt[idx] / cur_px;
                    batch.ord_sell_vol[idx] = batch.ord_sell_amt[idx] / cur_px;

                    // 撤单
                    batch.cxl_vol[idx] = (batch.ord_buy_vol[idx] + batch.ord_sell_vol[idx]) * 0.15f;
                    batch.trd_cnt[idx] = 25 + static_cast<uint32_t>(n * 5);
                }
            }
        }
        return batch;
    }

    /**
     * @brief 从真实的 XltWinState 二维矩阵载入 (供后续对接 SHM 环形缓冲区或归档 dump bin)
     */
    static XltTensorBatch from_window_matrix(
        const std::vector<std::vector<XltWinRecord>>& records,
        const std::vector<std::string>& codes,
        const std::vector<int64_t>& w_secs) {

        std::size_t T = records.size();
        if (T == 0) throw std::invalid_argument("records must not be empty");
        std::size_t N = records[0].size();
        if (N != codes.size()) throw std::invalid_argument("codes size mismatch");

        XltTensorBatch batch(T, N, codes, w_secs);
        for (std::size_t t = 0; t < T; ++t) {
            for (std::size_t n = 0; n < N; ++n) {
                const auto& rec = records[t][n];
                std::size_t idx = t * N + n;

                batch.last_price[idx] = static_cast<float>(rec.last_price_x) / 10000.0f;
                batch.limit_up[idx] = static_cast<float>(rec.limit_up_x) / 10000.0f;
                batch.act_buy_amt[idx] = static_cast<float>(rec.act_buy_amt);
                batch.act_sell_amt[idx] = static_cast<float>(rec.act_sell_amt);
                batch.act_neu_amt[idx] = static_cast<float>(rec.act_neu_amt);
                batch.large_act_buy_amt[idx] = static_cast<float>(rec.large_act_buy_amt);
                batch.large_act_sell_amt[idx] = static_cast<float>(rec.large_act_sell_amt);
                batch.ord_buy_amt[idx] = static_cast<float>(rec.ord_buy_amt);
                batch.ord_sell_amt[idx] = static_cast<float>(rec.ord_sell_amt);
                batch.cxl_vol[idx] = static_cast<float>(rec.cxl_vol);
                batch.ord_buy_vol[idx] = static_cast<float>(rec.ord_buy_vol);
                batch.ord_sell_vol[idx] = static_cast<float>(rec.ord_sell_vol);
                batch.trd_cnt[idx] = static_cast<float>(rec.trd_cnt);
            }
        }
        return batch;
    }

    /**
     * @brief 生产服务器 Parquet / Arrow 数据载入扩展接口预留
     *
     * 架构预留说明:
     * 当在生产服务器上处理全历史大规模 Parquet 数据时，可通过 Apache Arrow C++ 或
     * Polars C API 将指定的列直接零拷贝映射至 std::vector<float> 并构造 Tensor:
     * tensor::Tensor<float> col_tensor(tensor::Shape({T, N}), arrow_col_data);
     */
    static void log_data_source_spec() {
        std::cout << "[XltDataLoader] Ready for (1) In-Memory Mock, (2) SHM Frame Ring, (3) Parquet Batch.\n";
    }
};

} // namespace flux::xlt
