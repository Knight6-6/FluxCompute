#pragma once

#include <flux/flux.hpp>
#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <cmath>
#include <random>
#include <limits>
#include <stdexcept>
#include <iostream>
#include <cstring>

#if defined(__linux__) || defined(__unix__) || defined(__APPLE__)
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace flux::xlt {

/**
 * @brief 降采样引擎 XltWinState 结构体的严格 672B 二进制布局 (与 xlt_winstate.h v3 100% 对齐)
 */
struct XltWinRecord {
    // ---- 标识 (offset 0..39) ----
    char     code[16]{0};
    int64_t  window_seq{0};
    int64_t  window_close_sec{0};
    int64_t  publish_tsc{0};

    // ---- 持久状态 (offset 40..79) ----
    int64_t  resting_large_buy_vol{0};
    int64_t  resting_large_sell_vol{0};
    int64_t  seal_vol{0};
    int64_t  last_price_x{0};         // 最新成交价 × 10000 (定点精度)
    int64_t  limit_up_x{0};           // 涨停价 × 10000 (0=缺)

    // ---- 本窗 delta: 撤单 (offset 80..103) ----
    int64_t  cxl_vol{0};
    int64_t  large_cxl_buy_vol{0};
    int64_t  large_cxl_sell_vol{0};

    // ---- 本窗 delta: 大单委托新增 (offset 104..135) ----
    int64_t  new_large_buy_vol{0};    double new_large_buy_amt{0.0};
    int64_t  new_large_sell_vol{0};   double new_large_sell_amt{0.0};

    // ---- 本窗 delta: 大单成交 (offset 136..199) ----
    int64_t  large_act_buy_vol{0};    double large_act_buy_amt{0.0};
    int64_t  large_act_sell_vol{0};   double large_act_sell_amt{0.0};
    int64_t  large_pas_buy_vol{0};    double large_pas_buy_amt{0.0};
    int64_t  large_pas_sell_vol{0};   double large_pas_sell_amt{0.0};

    // ---- 本窗 delta: 全量成交 (offset 200..231) ----
    int64_t  act_buy_vol{0};   double act_buy_amt{0.0};
    int64_t  act_sell_vol{0};  double act_sell_amt{0.0};

    // ---- 本窗 delta: 封单变化 (offset 232..239) ----
    int64_t  seal_vol_delta{0};

    // ---- 计数 + 状态 (offset 240..255) ----
    uint32_t cxl_cnt{0};
    uint32_t new_large_buy_cnt{0};
    uint32_t new_large_sell_cnt{0};
    uint8_t  at_limit{0};
    uint8_t  opened_today{0};
    uint8_t  touched_limit{0};
    uint8_t  opened_this_window{0};

    // ---- 算法撤单 (offset 256..271) ----
    uint32_t cxl_buy_cnt{0};
    uint32_t cxl_sell_cnt{0};
    uint32_t algo_cxl_buy_cnt{0};
    uint32_t algo_cxl_sell_cnt{0};

    // ================= v3 增量字段 (272 -> 672 B) =================
    // ---- 成交概况 (offset 272..311) ----
    double   trd_open{0.0};
    double   trd_high{0.0};
    double   trd_low{0.0};
    uint32_t trd_cnt{0};
    uint32_t act_buy_cnt{0};
    uint32_t act_sell_cnt{0};

    // ---- 集合竞价中性成交 (offset 312..327) ----
    uint32_t act_neu_cnt{0};
    int64_t  act_neu_vol{0};
    double   act_neu_amt{0.0};

    // ---- 全量新增委托 (offset 328..383) ----
    int64_t  ord_buy_vol{0};    double ord_buy_amt{0.0};
    int64_t  ord_sell_vol{0};   double ord_sell_amt{0.0};
    uint32_t ord_buy_cnt{0};
    uint32_t ord_sell_cnt{0};
    uint32_t ord_mkt_buy_cnt{0};
    uint32_t ord_mkt_sell_cnt{0};
    uint32_t ord_sbo_buy_cnt{0};
    uint32_t ord_sbo_sell_cnt{0};

    // ---- 撤单分向 + 秒撤 (offset 384..407) ----
    int64_t  cxl_buy_vol{0};
    int64_t  cxl_sell_vol{0};
    uint32_t fastcxl_buy_cnt{0};
    uint32_t fastcxl_sell_cnt{0};

    // ---- C 族 金额档位矩阵 (offset 408..607) ----
    int64_t  tafix_b_vol[5]{0};  double tafix_b_amt[5]{0.0};
    int64_t  tafix_s_vol[5]{0};  double tafix_s_amt[5]{0.0};
    uint32_t tafix_b_cnt[5]{0};  uint32_t tafix_s_cnt[5]{0};

    // ---- D 族 成交间隔孤立性分桶 (offset 608..671) ----
    uint32_t gap_b_cnt[8]{0};
    uint32_t gap_s_cnt[8]{0};
};
static_assert(sizeof(XltWinRecord) == 672, "XltWinRecord ABI must strictly be 672 bytes matching XltWinState");
static_assert(offsetof(XltWinRecord, publish_tsc)         ==  32, "v2 layout drift");
static_assert(offsetof(XltWinRecord, act_sell_amt)        == 224, "v2 layout drift");
static_assert(offsetof(XltWinRecord, algo_cxl_sell_cnt)   == 268, "v2 layout drift");
static_assert(offsetof(XltWinRecord, trd_open)            == 272, "v3 layout drift");
static_assert(offsetof(XltWinRecord, act_neu_vol)         == 312, "v3 layout drift");
static_assert(offsetof(XltWinRecord, ord_buy_vol)         == 328, "v3 layout drift");
static_assert(offsetof(XltWinRecord, cxl_buy_vol)         == 384, "v3 layout drift");
static_assert(offsetof(XltWinRecord, tafix_b_vol)         == 408, "v3 layout drift");
static_assert(offsetof(XltWinRecord, gap_b_cnt)           == 608, "v3 layout drift");

/**
 * @brief Flux 计算图的批量输入张量集 (Shape 统一为 [T, N])
 */
struct XltTensorBatch {
    std::size_t T = 0;                     // 时间窗口数
    std::size_t N = 0;                     // 股票标的数
    std::vector<std::string> codes;        // [N] 股票代码
    std::vector<int64_t> window_secs;      // [T] 市场墙钟秒

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

    XltTensorBatch()
        : T(0), N(0),
          last_price(tensor::Shape({0, 0})),
          limit_up(tensor::Shape({0, 0})),
          act_buy_amt(tensor::Shape({0, 0})),
          act_sell_amt(tensor::Shape({0, 0})),
          act_neu_amt(tensor::Shape({0, 0})),
          large_act_buy_amt(tensor::Shape({0, 0})),
          large_act_sell_amt(tensor::Shape({0, 0})),
          ord_buy_amt(tensor::Shape({0, 0})),
          ord_sell_amt(tensor::Shape({0, 0})),
          cxl_vol(tensor::Shape({0, 0})),
          ord_buy_vol(tensor::Shape({0, 0})),
          ord_sell_vol(tensor::Shape({0, 0})),
          trd_cnt(tensor::Shape({0, 0})) {}

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
 * @brief 数据加载器与模拟器 (支持 Mock 仿真与直读服务器端回放产出的 win_raw.bin 归档)
 */
class XltDataLoader {
public:
    /**
     * @brief 构造仿真高频微观结构聚合数据 (严格符合 xltwin 行为)
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
        std::normal_distribution<float> ret_dist(0.0002f, 0.0015f);
        std::uniform_real_distribution<float> vol_dist(100.0f, 500.0f);

        std::vector<float> base_prices(N);
        for (std::size_t i = 0; i < N; ++i) {
            base_prices[i] = 10.0f + static_cast<float>(i * 12.5f);
        }

        for (std::size_t t = 0; t < T; ++t) {
            for (std::size_t n = 0; n < N; ++n) {
                std::size_t idx = t * N + n;
                batch.limit_up[idx] = base_prices[n] * 1.10f;

                if (t == 0) {
                    // Window 0: 开盘集合竞价撮合 (中性成交)
                    batch.last_price[idx] = base_prices[n];
                    batch.act_buy_amt[idx] = 0.0f;
                    batch.act_sell_amt[idx] = 0.0f;
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

                    // 制造一个测试边界：倒数第 1 支股票在第 3 个窗口完全无交易/无委托 (测试除零保护)
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

                    float buy_power = 1.0f + r * 100.0f;
                    float sell_power = 1.0f - r * 100.0f;
                    if (buy_power < 0.2f) buy_power = 0.2f;
                    if (sell_power < 0.2f) sell_power = 0.2f;

                    float base_amt = cur_px * vol_dist(rng);
                    float buy_amt = base_amt * buy_power;
                    float sell_amt = base_amt * sell_power;

                    batch.act_buy_amt[idx] = buy_amt;
                    batch.act_sell_amt[idx] = sell_amt;
                    batch.act_neu_amt[idx] = 0.0f;

                    batch.large_act_buy_amt[idx] = buy_amt * (0.3f + 0.1f * (n % 3));
                    batch.large_act_sell_amt[idx] = sell_amt * (0.25f + 0.1f * ((n + 1) % 3));

                    batch.ord_buy_amt[idx] = buy_amt * 2.0f;
                    batch.ord_sell_amt[idx] = sell_amt * 1.8f;
                    batch.ord_buy_vol[idx] = batch.ord_buy_amt[idx] / cur_px;
                    batch.ord_sell_vol[idx] = batch.ord_sell_amt[idx] / cur_px;

                    batch.cxl_vol[idx] = (batch.ord_buy_vol[idx] + batch.ord_sell_vol[idx]) * 0.15f;
                    batch.trd_cnt[idx] = 25 + static_cast<uint32_t>(n * 5);
                }
            }
        }
        return batch;
    }

    /**
     * @brief 直接从回放生成的二进制归档 win_raw.bin 零拷贝装载为张量批次 (可在服务器上直接运行)
     *
     * 对应 xlt_factor --dump <path> 产出的二进制归档
     */
    static XltTensorBatch load_from_dump_bin(const std::string& bin_path, std::size_t max_windows = 0) {
#if defined(__linux__) || defined(__unix__) || defined(__APPLE__)
        int fd = ::open(bin_path.c_str(), O_RDONLY);
        if (fd < 0) {
            throw std::runtime_error("无法打开二进制 dump 归档文件: " + bin_path);
        }
        struct stat st;
        if (::fstat(fd, &st) != 0 || st.st_size < 64) {
            ::close(fd);
            throw std::runtime_error("dump 文件体积非法或小于 64 字节: " + bin_path);
        }
        size_t sz = static_cast<size_t>(st.st_size);
        const char* base = static_cast<const char*>(::mmap(nullptr, sz, PROT_READ, MAP_PRIVATE, fd, 0));
        ::close(fd);
        if (base == MAP_FAILED) {
            throw std::runtime_error("mmap 失败: " + bin_path);
        }

        // 解析 64B XltWinDumpHdr
        struct DumpHdr {
            uint32_t magic;         // 0x44574C58 "XLWD"
            uint32_t version;       // 3
            uint32_t slot_sz;       // 672
            uint32_t trading_day;
            uint64_t start_unix;
            uint64_t n_windows;
            uint64_t codes_off;
            uint32_t n_slots;
            uint32_t step_sec;
            uint8_t  _pad[16];
        };
        const DumpHdr* h = reinterpret_cast<const DumpHdr*>(base);
        if (h->magic != 0x44574C58u) {
            ::munmap(const_cast<char*>(base), sz);
            throw std::runtime_error("dump 文件魔数校验失败 (非 XLWD 格式)");
        }
        if (h->slot_sz != 672) {
            ::munmap(const_cast<char*>(base), sz);
            throw std::runtime_error("dump 文件槽位大小不是 672 字节 (ABI 版本不一致)");
        }

        std::size_t T = h->n_windows;
        std::size_t N = h->n_slots;
        if (T == 0 || N == 0 || h->codes_off == 0) {
            ::munmap(const_cast<char*>(base), sz);
            throw std::runtime_error("dump 归档未正常收尾或无有效窗口");
        }

        if (max_windows > 0 && max_windows < T) {
            T = max_windows;
        }

        // 读取末尾股票代码表
        std::vector<std::string> codes;
        codes.reserve(N);
        const char* codes_ptr = base + h->codes_off;
        for (std::size_t i = 0; i < N; ++i) {
            char buf[17] = {0};
            std::memcpy(buf, codes_ptr + i * 16, 16);
            codes.push_back(buf);
        }

        std::vector<int64_t> w_secs;
        w_secs.reserve(T);

        XltTensorBatch batch(T, N, codes, {});

        // 顺序扫描每个窗口
        const char* ptr = base + 64;
        for (std::size_t t = 0; t < T; ++t) {
            if (ptr + 4 > base + h->codes_off) break;
            uint32_t K = *reinterpret_cast<const uint32_t*>(ptr);
            ptr += 4;
            const XltWinRecord* recs = reinterpret_cast<const XltWinRecord*>(ptr);
            ptr += K * 672;

            if (K > 0) {
                w_secs.push_back(recs[0].window_close_sec);
            } else {
                w_secs.push_back(0);
            }

            for (std::size_t k = 0; k < K && k < N; ++k) {
                const auto& r = recs[k];
                std::size_t idx = t * N + k;
                batch.last_price[idx]         = static_cast<float>(r.last_price_x) / 10000.0f;
                batch.limit_up[idx]           = static_cast<float>(r.limit_up_x) / 10000.0f;
                batch.act_buy_amt[idx]        = static_cast<float>(r.act_buy_amt);
                batch.act_sell_amt[idx]       = static_cast<float>(r.act_sell_amt);
                batch.act_neu_amt[idx]        = static_cast<float>(r.act_neu_amt);
                batch.large_act_buy_amt[idx]  = static_cast<float>(r.large_act_buy_amt);
                batch.large_act_sell_amt[idx] = static_cast<float>(r.large_act_sell_amt);
                batch.ord_buy_amt[idx]        = static_cast<float>(r.ord_buy_amt);
                batch.ord_sell_amt[idx]       = static_cast<float>(r.ord_sell_amt);
                batch.cxl_vol[idx]            = static_cast<float>(r.cxl_vol);
                batch.ord_buy_vol[idx]        = static_cast<float>(r.ord_buy_vol);
                batch.ord_sell_vol[idx]       = static_cast<float>(r.ord_sell_vol);
                batch.trd_cnt[idx]            = static_cast<float>(r.trd_cnt);
            }
        }
        batch.window_secs = std::move(w_secs);
        ::munmap(const_cast<char*>(base), sz);
        return batch;
#else
        throw std::runtime_error("当前操作系统不支持 POSIX mmap 二进制归档读取");
#endif
    }
};

} // namespace flux::xlt
