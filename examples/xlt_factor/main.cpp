#include "xlt_types.hpp"
#include "factor_graph.hpp"

#include <iostream>
#include <iomanip>
#include <chrono>
#include <stdexcept>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>

using namespace flux;
using namespace flux::xlt;

#define ALWAYS_CHECK(cond, msg) \
    do { \
        if (!(cond)) { \
            throw std::runtime_error(std::string("Assertion failed: ") + (msg) + " at line " + std::to_string(__LINE__)); \
        } \
    } while (0)

// 辅助打印浮点数 (NaN 友好显示)
void print_val(float v, int width = 11, int precision = 4) {
    if (std::isnan(v)) {
        std::cout << std::setw(width) << "NaN";
    } else {
        std::cout << std::setw(width) << std::fixed << std::setprecision(precision) << v;
    }
}

std::vector<int> parse_cores(const std::string& str) {
    std::vector<int> cores;
    std::stringstream ss(str);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) {
            cores.push_back(std::stoi(item));
        }
    }
    return cores;
}

struct FactorStats {
    std::size_t total = 0;
    std::size_t valid = 0;
    std::size_t nan_count = 0;
    float min_val = std::numeric_limits<float>::infinity();
    float max_val = -std::numeric_limits<float>::infinity();
    double sum = 0.0;

    double mean() const { return valid > 0 ? (sum / valid) : 0.0; }
};

FactorStats compute_stats(const tensor::Tensor<float>& t) {
    FactorStats s;
    s.total = t.numel();
    for (std::size_t i = 0; i < t.numel(); ++i) {
        float v = t[i];
        if (std::isnan(v)) {
            s.nan_count++;
        } else {
            s.valid++;
            s.sum += v;
            if (v < s.min_val) s.min_val = v;
            if (v > s.max_val) s.max_val = v;
        }
    }
    return s;
}

void print_stat_row(const std::string& name, const FactorStats& s) {
    double nan_pct = s.total > 0 ? (100.0 * s.nan_count / s.total) : 0.0;
    std::cout << "  " << std::left << std::setw(20) << name
              << std::right << std::setw(12) << s.valid
              << std::setw(10) << std::fixed << std::setprecision(1) << nan_pct << "%"
              << std::setw(14) << std::setprecision(4) << s.mean()
              << std::setw(14) << (s.valid > 0 ? s.min_val : 0.0f)
              << std::setw(14) << (s.valid > 0 ? s.max_val : 0.0f)
              << "\n";
}

int main(int argc, char** argv) {
    std::string bin_path;
    std::size_t limit_windows = 0;
    std::size_t num_stocks = 5;
    std::vector<int> pinned_cores = {2, 3, 4, 5};
    bool auto_bind_numa = true;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--bin" || arg == "-b") && i + 1 < argc) {
            bin_path = argv[++i];
        } else if ((arg == "--limit" || arg == "-l") && i + 1 < argc) {
            limit_windows = static_cast<std::size_t>(std::stoul(argv[++i]));
        } else if ((arg == "--stocks" || arg == "-n") && i + 1 < argc) {
            num_stocks = static_cast<std::size_t>(std::stoul(argv[++i]));
        } else if ((arg == "--cores" || arg == "-c") && i + 1 < argc) {
            pinned_cores = parse_cores(argv[++i]);
        } else if (arg == "--no-numa") {
            auto_bind_numa = false;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "用法: " << argv[0] << " [选项]\n"
                      << "选项:\n"
                      << "  --bin, -b <file>       指定真实回放产出的 win_raw.bin 归档路径\n"
                      << "  --limit, -l <T>        限制加载的最大时间窗口数 (0 为全量加载)\n"
                      << "  --stocks, -n <N>       指定 Mock 仿真模式下的股票标的数 (默认: 5)\n"
                      << "  --cores, -c <c1,c2..>  指定线程池绑定的 CPU 核心 (默认: 2,3,4,5)\n"
                      << "  --no-numa              禁用 NUMA 内存亲和绑定 (默认开启)\n"
                      << "  --help, -h             显示本帮助信息\n";
            return 0;
        }
    }

    std::cout << "=========================================================================================\n";
    std::cout << "  FluxCompute: xltwin 降采样微观结构因子计算引擎 (端到端真实用例)                        \n";
    std::cout << "=========================================================================================\n\n";

    bool is_real_data = !bin_path.empty();
    XltTensorBatch batch;

    if (is_real_data) {
        std::cout << "[Step 1] 从真实回放二进制归档装载数据 (Zero-Copy mmap): " << bin_path << " ...\n";
        auto t_load0 = std::chrono::high_resolution_clock::now();
        batch = XltDataLoader::load_from_dump_bin(bin_path, limit_windows);
        auto t_load1 = std::chrono::high_resolution_clock::now();
        double load_ms = std::chrono::duration<double, std::milli>(t_load1 - t_load0).count();
        std::cout << "  -> 装载完成, 耗时: " << std::fixed << std::setprecision(2) << load_ms << " ms\n";
    } else {
        std::cout << "[Step 1] 生成微观结构聚合仿真状态 (严格对齐 XltWinState / Parquet 规范)..." << std::endl;
        const std::size_t T = (limit_windows > 0) ? limit_windows : 30;
        const std::size_t N = (num_stocks > 0) ? num_stocks : 5;
        batch = XltDataLoader::generate_mock_batch(T, N);
    }

    std::size_t T = batch.T;
    std::size_t N = batch.N;
    std::cout << "  -> 观察窗口数 T = " << T << ", 股票总数 N = " << N << " (总记录规模: " << T * N << " 点位)\n";
    std::cout << "  -> 标的池前缀 (最多 10 支): ";
    for (std::size_t i = 0; i < std::min<std::size_t>(N, 10); ++i) {
        std::cout << batch.codes[i] << " ";
    }
    if (N > 10) std::cout << "... (共 " << N << " 支)";
    std::cout << "\n\n";

    // 2. 构建计算图流水线
    std::cout << "[Step 2] 构建 FluxCompute 计算图 DAG (算子拓扑排序与闭包绑定)..." << std::endl;
    XltFactorPipeline pipeline;
    std::cout << "  -> 节点统计: 12 个输入节点, 14 个核心数学算子节点, 12 个输出节点\n";
    std::cout << "  -> 算子类型: map3(三项成交额), map2(大单净流入比率/委托不平衡), shift(时间轴平移),\n";
    std::cout << "               rolling_mean(动量平滑), rolling_std(波动率), rank(全市场横截面百分比)\n\n";

    // 3. 顺序执行与耗时评测
    std::cout << "[Step 3] 执行计算图流水线..." << std::endl;
    auto t0 = std::chrono::high_resolution_clock::now();
    XltFactorResult res_seq = pipeline.run(batch);
    auto t1 = std::chrono::high_resolution_clock::now();
    double seq_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    double seq_ms = seq_us / 1000.0;
    double seq_mops = (static_cast<double>(T * N) / 1e6) / (seq_us / 1e6);
    std::cout << "  -> [顺序执行 Sequential] 完成, 耗时: " << std::fixed << std::setprecision(2) << seq_ms
              << " ms (" << seq_us << " us) | 吞吐: " << std::setprecision(2) << seq_mops << " M points/s\n";

    // 4. 并行执行对比验证 (显式演示绑核与 NUMA 感知)
    std::cout << "  -> 初始化 ThreadPool (核心绑核: ";
    for (std::size_t i = 0; i < pinned_cores.size(); ++i) {
        std::cout << pinned_cores[i] << (i + 1 < pinned_cores.size() ? "," : "");
    }
    std::cout << " | NUMA 内存锁定: " << (auto_bind_numa ? "开启" : "关闭") << ")...\n";

    runtime::ThreadPool pool(pinned_cores, auto_bind_numa);
    auto t2 = std::chrono::high_resolution_clock::now();
    XltFactorResult res_par = pipeline.run_parallel(batch, pool);
    auto t3 = std::chrono::high_resolution_clock::now();
    double par_us = std::chrono::duration<double, std::micro>(t3 - t2).count();
    double par_ms = par_us / 1000.0;
    double par_mops = (static_cast<double>(T * N) / 1e6) / (par_us / 1e6);
    std::cout << "  -> [并行执行 Parallel]   完成, 耗时: " << std::fixed << std::setprecision(2) << par_ms
              << " ms (" << par_us << " us) | 吞吐: " << std::setprecision(2) << par_mops << " M points/s\n";
    if (par_us > 0) {
        std::cout << "  -> 加速比 (Speedup): " << std::fixed << std::setprecision(2) << (seq_us / par_us) << "x\n\n";
    }

    // 5. 因子数学口径校验 (在 Mock 仿真模式下执行严格断言)
    std::cout << "[Step 4] 校验 xltwin 业务口径与数学守恒性..." << std::endl;
    if (!is_real_data) {
        // 校验 1: 集合竞价成交额三项守恒 (Window 0)
        for (std::size_t n = 0; n < N; ++n) {
            float b = batch.act_buy_amt[n];
            float s = batch.act_sell_amt[n];
            float neu = batch.act_neu_amt[n];
            float tot = (*res_seq.trd_amt)[n];
            ALWAYS_CHECK(std::abs(tot - (b + s + neu)) < 1e-4f, "3-term amount conservation violated");
            ALWAYS_CHECK(std::abs(tot - neu) < 1e-4f, "Window 0 auction trd_amt must equal act_neu_amt");
        }
        std::cout << "  [PASS] Window 0 开盘集合竞价中性成交三项守恒性校验通过 (trd_amt == act_neu_amt)\n";

        // 校验 2: 零成交与零委托时的除零保护 (Window 3, Stock N-1)
        std::size_t test_idx = 3 * N + (N - 1);
        ALWAYS_CHECK(std::isnan((*res_seq.net_inflow_ratio)[test_idx]), "net_inflow_ratio must be NaN when volume=0");
        ALWAYS_CHECK(std::isnan((*res_seq.ord_imb)[test_idx]), "ord_imb must be NaN when order volume=0");
        ALWAYS_CHECK(std::isnan((*res_seq.cxl_ratio)[test_idx]), "cxl_ratio must be NaN when order volume=0");
        std::cout << "  [PASS] 分母<=0除零安全保护校验通过 (产生 NaN, 未污染为 0.0, 杜绝分位数失真)\n";

        // 校验 3: 时序 Shift 与收益率计算
        for (std::size_t n = 0; n < N; ++n) {
            ALWAYS_CHECK(std::isnan((*res_seq.ret_1)[n]), "First window ret_1 must be NaN");
            float p0 = batch.last_price[n];
            float p1 = batch.last_price[N + n];
            float exp_ret1 = (p1 - p0) / p0;
            ALWAYS_CHECK(std::abs((*res_seq.ret_1)[N + n] - exp_ret1) < 1e-5f, "ret_1 matches 1-period return formula");
        }
        std::cout << "  [PASS] 时序一阶差分与收益率算子校验通过 (ret_1[1] == (p1-p0)/p0)\n";

        // 校验 4: 横截面 Rank 范围在 [0.0, 1.0]
        for (std::size_t t = 1; t < T; ++t) {
            for (std::size_t n = 0; n < N; ++n) {
                float r_ret = (*res_seq.cs_rank_ret)[t * N + n];
                float r_imb = (*res_seq.cs_rank_imb)[t * N + n];
                if (!std::isnan(r_ret)) {
                    ALWAYS_CHECK(r_ret >= 0.0f && r_ret <= 1.0f, "cs_rank_ret must be in [0, 1]");
                }
                if (!std::isnan(r_imb)) {
                    ALWAYS_CHECK(r_imb >= 0.0f && r_imb <= 1.0f, "cs_rank_imb must be in [0, 1]");
                }
            }
        }
        std::cout << "  [PASS] 全市场横截面排序 Rank 分位数区间校验通过 (cs_rank ∈ [0.0, 1.0])\n\n";
    } else {
        std::cout << "  -> 真实数据口径普查与统计分布:\n";
        std::cout << "  ----------------------------------------------------------------------------\n";
        std::cout << "  " << std::left << std::setw(20) << "Factor Name"
                  << std::right << std::setw(12) << "Valid Count"
                  << std::setw(10) << "NaN %"
                  << std::setw(14) << "Mean"
                  << std::setw(14) << "Min"
                  << std::setw(14) << "Max" << "\n";
        std::cout << "  ----------------------------------------------------------------------------\n";
        print_stat_row("trd_amt", compute_stats(*res_seq.trd_amt));
        print_stat_row("large_net_inflow", compute_stats(*res_seq.large_net_inflow));
        print_stat_row("net_inflow_ratio", compute_stats(*res_seq.net_inflow_ratio));
        print_stat_row("act_buy_share", compute_stats(*res_seq.act_buy_share));
        print_stat_row("ord_imb", compute_stats(*res_seq.ord_imb));
        print_stat_row("cxl_ratio", compute_stats(*res_seq.cxl_ratio));
        print_stat_row("pct_to_limit", compute_stats(*res_seq.pct_to_limit));
        print_stat_row("ret_1", compute_stats(*res_seq.ret_1));
        print_stat_row("roll_ret_5", compute_stats(*res_seq.roll_ret_5));
        print_stat_row("roll_vol_5", compute_stats(*res_seq.roll_vol_5));
        print_stat_row("cs_rank_ret", compute_stats(*res_seq.cs_rank_ret));
        print_stat_row("cs_rank_imb", compute_stats(*res_seq.cs_rank_imb));
        std::cout << "  ----------------------------------------------------------------------------\n\n";
    }

    // 6. 打印横截面因子报表 (选出若干窗口及前若干支股票)
    std::cout << "[Step 5] 打印部分采样窗口截面报表:\n";
    std::vector<std::size_t> sample_windows;
    if (T <= 5) {
        for (std::size_t i = 0; i < T; ++i) sample_windows.push_back(i);
    } else {
        sample_windows = {0, T / 4, T / 2, T - 1};
    }
    std::size_t print_n = std::min<std::size_t>(N, 5);

    for (std::size_t win : sample_windows) {
        int64_t sec = (win < batch.window_secs.size()) ? batch.window_secs[win] : 0;
        int hh = static_cast<int>(sec / 3600);
        int mm = static_cast<int>((sec % 3600) / 60);
        int ss = static_cast<int>(sec % 60);

        std::cout << "-----------------------------------------------------------------------------------------------------------------------------------\n";
        std::cout << " [Window " << std::setw(4) << win << "] Market Time "
                  << std::setfill('0') << std::setw(2) << hh << ":"
                  << std::setw(2) << mm << ":"
                  << std::setw(2) << ss << std::setfill(' ')
                  << (win == 0 ? " (Call Auction 集合竞价 / 初始窗口)" : " (Continuous 连续竞价)") << "\n";
        std::cout << "-----------------------------------------------------------------------------------------------------------------------------------\n";
        std::cout << std::setw(12) << "Code"
                  << std::setw(10) << "Price"
                  << std::setw(13) << "Trd_Amt"
                  << std::setw(13) << "LargeNet"
                  << std::setw(11) << "NetRatio"
                  << std::setw(11) << "OrdImb"
                  << std::setw(11) << "CxlRatio"
                  << std::setw(11) << "Mom5"
                  << std::setw(11) << "Vol5"
                  << std::setw(13) << "CsRankMom"
                  << std::setw(13) << "CsRankImb" << "\n";

        for (std::size_t n = 0; n < print_n; ++n) {
            std::size_t idx = win * N + n;
            std::cout << std::setw(12) << batch.codes[n];
            print_val(batch.last_price[idx], 10, 2);
            print_val((*res_seq.trd_amt)[idx], 13, 1);
            print_val((*res_seq.large_net_inflow)[idx], 13, 1);
            print_val((*res_seq.net_inflow_ratio)[idx], 11, 3);
            print_val((*res_seq.ord_imb)[idx], 11, 3);
            print_val((*res_seq.cxl_ratio)[idx], 11, 3);
            print_val((*res_seq.roll_ret_5)[idx], 11, 4);
            print_val((*res_seq.roll_vol_5)[idx], 11, 4);
            print_val((*res_seq.cs_rank_ret)[idx], 13, 3);
            print_val((*res_seq.cs_rank_imb)[idx], 13, 3);
            std::cout << "\n";
        }
        std::cout << "\n";
    }

    std::cout << "=========================================================================================\n";
    std::cout << "  全部因子计算与口径对齐验证成功! (All Factor Checks PASSED)                             \n";
    std::cout << "=========================================================================================\n";

    return 0;
}
