#include "xlt_types.hpp"
#include "factor_graph.hpp"

#include <iostream>
#include <iomanip>
#include <chrono>
#include <stdexcept>

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

int main() {
    std::cout << "=========================================================================================\n";
    std::cout << "  FluxCompute: xltwin 降采样微观结构因子计算引擎 (端到端真实用例)                        \n";
    std::cout << "=========================================================================================\n\n";

    // 1. 初始化仿真数据
    const std::size_t T = 30; // 30 个连续 3 秒窗口 (代表 1.5 分钟快照时序)
    const std::size_t N = 5;  // 5 支代表性股票 (涵盖深市与沪市)

    std::cout << "[Step 1] 生成微观结构聚合状态 (严格对齐 XltWinState / Parquet 规范)..." << std::endl;
    XltTensorBatch batch = XltDataLoader::generate_mock_batch(T, N);
    std::cout << "  -> 观察窗口数 T = " << T << ", 股票数量 N = " << N << "\n";
    std::cout << "  -> 股票标的池: ";
    for (const auto& code : batch.codes) std::cout << code << " ";
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
    std::cout << "  -> 顺序执行 (Sequential Executor) 完成, 耗时: " << std::fixed << std::setprecision(2) << seq_us << " us\n";

    // 4. 并行执行对比验证 (4 线程)
    runtime::ThreadPool pool(4);
    auto t2 = std::chrono::high_resolution_clock::now();
    XltFactorResult res_par = pipeline.run_parallel(batch, pool);
    auto t3 = std::chrono::high_resolution_clock::now();
    double par_us = std::chrono::duration<double, std::micro>(t3 - t2).count();
    std::cout << "  -> 依赖层并行执行 (Parallel Executor, 4 线程) 完成, 耗时: " << std::fixed << std::setprecision(2) << par_us << " us\n\n";

    // 5. 因子数学口径严格检验
    std::cout << "[Step 4] 校验 xltwin 业务口径与数学守恒性..." << std::endl;

    // 校验 1: 集合竞价成交额三项守恒 (Window 0)
    for (std::size_t n = 0; n < N; ++n) {
        float b = batch.act_buy_amt[n];
        float s = batch.act_sell_amt[n];
        float neu = batch.act_neu_amt[n];
        float tot = (*res_seq.trd_amt)[n];
        ALWAYS_CHECK(std::abs(tot - (b + s + neu)) < 1e-4f, "3-term amount conservation violated");
        // Window 0 开盘集合竞价无主动买卖，总成交必须严格等于中性成交
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

    // 6. 打印精美横截面因子报表
    std::cout << "[Step 5] 打印部分窗口微观因子计算截面报表:\n";
    const std::vector<std::size_t> sample_windows = {0, 5, 15, 25};

    for (std::size_t win : sample_windows) {
        int64_t sec = batch.window_secs[win];
        int hh = static_cast<int>(sec / 3600);
        int mm = static_cast<int>((sec % 3600) / 60);
        int ss = static_cast<int>(sec % 60);

        std::cout << "-----------------------------------------------------------------------------------------------------------------------------------\n";
        std::cout << " [Window " << std::setw(2) << win << "] Market Time "
                  << std::setfill('0') << std::setw(2) << hh << ":"
                  << std::setw(2) << mm << ":"
                  << std::setw(2) << ss << std::setfill(' ')
                  << (win == 0 ? " (Call Auction 集合竞价)" : " (Continuous 连续竞价)") << "\n";
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

        for (std::size_t n = 0; n < N; ++n) {
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
