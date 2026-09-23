#pragma once

#include <flux/flux.hpp>
#include "xlt_types.hpp"
#include <memory>
#include <stdexcept>
#include <cmath>
#include <limits>

namespace flux::xlt {

/**
 * @brief 因子计算结果集
 */
struct XltFactorResult {
    // 基础微观特征 (对齐 xltwin 派生列 §4)
    tensor::TensorPtr<float> trd_amt;             // 三项相加总成交额 (元, 含竞价中性)
    tensor::TensorPtr<float> large_net_inflow;    // 大单主动净流入 (元)
    tensor::TensorPtr<float> net_inflow_ratio;    // 大单净流入占总成交比率
    tensor::TensorPtr<float> act_buy_share;       // 主动买入成交额占比 (分母不含中性)
    tensor::TensorPtr<float> ord_imb;             // 委托订单流不平衡 (OFI, -1.0 ~ 1.0)
    tensor::TensorPtr<float> cxl_ratio;           // 全量撤单率 (cxl_vol / (ord_buy_vol + ord_sell_vol))
    tensor::TensorPtr<float> pct_to_limit;        // 现价距离涨停价幅度 (last_price / limit_up - 1.0)

    // 时序动量与均线平滑
    tensor::TensorPtr<float> ret_1;               // 单窗收益率 (跨窗价格涨跌幅)
    tensor::TensorPtr<float> roll_ret_5;          // 5窗动量滚动均值
    tensor::TensorPtr<float> roll_vol_5;          // 5窗滚动波动率 (样本标准差)

    // 全市场横截面排序标准化 (0.0 ~ 1.0 分位数)
    tensor::TensorPtr<float> cs_rank_ret;         // 动量因子截面 Rank
    tensor::TensorPtr<float> cs_rank_imb;         // 委托不平衡截面 Rank
};

/**
 * @brief xltwin 因子计算图流水线
 */
class XltFactorPipeline {
public:
    graph::Graph<float> graph;

    // 输入节点
    std::shared_ptr<graph::Node<float>> in_price;
    std::shared_ptr<graph::Node<float>> in_limit_up;
    std::shared_ptr<graph::Node<float>> in_act_buy_amt;
    std::shared_ptr<graph::Node<float>> in_act_sell_amt;
    std::shared_ptr<graph::Node<float>> in_act_neu_amt;
    std::shared_ptr<graph::Node<float>> in_large_act_buy_amt;
    std::shared_ptr<graph::Node<float>> in_large_act_sell_amt;
    std::shared_ptr<graph::Node<float>> in_ord_buy_amt;
    std::shared_ptr<graph::Node<float>> in_ord_sell_amt;
    std::shared_ptr<graph::Node<float>> in_cxl_vol;
    std::shared_ptr<graph::Node<float>> in_ord_buy_vol;
    std::shared_ptr<graph::Node<float>> in_ord_sell_vol;

    // 输出节点
    std::shared_ptr<graph::Node<float>> out_trd_amt;
    std::shared_ptr<graph::Node<float>> out_large_net_inflow;
    std::shared_ptr<graph::Node<float>> out_net_inflow_ratio;
    std::shared_ptr<graph::Node<float>> out_act_buy_share;
    std::shared_ptr<graph::Node<float>> out_ord_imb;
    std::shared_ptr<graph::Node<float>> out_cxl_ratio;
    std::shared_ptr<graph::Node<float>> out_pct_to_limit;
    std::shared_ptr<graph::Node<float>> out_ret_1;
    std::shared_ptr<graph::Node<float>> out_roll_ret_5;
    std::shared_ptr<graph::Node<float>> out_roll_vol_5;
    std::shared_ptr<graph::Node<float>> out_cs_rank_ret;
    std::shared_ptr<graph::Node<float>> out_cs_rank_imb;

    XltFactorPipeline() {
        build_graph();
    }

    /**
     * @brief 注入输入并顺序执行计算
     */
    XltFactorResult run(const XltTensorBatch& batch) {
        runtime::Executor<float> executor;
        set_batch_inputs(executor, batch);
        executor.run(graph);
        return extract_results(executor);
    }

    /**
     * @brief 按依赖层并行执行计算
     */
    XltFactorResult run_parallel(const XltTensorBatch& batch, runtime::ThreadPool& pool) {
        runtime::Executor<float> executor;
        set_batch_inputs(executor, batch);
        executor.run_parallel(graph, pool);
        return extract_results(executor);
    }

private:
    void build_graph() {
        using NodeType = graph::NodeType;

        // 1. 创建输入节点
        in_price              = graph.create_node("Input(last_price)", NodeType::Input);
        in_limit_up           = graph.create_node("Input(limit_up)", NodeType::Input);
        in_act_buy_amt        = graph.create_node("Input(act_buy_amt)", NodeType::Input);
        in_act_sell_amt       = graph.create_node("Input(act_sell_amt)", NodeType::Input);
        in_act_neu_amt        = graph.create_node("Input(act_neu_amt)", NodeType::Input);
        in_large_act_buy_amt  = graph.create_node("Input(large_act_buy_amt)", NodeType::Input);
        in_large_act_sell_amt = graph.create_node("Input(large_act_sell_amt)", NodeType::Input);
        in_ord_buy_amt        = graph.create_node("Input(ord_buy_amt)", NodeType::Input);
        in_ord_sell_amt       = graph.create_node("Input(ord_sell_amt)", NodeType::Input);
        in_cxl_vol            = graph.create_node("Input(cxl_vol)", NodeType::Input);
        in_ord_buy_vol        = graph.create_node("Input(ord_buy_vol)", NodeType::Input);
        in_ord_sell_vol       = graph.create_node("Input(ord_sell_vol)", NodeType::Input);

        // 2. 算子节点: §4.1 三项相加总成交额 trd_amt = act_buy_amt + act_sell_amt + act_neu_amt
        auto op_trd_amt = graph.create_node("Op(trd_amt)", NodeType::Operator);
        graph.add_edge(in_act_buy_amt, op_trd_amt);
        graph.add_edge(in_act_sell_amt, op_trd_amt);
        graph.add_edge(in_act_neu_amt, op_trd_amt);
        graph.bind_op(op_trd_amt, [](const std::vector<tensor::TensorPtr<float>>& ins) {
            return ops::map3(*ins[0], *ins[1], *ins[2], [](float b, float s, float n) {
                return b + s + n;
            });
        });

        // 3. 算子节点: §4.2 大单净流入 large_net_inflow = large_act_buy_amt - large_act_sell_amt
        auto op_large_net_inflow = graph.create_node("Op(large_net_inflow)", NodeType::Operator);
        graph.add_edge(in_large_act_buy_amt, op_large_net_inflow);
        graph.add_edge(in_large_act_sell_amt, op_large_net_inflow);
        graph.bind_op(op_large_net_inflow, [](const std::vector<tensor::TensorPtr<float>>& ins) {
            return ops::sub(*ins[0], *ins[1]);
        });

        // 4. 算子节点: §4.2 净流入占比 net_inflow_ratio = large_net_inflow / trd_amt (分母<=0置NaN)
        auto op_net_inflow_ratio = graph.create_node("Op(net_inflow_ratio)", NodeType::Operator);
        graph.add_edge(op_large_net_inflow, op_net_inflow_ratio);
        graph.add_edge(op_trd_amt, op_net_inflow_ratio);
        graph.bind_op(op_net_inflow_ratio, [](const std::vector<tensor::TensorPtr<float>>& ins) {
            return ops::map2(*ins[0], *ins[1], [](float inflow, float tot) {
                return (tot > 0.0f) ? (inflow / tot) : std::numeric_limits<float>::quiet_NaN();
            });
        });

        // 5. 算子节点: §4.2 主动买入成交占比 act_buy_share = act_buy_amt / (act_buy_amt + act_sell_amt)
        auto op_act_buy_share = graph.create_node("Op(act_buy_share)", NodeType::Operator);
        graph.add_edge(in_act_buy_amt, op_act_buy_share);
        graph.add_edge(in_act_sell_amt, op_act_buy_share);
        graph.bind_op(op_act_buy_share, [](const std::vector<tensor::TensorPtr<float>>& ins) {
            return ops::map2(*ins[0], *ins[1], [](float b, float s) {
                float denom = b + s;
                return (denom > 0.0f) ? (b / denom) : std::numeric_limits<float>::quiet_NaN();
            });
        });

        // 6. 算子节点: §4.4 委托订单流不平衡 ord_imb = (ord_buy_amt - ord_sell_amt) / (ord_buy_amt + ord_sell_amt)
        auto op_ord_imb = graph.create_node("Op(ord_imb)", NodeType::Operator);
        graph.add_edge(in_ord_buy_amt, op_ord_imb);
        graph.add_edge(in_ord_sell_amt, op_ord_imb);
        graph.bind_op(op_ord_imb, [](const std::vector<tensor::TensorPtr<float>>& ins) {
            return ops::map2(*ins[0], *ins[1], [](float b, float s) {
                float denom = b + s;
                return (denom > 0.0f) ? ((b - s) / denom) : std::numeric_limits<float>::quiet_NaN();
            });
        });

        // 7. 算子节点: §4.3 全量撤单率 cxl_ratio = cxl_vol / (ord_buy_vol + ord_sell_vol)
        auto op_cxl_ratio = graph.create_node("Op(cxl_ratio)", NodeType::Operator);
        graph.add_edge(in_cxl_vol, op_cxl_ratio);
        graph.add_edge(in_ord_buy_vol, op_cxl_ratio);
        graph.add_edge(in_ord_sell_vol, op_cxl_ratio);
        graph.bind_op(op_cxl_ratio, [](const std::vector<tensor::TensorPtr<float>>& ins) {
            return ops::map3(*ins[0], *ins[1], *ins[2], [](float c, float b, float s) {
                float denom = b + s;
                return (denom > 0.0f) ? (c / denom) : std::numeric_limits<float>::quiet_NaN();
            });
        });

        // 8. 算子节点: §4.4 距涨停幅度 pct_to_limit = (last_price / limit_up) - 1.0
        auto op_pct_to_limit = graph.create_node("Op(pct_to_limit)", NodeType::Operator);
        graph.add_edge(in_price, op_pct_to_limit);
        graph.add_edge(in_limit_up, op_pct_to_limit);
        graph.bind_op(op_pct_to_limit, [](const std::vector<tensor::TensorPtr<float>>& ins) {
            return ops::map2(*ins[0], *ins[1], [](float p, float l) {
                return (l > 0.0f) ? (p / l - 1.0f) : std::numeric_limits<float>::quiet_NaN();
            });
        });

        // 9. 算子节点: 时序 shift(price, 1) [按 axis=0 时间轴平移]
        auto op_shift_price = graph.create_node("Op(shift_price)", NodeType::Operator);
        graph.add_edge(in_price, op_shift_price);
        graph.bind_op(op_shift_price, [](const std::vector<tensor::TensorPtr<float>>& ins) {
            return ops::shift(*ins[0], /*offset=*/1, /*axis=*/0);
        });

        // 10. 算子节点: 单窗收益率 ret_1 = (last_price - shift_price) / shift_price
        auto op_ret_1 = graph.create_node("Op(ret_1)", NodeType::Operator);
        graph.add_edge(in_price, op_ret_1);
        graph.add_edge(op_shift_price, op_ret_1);
        graph.bind_op(op_ret_1, [](const std::vector<tensor::TensorPtr<float>>& ins) {
            return ops::map2(*ins[0], *ins[1], [](float cur, float prev) {
                return (!std::isnan(cur) && !std::isnan(prev) && prev > 0.0f)
                       ? ((cur - prev) / prev)
                       : std::numeric_limits<float>::quiet_NaN();
            });
        });

        // 11. 算子节点: 5窗滚动均值 roll_ret_5 = rolling_mean(ret_1, 5) [axis=0]
        auto op_roll_ret_5 = graph.create_node("Op(roll_ret_5)", NodeType::Operator);
        graph.add_edge(op_ret_1, op_roll_ret_5);
        graph.bind_op(op_roll_ret_5, [](const std::vector<tensor::TensorPtr<float>>& ins) {
            return ops::rolling_mean(*ins[0], /*window=*/5, /*min_periods=*/1, /*axis=*/0);
        });

        // 12. 算子节点: 5窗滚动波动率 roll_vol_5 = rolling_std(ret_1, 5) [axis=0]
        auto op_roll_vol_5 = graph.create_node("Op(roll_vol_5)", NodeType::Operator);
        graph.add_edge(op_ret_1, op_roll_vol_5);
        graph.bind_op(op_roll_vol_5, [](const std::vector<tensor::TensorPtr<float>>& ins) {
            return ops::rolling_std(*ins[0], /*window=*/5, /*min_periods=*/2, /*axis=*/0);
        });

        // 13. 算子节点: 横截面 Rank cs_rank_ret = rank(roll_ret_5, axis=1, pct=true)
        auto op_cs_rank_ret = graph.create_node("Op(cs_rank_ret)", NodeType::Operator);
        graph.add_edge(op_roll_ret_5, op_cs_rank_ret);
        graph.bind_op(op_cs_rank_ret, [](const std::vector<tensor::TensorPtr<float>>& ins) {
            return ops::rank(*ins[0], /*axis=*/1, /*pct=*/true);
        });

        // 14. 算子节点: 横截面 Rank cs_rank_imb = rank(ord_imb, axis=1, pct=true)
        auto op_cs_rank_imb = graph.create_node("Op(cs_rank_imb)", NodeType::Operator);
        graph.add_edge(op_ord_imb, op_cs_rank_imb);
        graph.bind_op(op_cs_rank_imb, [](const std::vector<tensor::TensorPtr<float>>& ins) {
            return ops::rank(*ins[0], /*axis=*/1, /*pct=*/true);
        });

        // 15. 创建输出节点并连边
        out_trd_amt          = graph.create_node("Out(trd_amt)", NodeType::Output);
        out_large_net_inflow = graph.create_node("Out(large_net_inflow)", NodeType::Output);
        out_net_inflow_ratio = graph.create_node("Out(net_inflow_ratio)", NodeType::Output);
        out_act_buy_share    = graph.create_node("Out(act_buy_share)", NodeType::Output);
        out_ord_imb          = graph.create_node("Out(ord_imb)", NodeType::Output);
        out_cxl_ratio        = graph.create_node("Out(cxl_ratio)", NodeType::Output);
        out_pct_to_limit     = graph.create_node("Out(pct_to_limit)", NodeType::Output);
        out_ret_1            = graph.create_node("Out(ret_1)", NodeType::Output);
        out_roll_ret_5       = graph.create_node("Out(roll_ret_5)", NodeType::Output);
        out_roll_vol_5       = graph.create_node("Out(roll_vol_5)", NodeType::Output);
        out_cs_rank_ret      = graph.create_node("Out(cs_rank_ret)", NodeType::Output);
        out_cs_rank_imb      = graph.create_node("Out(cs_rank_imb)", NodeType::Output);

        graph.add_edge(op_trd_amt, out_trd_amt);
        graph.add_edge(op_large_net_inflow, out_large_net_inflow);
        graph.add_edge(op_net_inflow_ratio, out_net_inflow_ratio);
        graph.add_edge(op_act_buy_share, out_act_buy_share);
        graph.add_edge(op_ord_imb, out_ord_imb);
        graph.add_edge(op_cxl_ratio, out_cxl_ratio);
        graph.add_edge(op_pct_to_limit, out_pct_to_limit);
        graph.add_edge(op_ret_1, out_ret_1);
        graph.add_edge(op_roll_ret_5, out_roll_ret_5);
        graph.add_edge(op_roll_vol_5, out_roll_vol_5);
        graph.add_edge(op_cs_rank_ret, out_cs_rank_ret);
        graph.add_edge(op_cs_rank_imb, out_cs_rank_imb);
    }

    void set_batch_inputs(runtime::Executor<float>& executor, const XltTensorBatch& batch) {
        executor.set_input(in_price, batch.last_price);
        executor.set_input(in_limit_up, batch.limit_up);
        executor.set_input(in_act_buy_amt, batch.act_buy_amt);
        executor.set_input(in_act_sell_amt, batch.act_sell_amt);
        executor.set_input(in_act_neu_amt, batch.act_neu_amt);
        executor.set_input(in_large_act_buy_amt, batch.large_act_buy_amt);
        executor.set_input(in_large_act_sell_amt, batch.large_act_sell_amt);
        executor.set_input(in_ord_buy_amt, batch.ord_buy_amt);
        executor.set_input(in_ord_sell_amt, batch.ord_sell_amt);
        executor.set_input(in_cxl_vol, batch.cxl_vol);
        executor.set_input(in_ord_buy_vol, batch.ord_buy_vol);
        executor.set_input(in_ord_sell_vol, batch.ord_sell_vol);
    }

    XltFactorResult extract_results(const runtime::Executor<float>& executor) {
        XltFactorResult res;
        res.trd_amt          = executor.get_output(out_trd_amt);
        res.large_net_inflow = executor.get_output(out_large_net_inflow);
        res.net_inflow_ratio = executor.get_output(out_net_inflow_ratio);
        res.act_buy_share    = executor.get_output(out_act_buy_share);
        res.ord_imb          = executor.get_output(out_ord_imb);
        res.cxl_ratio        = executor.get_output(out_cxl_ratio);
        res.pct_to_limit     = executor.get_output(out_pct_to_limit);
        res.ret_1            = executor.get_output(out_ret_1);
        res.roll_ret_5       = executor.get_output(out_roll_ret_5);
        res.roll_vol_5       = executor.get_output(out_roll_vol_5);
        res.cs_rank_ret      = executor.get_output(out_cs_rank_ret);
        res.cs_rank_imb      = executor.get_output(out_cs_rank_imb);
        return res;
    }
};

} // namespace flux::xlt
