# xltwin 微观结构降采样因子计算示例 (FluxCompute 落地示例)

本目录展示了如何使用 **FluxCompute** 高性能计算图框架，承接 `qamdbroker_xlight` 降采样引擎输出的 `XltWinState` 截面数据，并构建端到端的时序与横截面微观结构因子。

---

## 一、架构定位与数据流

```text
[逐笔行情 (L2 Tick/Order/Exe)]
               │
               ▼
[降采样引擎 qamdbroker_xlight] (C++, 每3秒输出截面，不做因子)
               │
               ├─ 实时流: shm_md_frame<XltWinState> 环形缓冲区 (512 帧)
               └─ 历史批: win_raw.bin / win.parquet (Parquet 归档)
               │
               ▼
[FluxCompute 因子计算图 (本目录)]
   1. 基础特征提取 (Tensor<float>, Shape [T, N])
   2. 算子拓扑计算 (map2 / map3 / sub / div / shift / rolling / rank)
   3. 产出因子: 大单净流入比率、OFI 订单流不平衡、撤单率、滚动动量、横截面分位数
```

---

## 二、因子数学口径（严格对齐 xltwin 规范）

1. **总成交额（三项相加守恒）**：
   $$\text{trd\_amt} = \text{act\_buy\_amt} + \text{act\_sell\_amt} + \text{act\_neu\_amt}$$
   *必须包含集合竞价中性成交（`act_neu_amt`），开盘 09:25 集合竞价撮合无主动方向，遗漏会导致总额偏小。*

2. **大单主动净流入与净流比率**：
   $$\text{large\_net\_inflow} = \text{large\_act\_buy\_amt} - \text{large\_act\_sell\_amt}$$
   $$\text{net\_inflow\_ratio} = \frac{\text{large\_net\_inflow}}{\text{trd\_amt}} \quad (\text{分母} \le 0 \text{ 置 NaN})$$

3. **委托订单流不平衡（OFI, ord_imb）**：
   $$\text{ord\_imb} = \frac{\text{ord\_buy\_amt} - \text{ord\_sell\_amt}}{\text{ord\_buy\_amt} + \text{ord\_sell\_amt}} \quad (\text{分母} \le 0 \text{ 置 NaN})$$

4. **全量撤单率（cxl_ratio）**：
   $$\text{cxl\_ratio} = \frac{\text{cxl\_vol}}{\text{ord\_buy\_vol} + \text{ord\_sell\_vol}} \quad (\text{分母} \le 0 \text{ 置 NaN})$$

5. **时序动量与均线平滑（axis=0）**：
   $$\text{ret\_1}_t = \frac{P_t - P_{t-1}}{P_{t-1}}, \quad \text{roll\_ret\_5} = \text{rolling\_mean}(\text{ret\_1}, 5), \quad \text{roll\_vol\_5} = \text{rolling\_std}(\text{ret\_1}, 5)$$

6. **全市场横截面标准化（axis=1）**：
   $$\text{cs\_rank\_ret} = \text{rank}(\text{roll\_ret\_5}, \text{axis}=1, \text{pct}=\text{true}) \in [0.0, 1.0]$$

---

## 三、构建与运行

在 FluxCompute 根目录下执行：

```bash
# 1. 编译
cmake -B build
cmake --build build --target xlt_factor_example -j

# 2. 运行端到端计算与报表打印
./build/examples/xlt_factor/xlt_factor_example
```

---

## 四、对接生产服务器指南

### 1. 对接历史 Parquet 大数据
通过 Apache Arrow C++ 或 Polars C API 将 Parquet 文件的各列直接装载为 `[T, N]` 的 `std::vector<float>`，然后调用：
```cpp
tensor::Tensor<float> col_tensor(tensor::Shape({T, N}), arrow_col_buffer);
```
注入 `XltTensorBatch` 即可批量运行历史因子回测与参数网格搜索。

### 2. 对接实盘/回放共享内存（SHM）
若直接挂载 `/dev/shm/xlt_factor`（`shm_md_frame<XltWinState>`）：
通过 `XltDataLoader::from_window_matrix(...)` 将读取到的 `XltWinState` 数组转化为张量批次，直接调用 `pipeline.run(batch)` 获得全市场因子输出。
