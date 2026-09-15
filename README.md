# FluxCompute

FluxCompute 是一个面向高性能计算的、与具体业务解耦的通用计算库。

项目的目标不是实现一个固定业务的量化框架，而是从底层计算出发，逐步实现：

- 数据抽象
- 基础计算算子
- CPU 执行
- 内存管理
- Runtime
- 计算图
- 算子优化

上层应用可以根据自己的需求组合这些基础能力，构建不同的计算任务。

> **当前范围**：仅实现 CPU Backend（含 SIMD / 多线程优化路径），不引入 CUDA / GPU。
> 项目聚焦于把 CPU 上的计算框架本身做扎实。

---

## 一、项目背景

为了学习理解因子计算、回测等量化业务。

在学习这些业务的过程中，可以发现其中存在大量重复的数值计算，例如：

- 加减乘除
- 求和 / 求均值
- 排序 / Rank
- Rolling Window
- Reduce
- Scan
- 矩阵运算

这些计算本身并不属于某一个具体业务。

因此，本项目希望从这些通用计算能力出发，构建一个独立的计算库。

量化中的：

- 因子计算
- 回测
- 参数搜索

作为项目早期的重要应用场景，用于验证框架的正确性、易用性和性能。

---

## 二、项目定位

FluxCompute 的核心思想：

> **底层提供计算能力，上层负责组合计算逻辑。**

例如底层提供：

```text
add
sub
mul
div

sum
mean
max
min

shift
rolling_mean
rolling_std

sort
rank
reduce
scan
```

上层可以组合这些算子。

例如一个量化因子可以表示为：

```text
原始数据
   ↓
shift
   ↓
divide
   ↓
rolling_mean
   ↓
rank
   ↓
Factor
```

回测也可以使用相同的基础计算能力：

```text
行情数据
   ↓
return
   ↓
signal
   ↓
position
   ↓
PnL
   ↓
回测结果
```

对于 FluxCompute 来说，它们都只是计算任务。

**FluxCompute 本身不需要理解"因子""策略""订单""持仓"等业务概念。**

---

## 三、核心设计原则

### 1. 业务无关

核心库不绑定：

- 量化
- 因子
- 回测
- 股票
- 期货
- 策略
- 订单
- 持仓

这些内容属于上层应用。

### 2. 上层负责组合

FluxCompute 提供基础计算能力。

上层根据业务需求组合 Operator。

例如：

```text
Operator A
    ↓
Operator B
    ↓
Operator C
    ↓
Result
```

复杂的业务逻辑由上层构建，而不是写入底层计算库。

### 3. 实现层解耦

算子与具体的内核实现解耦：

```text
           Operator
              │
              ↓
         CPU Backend
              │
              ↓
      CPU + SIMD / 多线程
```

核心计算逻辑不应绑定某个具体实现。

后续可以在**不改动算子接口**的前提下优化内核：

- SIMD 向量化
- 多线程并行
- 内存布局 / 缓存友好

### 4. 先正确，再优化

项目开发遵循：

```text
功能正确
   ↓
建立测试
   ↓
性能测试
   ↓
定位瓶颈
   ↓
优化
```

不在没有 Benchmark 的情况下盲目优化。

### 5. 聚焦 CPU

项目初期**不引入 CUDA / GPU**。

原因：

1. 训练/验证环境的显存资源有限，写出来的 GPU 代码无法实际使用；
2. CPU 计算框架本身（数据抽象、算子、Runtime、计算图）仍有大量可深入的内容。

因此所有精力集中在：

- 把 CPU 执行做正确、做扎实；
- 研究 SIMD / 多线程对性能的影响。

---

## 四、总体架构

项目初期架构：

```text
                    上层应用
                       │
             ┌─────────┴─────────┐
             ↓                   ↓
          因子计算              回测
             │                   │
             └─────────┬─────────┘
                       ↓
                 FluxCompute
                       │
              ┌────────┴────────┐
              ↓                 ↓
           Operator          Runtime
              │                 │
              └────────┬────────┘
                       ↓
                    Backend
                       ↓
                      CPU
```

随着项目发展，再逐渐加入：

```text
                    Runtime
                       │
              ┌────────┼────────┐
              ↓        ↓        ↓
          Scheduler  Memory  Executor
                       │
                       ↓
                 Computation Graph
                       │
                       ↓
                 Operator Fusion
```

---

## 五、核心模块

### 5.1 Tensor

Tensor 是计算数据的基础抽象。

负责描述：

- 数据
- Shape
- DType
- Layout
- Memory

例如：

```cpp
Tensor<float> a;
Tensor<float> b;

auto c = add(a, b);
```

Tensor 本身不包含任何业务逻辑。

### 5.2 Operator

Operator 是最基础的计算单元。

第一阶段计划实现：

```text
add
sub
mul
div

sum
mean
max
min
```

之后逐步增加：

```text
shift
rolling
sort
rank
scan
reduce
matrix operations
```

Operator 可以根据 Backend 使用不同实现。

### 5.3 Runtime

Runtime 负责执行计算。

未来负责：

- 任务调度
- 执行顺序
- CPU 执行
- 异步执行
- 同步

初期只实现最简单的 CPU Runtime。

### 5.4 Memory

负责管理计算过程中使用的内存。

后续逐步支持：

- CPU Memory
- Memory Pool
- Temporary Buffer
- Memory Reuse

### 5.5 Computation Graph

当基础 Operator 足够完善以后，将多个 Operator 组织成计算图。

例如：

```text
A ─────┐
       ↓
      Add
       ↓
      Mul ←──── B
       ↓
      Mean
       ↓
    Result
```

计算图可以为后续的：

- 依赖分析
- 执行调度
- 内存复用
- 算子融合
- 执行优化

提供基础。

---

## 六、项目目录

第一阶段：

```text
FluxCompute/
├── CMakeLists.txt
├── README.md
│
├── include/
│   └── flux/
│
├── src/
│
├── tests/
│
├── examples/
│
├── benchmarks/
│
└── docs/
```

### 目录说明

| 目录 | 作用 |
|---|---|
| `include/` | 对外暴露的头文件 |
| `src/` | 核心实现 |
| `tests/` | 正确性测试 |
| `examples/` | 使用示例、量化验证 |
| `benchmarks/` | 性能测试 |
| `docs/` | 设计文档 |

其中：

```text
tests/
    ↓
"算得对不对？"

benchmarks/
    ↓
"算得快不快？"
```

---

## 七、应用验证

FluxCompute 的第一个主要应用场景是量化计算。

可以逐渐建立：

```text
examples/
└── quant/
    ├── factor/
    └── backtest/
```

用于验证：

### 因子计算

```text
原始行情
    ↓
Tensor
    ↓
基础 Operator
    ↓
组合计算
    ↓
Factor
```

### 回测计算

```text
Historical Data
       ↓
Signal
       ↓
Position
       ↓
Return
       ↓
PnL
       ↓
Statistics
```

量化应用是验证层，而不是 FluxCompute Core 的组成部分。

---

## 八、最终目标

FluxCompute 最终希望形成一个这样的计算基础：

```text
                         上层应用
                            │
              ┌─────────────┼─────────────┐
              ↓             ↓             ↓
            因子           回测          其他
              │             │             │
              └─────────────┼─────────────┘
                            ↓
                       FluxCompute
                            │
              ┌─────────────┼─────────────┐
              ↓             ↓             ↓
           Tensor        Operator       Runtime
                            │
                            ↓
                           CPU
                            │
                            ↓
                           SIMD
```
