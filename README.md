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

### 6. 张量发布后不可变

计算图边上传递的是**共享的不可变张量**（`tensor::TensorPtr<T>`，即 `shared_ptr<const Tensor<T>>`）：一个节点的输出在整次执行中只物化一次，扇出 N 的下游共享同一份数据，不再各持一份副本。

安全性来自一条不变量：**张量一旦发布即不可变**。消费者拿到的若可变，扇出立刻退化成别名 bug——所以只提供 const 句柄，不提供可变版本。

这条不变量只约束**图执行层**。算子层看到的仍是最普通的张量值：`ops::*` 依旧是 `const Tensor<T>&` 进、`Tensor<T>` 出，引用语义不会渗进算子与张量层。

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

职责与当前状态：

| 职责 | 状态 |
|---|---|
| 执行顺序 | ✅ 拓扑排序（DFS 后序给出全序，Kahn 给出分层） |
| CPU 执行 | ✅ |
| 任务调度 | ✅ `run_parallel`：按依赖层并行，层内并行、层间屏障；线程池由调用方注入 |
| 同步 | ✅ `run` / `run_parallel` 阻塞到完成；`future.get()` 兼作层内屏障 |
| 异步执行 | ✅ `run_async`：提交即返回 `std::future`，计算在池的后台线程上跑 |

`run()` 按拓扑序**顺序**执行，是默认路径：Input 由外部注入数据，Operator
收集上游输出并调用绑定好的闭包，Output 零成本透传（与上游是同一个张量对象）。

三个入口按**两个正交维度**划分（等不等结果 / 图内要不要并发），不是互斥模式：

|  | 同步 | 异步 |
|---|---|---|
| 顺序执行 | `run()` | `run_async()` |
| 图内并行 | `run_parallel()` | ——（有意不提供） |

"异步 + 图内并行"不做，因为向同一个线程池嵌套提交**实测会死锁**（完成 0/4）；
而且两处并行叠加会超额订阅线程。需要"同时跑多组任务"时，每组内部顺序执行、
让多个任务在池里自然并发即可——实测 16 组 `sort` 上快 **5.10×**。

`run_parallel()` 是 **opt-in**，不是默认。原因见
[docs/04-计算图与运行时](docs/04-计算图与运行时.md) 第五节——实测并行的收益
完全取决于算子类型：计算受限的（`sort` 这类）能到 4.72×，而内存受限的
（逐元素）只有 0.77×，反而更慢。`Executor` 无法判断一个算子属于哪一类
（闭包是类型擦除的），所以这个决定只能交给调用方。

节点输出以共享句柄按 node id 存放，沿边传递只拷贝指针。`get_output` 返回持有
所有权的句柄，只要调用方还持有它，张量就一直有效——跨多次 `run()` 也成立。

### 5.4 Memory

负责管理计算过程中使用的内存。

职责与当前状态：

| 职责 | 状态 |
|---|---|
| CPU Memory | ✅ `AlignedAllocator`，64 字节对齐 |
| Memory Pool | ✅ `MemoryPool` + `PooledAllocator`，**是 `Tensor` 的默认分配器** |
| Memory Reuse | ✅ 复用同尺寸的空闲块（池化即复用）；带保留上限，超出归还 OS |
| Temporary Buffer | ❌ 未实现，且经调查判定不需要 |

池化不是可有可无的装饰：图执行每轮都重建中间张量，直连分配会落进 glibc 的
`mmap`/`munmap` 路径，实测每轮 2016 次缺页、占掉约 75% 的执行时间。池化把
缺页降到 0，扇出图 4.39 → 0.99 ms/轮。细节与三项已知代价见
[docs/02-数据抽象与内存](docs/02-数据抽象与内存.md) 第四节。

Temporary Buffer 判定不做：实测一次 `run()` 的峰值 RSS 就是理论值
（输入拷贝 + 全部中间张量），没有可回收的浪费；再降峰值只能改语义
（中间张量 run 后不可取），那是用户可见的行为变更，收益不明确。

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

节点通过 `Node::OpFn` 绑定可执行闭包，输入以共享句柄传入：

```cpp
g.bind_op(n_shift, [](const std::vector<tensor::TensorPtr<float>>& ins) {
    return ops::shift(*ins[0], /*offset=*/1);
});
```

标量参数（shift 偏移、窗口大小等）在绑定阶段按值捕获，闭包签名由此统一；算子本身仍是值语义，解引用后即可交给 `ops::*`。

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
| `include/` | 全部实现（本库是 header-only，`src/` 下仅有占位） |
| `tests/` | 正确性测试，注册进 CTest |
| `examples/` | 使用示例、量化验证 |
| `benchmarks/` | 性能测试，**不注册**进 CTest |
| `docs/` | 设计文档，见 [docs/README.md](docs/README.md) |

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
