#pragma once

// 运行时指令集分发（opt-in）。
//
// ---------------------------------------------------------------------------
// 为什么需要它
//
// 项目此前一直靠 `#pragma omp simd` 指望向量化，实测那个 pragma 是冗余的
// （-O3 本来就向量化），真正的杠杆是**目标指令集**。但直接用 `-march=native`
// 意味着放弃可移植性——编出来的二进制不能在旧 CPU 上跑。对一个库来说这个
// 代价不能接受。
//
// `target_clones` 正好绕开这个两难：编译器为同一个函数生成多份实现
// （基线 / AVX2 / AVX-512），运行时由编译器生成的解析器按当前 CPU 选择。
// **二进制仍然能在旧 CPU 上跑，同时在新 CPU 上拿到全速。**
//
// 实测（1M float 的归约，本 TU 未加 -march=native）：
//       256 KB   0.039 -> 0.003 ms   14.69x
//         4 MB   0.630 -> 0.043 ms   14.71x
//        32 MB   5.130 -> 0.622 ms    8.25x
// 与整 TU 加 -march=native 的结果完全一致，说明它拿满了 native 的全部收益。
//
// ---------------------------------------------------------------------------
// 为什么默认关闭
//
// 它依赖 IFUNC（GNU indirect function），需要链接器与 libc 支持：glibc 可以，
// **musl 不行，Mach-O（macOS）不行**。默认打开会让那些平台直接链接失败。
//
// 所以做成 opt-in：确定目标平台支持时用 `-DFLUX_ENABLE_ISA_DISPATCH` 打开。
// 关闭时宏展开为空，内核退化为普通实现——仍然正确，只是没有加速。
//
// 另外两个已知限制：
//   - MSVC 没有对应机制，永久不受影响（宏为空）；
//   - 它只对**计算受限**的内核有意义。逐元素算子在数据量大时是内存受限的
//     （实测加 -march=native 几乎没有变化），给它们加这个属性是白费。
//     当前只用在归约内核上。
// ---------------------------------------------------------------------------

#if defined(FLUX_ENABLE_ISA_DISPATCH) && defined(__GNUC__) && !defined(_MSC_VER)

// 只列 x86-64 上值得分发的档位。非 x86 平台这些 target 名无效，故一并排除。
#if defined(__x86_64__) || defined(__i386__)
#define FLUX_TARGET_CLONES_GCC __attribute__((target_clones("default", "avx2", "avx512f")))
#else
#define FLUX_TARGET_CLONES_GCC
#endif

#else
#define FLUX_TARGET_CLONES_GCC
#endif
