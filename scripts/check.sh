#!/usr/bin/env bash
#
# 跑全部四种构建 + 测试。提交前用。
#
#     ./scripts/check.sh            # 全部
#     ./scripts/check.sh release    # 只跑某一个
#
# 构建目录统一在 build/ 下按预设名分层（见 CMakePresets.json）：
#     build/release  build/isa  build/san  build/tsan
#
# 两处刻意的写法，都是为了不重蹈 docs/05 第 3.6 节的覆辙
# （构建失败时测试跑的是旧二进制，却报全绿）：
#   1. 用命令的**退出码**判断成败，而不是 grep 构建输出——编译错误走 stderr，
#      只看 stdout 会漏；
#   2. 构建一失败就立刻退出，绝不让 ctest 去跑上一次的旧二进制。

set -uo pipefail
cd "$(dirname "$0")/.."

PRESETS=${1:-"release isa san tsan"}
JOBS=$(nproc 2>/dev/null || echo 4)
FAILED=0

for preset in $PRESETS; do
    echo "=== $preset ==="

    if ! out=$(cmake --preset "$preset" 2>&1); then
        echo "  配置失败"; echo "$out" | tail -15; FAILED=1; continue
    fi

    if ! out=$(cmake --build --preset "$preset" -j"$JOBS" 2>&1); then
        echo "  构建失败"; echo "$out" | grep -iE "error" | head -15; FAILED=1; continue
    fi

    # 项目开 -Wall -Wextra，**自家代码**的告警按失败处理。
    #
    # 只看本仓库路径下的告警，而不是一有 warning 就失败：GCC 13 在 -O1 +
    # sanitizer 下会对 libstdc++ 的 memmove 报 -Warray-bounds 误报
    # （stl_algobase.h:437，"offset 8 out of bounds [0, 8]" —— 把半开区间
    # 当成了闭区间），触发者是 Shape 构造里那句 dimensions_(dimensions)。
    # 为一条系统头的误报关掉 -Warray-bounds 是拿真问题的可见性换安静，
    # 按路径区分才是对的。
    ours=$(echo "$out" | grep -i "warning" | grep -F "$PWD/" || true)
    if [ -n "$ours" ]; then
        echo "  自家代码有告警"; echo "$ours" | head -10; FAILED=1; continue
    fi
    n_sys=$(echo "$out" | grep -ci "warning" || true)
    if [ "${n_sys:-0}" -gt 0 ]; then
        echo "  （另有 $n_sys 条来自系统头的告警，已知误报，见 docs/05 第 3.8 节）"
    fi

    if [ "$preset" = "tsan" ]; then
        # TSan 在本机需要 setarch -R：高 ASLR 熵会让它报 unexpected memory
        # mapping 直接退出、所有测试都失败（见 docs/05 第 3.8 节）。
        # allocator_may_return_null=1：test_memory 会主动探测分配失败路径。
        arch=$(uname -m)
        tested=0; failed_tests=0
        for t in build/tsan/tests/test_*; do
            [ -x "$t" ] || continue
            tested=$((tested + 1))
            if ! o=$(TSAN_OPTIONS=allocator_may_return_null=1 setarch "$arch" -R "$t" 2>&1); then
                failed_tests=$((failed_tests + 1))
                echo "  ❌ $(basename "$t")"
                echo "$o" | grep -iE "WARNING: ThreadSanitizer|data race" | head -5
            fi
        done
        if [ $failed_tests -eq 0 ]; then
            echo "  ✅ $tested 组测试，零数据竞争"
        else
            FAILED=1
        fi
    else
        if ! out=$(ctest --test-dir "build/$preset" 2>&1); then
            echo "  测试失败"; echo "$out" | tail -20; FAILED=1; continue
        fi
        echo "  ✅ $(echo "$out" | grep -oE '[0-9]+% tests passed[^,]*')"
    fi
done

echo
[ $FAILED -eq 0 ] && echo "全部通过" || echo "有失败项"
exit $FAILED
