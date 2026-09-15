# 基准测试：只产出可执行文件，**不注册进 CTest**。
# 它们不含断言，跑 ctest 时不该被带上（也不该让 CI 因为机器快慢而红）。
#
# 运行方式：
#   cmake --build build && ./build/benchmarks/bench_executor
#   cmake --build build && ./build/benchmarks/bench_memory
#   cmake --build build && ./build/benchmarks/bench_elementwise
#   cmake --build build && ./build/benchmarks/bench_elementwise_noomp   # 对照
#
# 注意：务必用 Release 构建（根 CMakeLists 默认就是）。Debug 下的数字没有意义。

add_executable(bench_executor bench_executor.cpp)
target_link_libraries(bench_executor PRIVATE flux)

add_executable(bench_memory bench_memory.cpp)
target_link_libraries(bench_memory PRIVATE flux)

# 同一个源文件编两份：一份带 -fopenmp，一份不带。
# 两者的差就是 `#pragma omp simd` 的净效果——见 bench_elementwise.cpp 顶部说明。
add_executable(bench_parallel bench_parallel.cpp)
target_link_libraries(bench_parallel PRIVATE flux)

add_executable(bench_elementwise bench_elementwise.cpp)
target_link_libraries(bench_elementwise PRIVATE flux)

add_executable(bench_elementwise_noomp bench_elementwise.cpp)
target_link_libraries(bench_elementwise_noomp PRIVATE flux_no_omp)
