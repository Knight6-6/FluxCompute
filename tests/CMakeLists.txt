# 每个算子一组测试，一个可执行文件 + CTest 注册。
# 新增测试：往 test_sources 里加一项即可。
set(test_sources
    test_elementwise
    test_map
    test_matmul
    test_reduce
    test_scan
    test_sum
    test_shift
    test_rolling
    test_executor
    test_mean
    test_max
    test_min
    test_sort
    test_rank
    test_memory
    test_tensor
)

foreach(t ${test_sources})
    add_executable(${t} ${t}.cpp)
    target_link_libraries(${t} PRIVATE flux)
    add_test(NAME ${t} COMMAND ${t})
endforeach()

# test_memory 会主动探测"分配失败"这条路径。ASan 默认对超大分配直接 abort，
# 库自己的 bad_alloc 分支根本走不到；让它返回 NULL 才能覆盖到。
# 非消毒器构建下这个环境变量不起作用，无副作用。
set_tests_properties(test_memory PROPERTIES
    ENVIRONMENT "ASAN_OPTIONS=allocator_may_return_null=1")
