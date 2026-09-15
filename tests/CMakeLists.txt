# 每个算子一组测试，一个可执行文件 + CTest 注册。
# 新增测试：往 test_sources 里加一项即可。
set(test_sources
    test_elementwise
    test_reduce
    test_shift
    test_rolling
    test_executor
    test_mean
    test_max
    test_min
    test_sort
    test_rank
)

foreach(t ${test_sources})
    add_executable(${t} ${t}.cpp)
    target_link_libraries(${t} PRIVATE flux)
    add_test(NAME ${t} COMMAND ${t})
endforeach()
