cmake_minimum_required(VERSION 3.10)

project(fluxcompute VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 默认 Release 构建（正确性测试由 CTest 保证，性能默认开优化）
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# 告警
if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra)
endif()

# ------------------------------------------------------------------
# 核心库：header-only，暴露给使用者的接口目标
# ------------------------------------------------------------------
add_library(flux INTERFACE)
target_include_directories(flux INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(flux INTERFACE cxx_std_17)

# OpenMP：可选。提供 SIMD / 并行编译指示；
# 未找到时 `#pragma omp simd` 等会被编译器忽略，不影响正确性。 
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(flux INTERFACE OpenMP::OpenMP_CXX)
    message(STATUS "FluxCompute: OpenMP enabled")
else()
    message(STATUS "FluxCompute: OpenMP not found (simd pragmas become no-ops)")
endif()

# ------------------------------------------------------------------
# 测试与示例
# ------------------------------------------------------------------
enable_testing()
add_subdirectory(tests)
add_subdirectory(examples)
