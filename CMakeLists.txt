cmake_minimum_required(VERSION 4.1)
project(itrack CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(itrack main.cu)

set_target_properties(itrack PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set(ZLIB_INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/zlib)
execute_process(COMMAND ${CMAKE_COMMAND} -S ${CMAKE_CURRENT_SOURCE_DIR}/external/zlib -B ${CMAKE_CURRENT_BINARY_DIR}/zlib-build COMMAND_ERROR_IS_FATAL ANY)
execute_process(COMMAND ${CMAKE_COMMAND} --build ${CMAKE_CURRENT_BINARY_DIR}/zlib-build --config Release COMMAND_ERROR_IS_FATAL ANY)
execute_process(COMMAND ${CMAKE_COMMAND} --install ${CMAKE_CURRENT_BINARY_DIR}/zlib-build --prefix ${ZLIB_INSTALL_DIR} COMMAND_ERROR_IS_FATAL ANY)
list(APPEND CMAKE_SYSTEM_PREFIX_PATH ${ZLIB_INSTALL_DIR})

find_package(ZLIB REQUIRED)

#add_subdirectory(external/stb)
#target_link_libraries(itrack PUBLIC png_static ZLIB::ZLIB)

target_include_directories(itrack PRIVATE external/stb)

add_custom_command( TARGET itrack POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        ${ZLIB_INSTALL_DIR}/bin/z.dll
        # $<TARGET_RUNTIME_DLLS:ZLIB>
        $<TARGET_FILE_DIR:itrack>
        COMMAND_EXPAND_LISTS)
