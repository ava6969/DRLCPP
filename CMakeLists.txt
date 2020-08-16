cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

set(CMAKE_TOOLCHAIN_FILE ${CMAKE_CURRENT_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake
        CACHE STRING "Vcpkg toolchain file")


project(DRLCPP)

find_package(Torch REQUIRED)
find_package(CURL CONFIG REQUIRED)
find_package(jsoncpp CONFIG REQUIRED)



set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
include_directories(ale/src)
add_executable(DRLCPP main.cpp Experimental/Box.h Experimental/CartPoleEnv.h Experimental/Discrete.h algorithms/DQN.h algorithms/DQN.cpp Experimental/Env.h models/FCQ.h algorithms/NFQ.h algorithms/NFQ.cpp buffers/ReplayBuffer.h buffers/ReplayBuffer.cpp Experimental/Space.h utility/Strategy.h utility/utils.h buffers/PrioritizedReplayBuffer.cpp buffers/PrioritizedReplayBuffer.h)
target_link_libraries(DRLCPP "${TORCH_LIBRARIES}"  CURL::libcurl jsoncpp_lib)
set_property(TARGET DRLCPP PROPERTY CXX_STANDARD 17)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET example-app
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:example-app>)
endif (MSVC)
