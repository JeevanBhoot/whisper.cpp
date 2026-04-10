set(TARGET whisper-cli)
add_executable(${TARGET} cli.cpp)

include(DefaultTargetOptions)

target_include_directories(${TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src)
target_link_libraries(${TARGET} PRIVATE common whisper whisper-cohere ${FFMPEG_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

install(TARGETS ${TARGET} RUNTIME)
