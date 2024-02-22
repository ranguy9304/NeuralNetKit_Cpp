add_library(loader "")

target_sources(loader
  PRIVATE
    loader.cpp
  PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/loader.hpp
  )
target_include_directories(loader
  PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}
  )