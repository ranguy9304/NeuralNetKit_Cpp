
find_package( OpenCV REQUIRED )
find_package(Eigen3 REQUIRED)
# find_package (Eigen3 3.3 REQUIRED NO_MODULE)
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories( ${OpenCV_INCLUDE_DIRS} )
# include_directories(${Eigen3_INCLUDE_DIRS})


add_executable( NNfs main.cpp )

target_link_libraries( NNfs ${OpenCV_LIBS} )
target_link_libraries(NNfs stdc++fs)
target_link_libraries(NNfs "${CMAKE_THREAD_LIBS_INIT}")
target_link_libraries (NNfs Eigen3::Eigen)
target_link_libraries(NNfs m)
add_subdirectory(loader)
add_subdirectory(model)
target_link_libraries(NNfs
    loader
    model
  )

   


  