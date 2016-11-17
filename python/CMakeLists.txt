cmake_minimum_required( VERSION 2.8 )

## Sources and headers
include_directories( ${CMAKE_CURRENT_SOURCE_DIR} )

set(CMAKE_SWIG_OUTDIR ${PROJECT_SOURCE_DIR}/swig)
find_package(SWIG REQUIRED)
include(${SWIG_USE_FILE})

find_package(PythonLibs)
include_directories(${PYTHON_INCLUDE_PATH} swig)
set(CMAKE_SWIG_FLAGS "-threads")

# function for setting source properties
function(swig_cpp files)
  foreach(f ${files})
    message("swig src properties " ${f})
    set_source_files_properties(${f} PROPERTIES CPLUSPLUS ON)
    #    set_source_files_properties(${f} PROPERTIES SWIG_FLAGS "-includeall")
  endforeach(f)
endfunction(swig_cpp)

set(SWIG_FILES gast.i pygast.cpp)
swig_cpp("${SWIG_FILES}")

swig_add_module(pygast python gast.i pygast.cpp)
swig_link_libraries(pygast ${PYTHON_LIBRARIES} gast ${CUDA_LIBRARIES} ${OpenCV_LIBRARIES})

install(TARGETS _pygast
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}/python2.7/dist-packages/)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/swig/
  DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}/python2.7/dist-packages/
  FILES_MATCHING PATTERN "*.py")
