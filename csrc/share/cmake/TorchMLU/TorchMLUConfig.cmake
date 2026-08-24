# FindTorchMLU
# -------
#
# Finds the Torch MLU library
#
#  TORCH_MLU_LIBRARIES
#  TORCH_MLU_INCLUDE_DIRS
#  TORCH_MLU_LIBRARY_DIRS
#

include(FindPackageHandleStandardArgs)

get_filename_component(CMAKE_CURRENT_LIST_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/modules $ENV{NEUWARE_HOME}/cmake/modules)

if(DEFINED ENV{TORCH_MLU_INSTALL_PREFIX})
  set(TORCH_MLU_INSTALL_PREFIX $ENV{TORCH_MLU_INSTALL_PREFIX})
else()
  # Assume we are in <install-prefix>/share/cmake/TorchMLU/TorchConfig.cmake
  get_filename_component(TORCH_MLU_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)
endif()

set(TORCH_MLU_INCLUDE_DIRS ${TORCH_MLU_INSTALL_PREFIX}/include)
list(APPEND TORCH_MLU_INCLUDE_DIRS ${TORCH_MLU_INSTALL_PREFIX}/include/api/include)
set(TORCH_MLU_LIBRARY_DIRS ${TORCH_MLU_INSTALL_PREFIX}/lib)
set(TORCH_MLU_LIBRARIES "")

# Find cnrt header files and libs
find_package(CNRT)
if (CNRT_FOUND)
    list(APPEND TORCH_MLU_INCLUDE_DIRS ${CNRT_INCLUDE_DIRS})
    list(APPEND TORCH_MLU_LIBRARIES ${CNRT_LIBRARIES})
endif()

# Find cnnl header files and libs
find_package(CNNL)
if (CNNL_FOUND)
    list(APPEND TORCH_MLU_INCLUDE_DIRS ${CNNL_INCLUDE_DIRS})
    list(APPEND TORCH_MLU_LIBRARIES ${CNNL_LIBRARIES})
endif()

# Find cndrv header files and libs
find_package(CNDRV)
if (CNDRV_FOUND)
    list(APPEND TORCH_MLU_INCLUDE_DIRS ${CNDRV_INCLUDE_DIRS})
    list(APPEND TORCH_MLU_LIBRARIES ${CNDRV_LIBRARIES})
endif()

find_library(TORCH_MLU_LIBRARY torch_mlu PATHS "${TORCH_MLU_LIBRARY_DIRS}")
find_package_handle_standard_args(TorchMLU DEFAULT_MSG TORCH_MLU_LIBRARY)
list(APPEND TORCH_MLU_LIBRARIES ${TORCH_MLU_LIBRARY})

include_directories(${TORCH_MLU_INCLUDE_DIRS})
include_directories($ENV{NEUWARE_HOME}/lib/clang/*/include)
link_directories(${TORCH_MLU_LIBRARY_DIRS})
link_directories($ENV{NEUWARE_HOME}/lib)
include_directories($ENV{NEUWARE_HOME}/include)
link_directories($ENV{NEUWARE_HOME}/lib64)
