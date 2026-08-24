CAMBRICON_OUTPUT_DIR_NAME = "cambricon_output"

# csv file name
KERNEL_DETAILS_FILE_NAME = "kernel_details.csv"
KERNEL_STATISTIC_FILE_NAME = "kernel_statistic.csv"
OP_KERNEL_STATISTIC_FILE_NAME = "op_kernel_statistic.csv"
OPERATOR_DETAILS_FILE_NAME = "operator_details.csv"
OPERATOR_STATISTIC_FILE_NAME = "operator_statistic.csv"
MEMORY_RECORD_FILE_NAME = "memory_record.csv"
OPERATOR_MEMORY_FILE_NAME = "operator_memory.csv"
L2CACHE_FILE_NAME = "l2_cache.csv"
COMM_DETAILS_FILE_NAME = "communication_op_details.csv"
OP_PERFORMANCE_FILE_NAME = "op_performance_details.csv"

# EFFICIENCY_PID_BEGIN is same as kOpAnalysisSortIndexBegin defined in OpPerformance.cpp
EFFICIENCY_PID_BEGIN = 10001024

# used to calculate tensor size for triton kernel num gb
TYPE_SIZE_MAP = {
    "c10::BFloat16": 2,
    "c10::Half": 2,
    "float": 4,
    "int": 4,
    "double": 8,
    "long int": 8,
    "bool": 1,
}
