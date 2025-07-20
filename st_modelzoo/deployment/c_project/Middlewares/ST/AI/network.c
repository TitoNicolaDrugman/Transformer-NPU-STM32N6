/**
  ******************************************************************************
  * @file    network.c
  * @author  AST Embedded Analytics Research Platform
  * @date    2025-06-26T14:13:18+0200
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2025 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */


#include "network.h"
#include "network_data.h"

#include "ai_platform.h"
#include "ai_platform_interface.h"
#include "ai_math_helpers.h"

#include "core_common.h"
#include "core_convert.h"

#include "layers.h"



#undef AI_NET_OBJ_INSTANCE
#define AI_NET_OBJ_INSTANCE g_network
 
#undef AI_NETWORK_MODEL_SIGNATURE
#define AI_NETWORK_MODEL_SIGNATURE     "0xc19f437a9ce64d694a0e9ef7a74b2a2c"

#ifndef AI_TOOLS_REVISION_ID
#define AI_TOOLS_REVISION_ID     ""
#endif

#undef AI_TOOLS_DATE_TIME
#define AI_TOOLS_DATE_TIME   "2025-06-26T14:13:18+0200"

#undef AI_TOOLS_COMPILE_TIME
#define AI_TOOLS_COMPILE_TIME    __DATE__ " " __TIME__

#undef AI_NETWORK_N_BATCHES
#define AI_NETWORK_N_BATCHES         (1)

static ai_ptr g_network_activations_map[1] = AI_C_ARRAY_INIT;
static ai_ptr g_network_weights_map[1] = AI_C_ARRAY_INIT;



/**  Array declarations section  **********************************************/
/* Array#0 */
AI_ARRAY_OBJ_DECLARE(
  gemm_292_bias_0_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)

/* Array#1 */
AI_ARRAY_OBJ_DECLARE(
  gemm_284_bias_0_2_gemm_284_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)

/* Array#2 */
AI_ARRAY_OBJ_DECLARE(
  gemm_205_bias_0_2_gemm_205_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)

/* Array#3 */
AI_ARRAY_OBJ_DECLARE(
  gemm_197_bias_0_2_gemm_197_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)

/* Array#4 */
AI_ARRAY_OBJ_DECLARE(
  gemm_118_bias_0_2_gemm_118_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)

/* Array#5 */
AI_ARRAY_OBJ_DECLARE(
  gemm_110_bias_0_2_gemm_110_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)

/* Array#6 */
AI_ARRAY_OBJ_DECLARE(
  gemm_31_bias_0_2_gemm_31_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)

/* Array#7 */
AI_ARRAY_OBJ_DECLARE(
  gemm_23_bias_0_2_gemm_23_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)

/* Array#8 */
AI_ARRAY_OBJ_DECLARE(
  serving_default_embedded_tokens0_output_array, AI_ARRAY_FORMAT_S8|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 25600, AI_STATIC)

/* Array#9 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#10 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_1_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#11 */
AI_ARRAY_OBJ_DECLARE(
  gemm_26_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#12 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_28_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#13 */
AI_ARRAY_OBJ_DECLARE(
  transpose_30_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#14 */
AI_ARRAY_OBJ_DECLARE(
  transpose_30_0_1_gemm_31_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#15 */
AI_ARRAY_OBJ_DECLARE(
  gemm_18_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#16 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_20_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#17 */
AI_ARRAY_OBJ_DECLARE(
  transpose_22_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#18 */
AI_ARRAY_OBJ_DECLARE(
  transpose_22_0_0_gemm_23_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#19 */
AI_ARRAY_OBJ_DECLARE(
  gemm_13_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#20 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_15_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#21 */
AI_ARRAY_OBJ_DECLARE(
  transpose_bgemm_23_out_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#22 */
AI_ARRAY_OBJ_DECLARE(
  transpose_bgemm_23_out_0_1_gemm_23_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#23 */
AI_ARRAY_OBJ_DECLARE(
  gemm_23_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 40000, AI_STATIC)

/* Array#24 */
AI_ARRAY_OBJ_DECLARE(
  gemm_23_0_0_eltwise_24_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 40000, AI_STATIC)

/* Array#25 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_24_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 40000, AI_STATIC)

/* Array#26 */
AI_ARRAY_OBJ_DECLARE(
  nl_25_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 40000, AI_STATIC)

/* Array#27 */
AI_ARRAY_OBJ_DECLARE(
  nl_25_0_0_gemm_31_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 40000, AI_STATIC)

/* Array#28 */
AI_ARRAY_OBJ_DECLARE(
  gemm_31_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#29 */
AI_ARRAY_OBJ_DECLARE(
  gemm_31_0_0_transpose_32_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#30 */
AI_ARRAY_OBJ_DECLARE(
  transpose_32_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#31 */
AI_ARRAY_OBJ_DECLARE(
  gemm_42_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#32 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_44_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#33 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_45_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#34 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_45_0_0_reduce_46_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#35 */
AI_ARRAY_OBJ_DECLARE(
  reduce_46_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#36 */
AI_ARRAY_OBJ_DECLARE(
  reduce_46_Mul_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#37 */
AI_ARRAY_OBJ_DECLARE(
  reduce_46_Mul_0_1_eltwise_47_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 100, AI_STATIC)

/* Array#38 */
AI_ARRAY_OBJ_DECLARE(
  reduce_46_Mul_0_1_eltwise_47_conversion_0_1_eltwise_47_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#39 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_47_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#40 */
AI_ARRAY_OBJ_DECLARE(
  reduce_48_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#41 */
AI_ARRAY_OBJ_DECLARE(
  reduce_48_Mul_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#42 */
AI_ARRAY_OBJ_DECLARE(
  reduce_48_Mul_0_0_eltwise_49_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 100, AI_STATIC)

/* Array#43 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_49_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 100, AI_STATIC)

/* Array#44 */
AI_ARRAY_OBJ_DECLARE(
  nl_50_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 100, AI_STATIC)

/* Array#45 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_51_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#46 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_53_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#47 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_54_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#48 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_52_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#49 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_55_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#50 */
AI_ARRAY_OBJ_DECLARE(
  gemm_64_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 51200, AI_STATIC)

/* Array#51 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_66_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 51200, AI_STATIC)

/* Array#52 */
AI_ARRAY_OBJ_DECLARE(
  gemm_75_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#53 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_77_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#54 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_78_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#55 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_78_0_0_reduce_79_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#56 */
AI_ARRAY_OBJ_DECLARE(
  reduce_79_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#57 */
AI_ARRAY_OBJ_DECLARE(
  reduce_79_Mul_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#58 */
AI_ARRAY_OBJ_DECLARE(
  reduce_79_Mul_0_1_eltwise_80_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 100, AI_STATIC)

/* Array#59 */
AI_ARRAY_OBJ_DECLARE(
  reduce_79_Mul_0_1_eltwise_80_conversion_0_1_eltwise_80_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#60 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_80_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#61 */
AI_ARRAY_OBJ_DECLARE(
  reduce_81_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#62 */
AI_ARRAY_OBJ_DECLARE(
  reduce_81_Mul_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#63 */
AI_ARRAY_OBJ_DECLARE(
  reduce_81_Mul_0_0_eltwise_82_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 100, AI_STATIC)

/* Array#64 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_82_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 100, AI_STATIC)

/* Array#65 */
AI_ARRAY_OBJ_DECLARE(
  nl_83_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 100, AI_STATIC)

/* Array#66 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_84_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#67 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_86_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#68 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_87_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#69 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_85_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#70 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_88_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#71 */
AI_ARRAY_OBJ_DECLARE(
  gemm_113_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#72 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_115_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#73 */
AI_ARRAY_OBJ_DECLARE(
  transpose_117_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#74 */
AI_ARRAY_OBJ_DECLARE(
  transpose_117_0_1_gemm_118_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#75 */
AI_ARRAY_OBJ_DECLARE(
  gemm_105_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#76 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_107_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#77 */
AI_ARRAY_OBJ_DECLARE(
  transpose_109_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#78 */
AI_ARRAY_OBJ_DECLARE(
  transpose_109_0_0_gemm_110_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#79 */
AI_ARRAY_OBJ_DECLARE(
  gemm_100_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#80 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_102_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#81 */
AI_ARRAY_OBJ_DECLARE(
  transpose_bgemm_110_out_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#82 */
AI_ARRAY_OBJ_DECLARE(
  transpose_bgemm_110_out_0_1_gemm_110_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#83 */
AI_ARRAY_OBJ_DECLARE(
  gemm_110_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 40000, AI_STATIC)

/* Array#84 */
AI_ARRAY_OBJ_DECLARE(
  gemm_110_0_0_eltwise_111_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 40000, AI_STATIC)

/* Array#85 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_111_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 40000, AI_STATIC)

/* Array#86 */
AI_ARRAY_OBJ_DECLARE(
  nl_112_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 40000, AI_STATIC)

/* Array#87 */
AI_ARRAY_OBJ_DECLARE(
  nl_112_0_0_gemm_118_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 40000, AI_STATIC)

/* Array#88 */
AI_ARRAY_OBJ_DECLARE(
  gemm_118_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#89 */
AI_ARRAY_OBJ_DECLARE(
  gemm_118_0_0_transpose_119_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#90 */
AI_ARRAY_OBJ_DECLARE(
  transpose_119_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#91 */
AI_ARRAY_OBJ_DECLARE(
  gemm_129_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#92 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_131_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#93 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_132_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#94 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_132_0_0_reduce_133_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#95 */
AI_ARRAY_OBJ_DECLARE(
  reduce_133_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#96 */
AI_ARRAY_OBJ_DECLARE(
  reduce_133_Mul_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#97 */
AI_ARRAY_OBJ_DECLARE(
  reduce_133_Mul_0_1_eltwise_134_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 100, AI_STATIC)

/* Array#98 */
AI_ARRAY_OBJ_DECLARE(
  reduce_133_Mul_0_1_eltwise_134_conversion_0_1_eltwise_134_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#99 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_134_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#100 */
AI_ARRAY_OBJ_DECLARE(
  reduce_135_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#101 */
AI_ARRAY_OBJ_DECLARE(
  reduce_135_Mul_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#102 */
AI_ARRAY_OBJ_DECLARE(
  reduce_135_Mul_0_0_eltwise_136_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 100, AI_STATIC)

/* Array#103 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_136_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 100, AI_STATIC)

/* Array#104 */
AI_ARRAY_OBJ_DECLARE(
  nl_137_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 100, AI_STATIC)

/* Array#105 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_138_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#106 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_140_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#107 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_141_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#108 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_139_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#109 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_142_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#110 */
AI_ARRAY_OBJ_DECLARE(
  gemm_151_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 51200, AI_STATIC)

/* Array#111 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_153_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 51200, AI_STATIC)

/* Array#112 */
AI_ARRAY_OBJ_DECLARE(
  gemm_162_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#113 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_164_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#114 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_165_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#115 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_165_0_0_reduce_166_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#116 */
AI_ARRAY_OBJ_DECLARE(
  reduce_166_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#117 */
AI_ARRAY_OBJ_DECLARE(
  reduce_166_Mul_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#118 */
AI_ARRAY_OBJ_DECLARE(
  reduce_166_Mul_0_1_eltwise_167_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 100, AI_STATIC)

/* Array#119 */
AI_ARRAY_OBJ_DECLARE(
  reduce_166_Mul_0_1_eltwise_167_conversion_0_1_eltwise_167_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#120 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_167_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#121 */
AI_ARRAY_OBJ_DECLARE(
  reduce_168_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#122 */
AI_ARRAY_OBJ_DECLARE(
  reduce_168_Mul_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#123 */
AI_ARRAY_OBJ_DECLARE(
  reduce_168_Mul_0_0_eltwise_169_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 100, AI_STATIC)

/* Array#124 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_169_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 100, AI_STATIC)

/* Array#125 */
AI_ARRAY_OBJ_DECLARE(
  nl_170_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 100, AI_STATIC)

/* Array#126 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_171_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#127 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_173_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#128 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_174_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#129 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_172_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#130 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_175_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#131 */
AI_ARRAY_OBJ_DECLARE(
  gemm_200_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#132 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_202_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#133 */
AI_ARRAY_OBJ_DECLARE(
  transpose_204_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#134 */
AI_ARRAY_OBJ_DECLARE(
  transpose_204_0_1_gemm_205_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#135 */
AI_ARRAY_OBJ_DECLARE(
  gemm_192_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#136 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_194_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#137 */
AI_ARRAY_OBJ_DECLARE(
  transpose_196_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#138 */
AI_ARRAY_OBJ_DECLARE(
  transpose_196_0_0_gemm_197_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#139 */
AI_ARRAY_OBJ_DECLARE(
  gemm_187_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#140 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_189_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#141 */
AI_ARRAY_OBJ_DECLARE(
  transpose_bgemm_197_out_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#142 */
AI_ARRAY_OBJ_DECLARE(
  transpose_bgemm_197_out_0_1_gemm_197_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#143 */
AI_ARRAY_OBJ_DECLARE(
  gemm_197_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 40000, AI_STATIC)

/* Array#144 */
AI_ARRAY_OBJ_DECLARE(
  gemm_197_0_0_eltwise_198_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 40000, AI_STATIC)

/* Array#145 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_198_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 40000, AI_STATIC)

/* Array#146 */
AI_ARRAY_OBJ_DECLARE(
  nl_199_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 40000, AI_STATIC)

/* Array#147 */
AI_ARRAY_OBJ_DECLARE(
  nl_199_0_0_gemm_205_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 40000, AI_STATIC)

/* Array#148 */
AI_ARRAY_OBJ_DECLARE(
  gemm_205_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#149 */
AI_ARRAY_OBJ_DECLARE(
  gemm_205_0_0_transpose_206_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#150 */
AI_ARRAY_OBJ_DECLARE(
  transpose_206_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#151 */
AI_ARRAY_OBJ_DECLARE(
  gemm_216_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#152 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_218_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#153 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_219_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#154 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_219_0_0_reduce_220_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#155 */
AI_ARRAY_OBJ_DECLARE(
  reduce_220_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#156 */
AI_ARRAY_OBJ_DECLARE(
  reduce_220_Mul_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#157 */
AI_ARRAY_OBJ_DECLARE(
  reduce_220_Mul_0_1_eltwise_221_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 100, AI_STATIC)

/* Array#158 */
AI_ARRAY_OBJ_DECLARE(
  reduce_220_Mul_0_1_eltwise_221_conversion_0_1_eltwise_221_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#159 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_221_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#160 */
AI_ARRAY_OBJ_DECLARE(
  reduce_222_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#161 */
AI_ARRAY_OBJ_DECLARE(
  reduce_222_Mul_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#162 */
AI_ARRAY_OBJ_DECLARE(
  reduce_222_Mul_0_0_eltwise_223_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 100, AI_STATIC)

/* Array#163 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_223_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 100, AI_STATIC)

/* Array#164 */
AI_ARRAY_OBJ_DECLARE(
  nl_224_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 100, AI_STATIC)

/* Array#165 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_225_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#166 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_227_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#167 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_228_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#168 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_226_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#169 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_229_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#170 */
AI_ARRAY_OBJ_DECLARE(
  gemm_238_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 51200, AI_STATIC)

/* Array#171 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_240_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 51200, AI_STATIC)

/* Array#172 */
AI_ARRAY_OBJ_DECLARE(
  gemm_249_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#173 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_251_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#174 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_252_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#175 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_252_0_0_reduce_253_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#176 */
AI_ARRAY_OBJ_DECLARE(
  reduce_253_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#177 */
AI_ARRAY_OBJ_DECLARE(
  reduce_253_Mul_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#178 */
AI_ARRAY_OBJ_DECLARE(
  reduce_253_Mul_0_1_eltwise_254_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 100, AI_STATIC)

/* Array#179 */
AI_ARRAY_OBJ_DECLARE(
  reduce_253_Mul_0_1_eltwise_254_conversion_0_1_eltwise_254_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#180 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_254_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#181 */
AI_ARRAY_OBJ_DECLARE(
  reduce_255_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#182 */
AI_ARRAY_OBJ_DECLARE(
  reduce_255_Mul_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#183 */
AI_ARRAY_OBJ_DECLARE(
  reduce_255_Mul_0_0_eltwise_256_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 100, AI_STATIC)

/* Array#184 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_256_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 100, AI_STATIC)

/* Array#185 */
AI_ARRAY_OBJ_DECLARE(
  nl_257_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 100, AI_STATIC)

/* Array#186 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_258_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#187 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_260_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#188 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_261_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#189 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_259_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#190 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_262_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#191 */
AI_ARRAY_OBJ_DECLARE(
  gemm_287_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#192 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_289_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#193 */
AI_ARRAY_OBJ_DECLARE(
  transpose_291_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#194 */
AI_ARRAY_OBJ_DECLARE(
  transpose_291_0_1_gemm_292_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#195 */
AI_ARRAY_OBJ_DECLARE(
  gemm_279_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#196 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_281_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#197 */
AI_ARRAY_OBJ_DECLARE(
  transpose_283_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#198 */
AI_ARRAY_OBJ_DECLARE(
  transpose_283_0_0_gemm_284_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#199 */
AI_ARRAY_OBJ_DECLARE(
  gemm_274_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#200 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_276_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#201 */
AI_ARRAY_OBJ_DECLARE(
  transpose_bgemm_284_out_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#202 */
AI_ARRAY_OBJ_DECLARE(
  transpose_bgemm_284_out_0_1_gemm_284_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#203 */
AI_ARRAY_OBJ_DECLARE(
  gemm_284_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 40000, AI_STATIC)

/* Array#204 */
AI_ARRAY_OBJ_DECLARE(
  gemm_284_0_0_eltwise_285_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 40000, AI_STATIC)

/* Array#205 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_285_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 40000, AI_STATIC)

/* Array#206 */
AI_ARRAY_OBJ_DECLARE(
  nl_286_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 40000, AI_STATIC)

/* Array#207 */
AI_ARRAY_OBJ_DECLARE(
  nl_286_0_0_gemm_292_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 40000, AI_STATIC)

/* Array#208 */
AI_ARRAY_OBJ_DECLARE(
  gemm_292_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#209 */
AI_ARRAY_OBJ_DECLARE(
  gemm_292_0_0_transpose_293_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#210 */
AI_ARRAY_OBJ_DECLARE(
  transpose_293_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#211 */
AI_ARRAY_OBJ_DECLARE(
  gemm_303_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#212 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_305_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#213 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_306_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#214 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_306_0_0_reduce_307_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#215 */
AI_ARRAY_OBJ_DECLARE(
  reduce_307_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#216 */
AI_ARRAY_OBJ_DECLARE(
  reduce_307_Mul_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#217 */
AI_ARRAY_OBJ_DECLARE(
  reduce_307_Mul_0_1_eltwise_308_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 100, AI_STATIC)

/* Array#218 */
AI_ARRAY_OBJ_DECLARE(
  reduce_307_Mul_0_1_eltwise_308_conversion_0_1_eltwise_308_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#219 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_308_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#220 */
AI_ARRAY_OBJ_DECLARE(
  reduce_309_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#221 */
AI_ARRAY_OBJ_DECLARE(
  reduce_309_Mul_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#222 */
AI_ARRAY_OBJ_DECLARE(
  reduce_309_Mul_0_0_eltwise_310_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 100, AI_STATIC)

/* Array#223 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_310_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 100, AI_STATIC)

/* Array#224 */
AI_ARRAY_OBJ_DECLARE(
  nl_311_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 100, AI_STATIC)

/* Array#225 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_312_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#226 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_314_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#227 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_315_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#228 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_313_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#229 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_316_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#230 */
AI_ARRAY_OBJ_DECLARE(
  gemm_325_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 51200, AI_STATIC)

/* Array#231 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_327_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 51200, AI_STATIC)

/* Array#232 */
AI_ARRAY_OBJ_DECLARE(
  gemm_336_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#233 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_338_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#234 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_339_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#235 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_339_0_0_reduce_340_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#236 */
AI_ARRAY_OBJ_DECLARE(
  reduce_340_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#237 */
AI_ARRAY_OBJ_DECLARE(
  reduce_340_Mul_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#238 */
AI_ARRAY_OBJ_DECLARE(
  reduce_340_Mul_0_1_eltwise_341_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 100, AI_STATIC)

/* Array#239 */
AI_ARRAY_OBJ_DECLARE(
  reduce_340_Mul_0_1_eltwise_341_conversion_0_1_eltwise_341_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#240 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_341_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#241 */
AI_ARRAY_OBJ_DECLARE(
  reduce_342_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#242 */
AI_ARRAY_OBJ_DECLARE(
  reduce_342_Mul_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#243 */
AI_ARRAY_OBJ_DECLARE(
  reduce_342_Mul_0_0_eltwise_343_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 100, AI_STATIC)

/* Array#244 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_343_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 100, AI_STATIC)

/* Array#245 */
AI_ARRAY_OBJ_DECLARE(
  nl_344_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 100, AI_STATIC)

/* Array#246 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_345_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#247 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_347_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#248 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_348_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#249 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_346_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#250 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_349_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#251 */
AI_ARRAY_OBJ_DECLARE(
  gemm_358_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 6600, AI_STATIC)

/* Array#252 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_360_output_array, AI_ARRAY_FORMAT_S8|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 6600, AI_STATIC)

/* Array#253 */
AI_ARRAY_OBJ_DECLARE(
  gemm_292_bias_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1, AI_STATIC)

/* Array#254 */
AI_ARRAY_OBJ_DECLARE(
  gemm_284_bias_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1, AI_STATIC)

/* Array#255 */
AI_ARRAY_OBJ_DECLARE(
  gemm_205_bias_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1, AI_STATIC)

/* Array#256 */
AI_ARRAY_OBJ_DECLARE(
  gemm_197_bias_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1, AI_STATIC)

/* Array#257 */
AI_ARRAY_OBJ_DECLARE(
  gemm_118_bias_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1, AI_STATIC)

/* Array#258 */
AI_ARRAY_OBJ_DECLARE(
  gemm_110_bias_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1, AI_STATIC)

/* Array#259 */
AI_ARRAY_OBJ_DECLARE(
  gemm_31_bias_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1, AI_STATIC)

/* Array#260 */
AI_ARRAY_OBJ_DECLARE(
  gemm_23_bias_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1, AI_STATIC)

/* Array#261 */
AI_ARRAY_OBJ_DECLARE(
  tiny_bert_generator_tf_math_multiply_1_Mul_y_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1, AI_STATIC)

/* Array#262 */
AI_ARRAY_OBJ_DECLARE(
  tiny_bert_generator_enc_0_mha_wk_BiasAdd_ReadVariableOp_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#263 */
AI_ARRAY_OBJ_DECLARE(
  tiny_bert_generator_enc_0_mha_wq_BiasAdd_ReadVariableOp_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#264 */
AI_ARRAY_OBJ_DECLARE(
  tiny_bert_generator_enc_3_mha_truedivtiny_bert_generator_enc_3_mha_Sqrt_4D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1, AI_STATIC)

/* Array#265 */
AI_ARRAY_OBJ_DECLARE(
  tiny_bert_generator_enc_0_mha_wv_BiasAdd_ReadVariableOp_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#266 */
AI_ARRAY_OBJ_DECLARE(
  tiny_bert_generator_enc_0_mha_output_dense_BiasAdd_ReadVariableOp_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#267 */
AI_ARRAY_OBJ_DECLARE(
  tiny_bert_generator_enc_0_ln1_batchnorm_add_y_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1, AI_STATIC)

/* Array#268 */
AI_ARRAY_OBJ_DECLARE(
  tiny_bert_generator_enc_0_ln1_batchnorm_mul_ReadVariableOp_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#269 */
AI_ARRAY_OBJ_DECLARE(
  tiny_bert_generator_enc_0_ln1_batchnorm_ReadVariableOp_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#270 */
AI_ARRAY_OBJ_DECLARE(
  tiny_bert_generator_enc_0_ffn_dense_24_BiasAdd_ReadVariableOp_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 512, AI_STATIC)

/* Array#271 */
AI_ARRAY_OBJ_DECLARE(
  tiny_bert_generator_enc_0_ffn_dense_25_BiasAdd_ReadVariableOp_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#272 */
AI_ARRAY_OBJ_DECLARE(
  tiny_bert_generator_enc_0_ln2_batchnorm_mul_ReadVariableOp_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#273 */
AI_ARRAY_OBJ_DECLARE(
  tiny_bert_generator_enc_0_ln2_batchnorm_ReadVariableOp_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#274 */
AI_ARRAY_OBJ_DECLARE(
  tiny_bert_generator_enc_1_mha_wk_BiasAdd_ReadVariableOp_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#275 */
AI_ARRAY_OBJ_DECLARE(
  tiny_bert_generator_enc_1_mha_wq_BiasAdd_ReadVariableOp_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#276 */
AI_ARRAY_OBJ_DECLARE(
  tiny_bert_generator_enc_1_mha_wv_BiasAdd_ReadVariableOp_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#277 */
AI_ARRAY_OBJ_DECLARE(
  tiny_bert_generator_enc_1_mha_output_dense_BiasAdd_ReadVariableOp_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#278 */
AI_ARRAY_OBJ_DECLARE(
  tiny_bert_generator_enc_1_ln1_batchnorm_mul_ReadVariableOp_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#279 */
AI_ARRAY_OBJ_DECLARE(
  tiny_bert_generator_enc_1_ln1_batchnorm_ReadVariableOp_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#280 */
AI_ARRAY_OBJ_DECLARE(
  tiny_bert_generator_enc_1_ffn_dense_26_BiasAdd_ReadVariableOp_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 512, AI_STATIC)

/* Array#281 */
AI_ARRAY_OBJ_DECLARE(
  tiny_bert_generator_enc_1_ffn_dense_27_BiasAdd_ReadVariableOp_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#282 */
AI_ARRAY_OBJ_DECLARE(
  tiny_bert_generator_enc_1_ln2_batchnorm_mul_ReadVariableOp_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#283 */
AI_ARRAY_OBJ_DECLARE(
  tiny_bert_generator_enc_1_ln2_batchnorm_ReadVariableOp_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#284 */
AI_ARRAY_OBJ_DECLARE(
  tiny_bert_generator_enc_2_mha_wk_BiasAdd_ReadVariableOp_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#285 */
AI_ARRAY_OBJ_DECLARE(
  tiny_bert_generator_enc_2_mha_wq_BiasAdd_ReadVariableOp_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#286 */
AI_ARRAY_OBJ_DECLARE(
  tiny_bert_generator_enc_2_mha_wv_BiasAdd_ReadVariableOp_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#287 */
AI_ARRAY_OBJ_DECLARE(
  tiny_bert_generator_enc_2_mha_output_dense_BiasAdd_ReadVariableOp_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#288 */
AI_ARRAY_OBJ_DECLARE(
  tiny_bert_generator_enc_2_ln1_batchnorm_mul_ReadVariableOp_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#289 */
AI_ARRAY_OBJ_DECLARE(
  tiny_bert_generator_enc_2_ln1_batchnorm_ReadVariableOp_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#290 */
AI_ARRAY_OBJ_DECLARE(
  tiny_bert_generator_enc_2_ffn_dense_28_BiasAdd_ReadVariableOp_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 512, AI_STATIC)

/* Array#291 */
AI_ARRAY_OBJ_DECLARE(
  tiny_bert_generator_enc_2_ffn_dense_29_BiasAdd_ReadVariableOp_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#292 */
AI_ARRAY_OBJ_DECLARE(
  tiny_bert_generator_enc_2_ln2_batchnorm_mul_ReadVariableOp_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#293 */
AI_ARRAY_OBJ_DECLARE(
  tiny_bert_generator_enc_2_ln2_batchnorm_ReadVariableOp_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#294 */
AI_ARRAY_OBJ_DECLARE(
  tiny_bert_generator_enc_3_mha_wk_BiasAdd_ReadVariableOp_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#295 */
AI_ARRAY_OBJ_DECLARE(
  tiny_bert_generator_enc_3_mha_wq_BiasAdd_ReadVariableOp_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#296 */
AI_ARRAY_OBJ_DECLARE(
  tiny_bert_generator_enc_3_mha_wv_BiasAdd_ReadVariableOp_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#297 */
AI_ARRAY_OBJ_DECLARE(
  tiny_bert_generator_enc_3_mha_output_dense_BiasAdd_ReadVariableOp_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#298 */
AI_ARRAY_OBJ_DECLARE(
  tiny_bert_generator_enc_3_ln1_batchnorm_mul_ReadVariableOp_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#299 */
AI_ARRAY_OBJ_DECLARE(
  tiny_bert_generator_enc_3_ln1_batchnorm_ReadVariableOp_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#300 */
AI_ARRAY_OBJ_DECLARE(
  tiny_bert_generator_enc_3_ffn_dense_30_BiasAdd_ReadVariableOp_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 512, AI_STATIC)

/* Array#301 */
AI_ARRAY_OBJ_DECLARE(
  tiny_bert_generator_enc_3_ffn_dense_31_BiasAdd_ReadVariableOp_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#302 */
AI_ARRAY_OBJ_DECLARE(
  tiny_bert_generator_enc_3_ln2_batchnorm_mul_ReadVariableOp_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#303 */
AI_ARRAY_OBJ_DECLARE(
  tiny_bert_generator_enc_3_ln2_batchnorm_ReadVariableOp_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#304 */
AI_ARRAY_OBJ_DECLARE(
  tiny_bert_generator_output_dense_BiasAdd_ReadVariableOp_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 66, AI_STATIC)

/* Array#305 */
AI_ARRAY_OBJ_DECLARE(
  tiny_bert_generator_tf___operators___add_1_y_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#306 */
AI_ARRAY_OBJ_DECLARE(
  gemm_26_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 65536, AI_STATIC)

/* Array#307 */
AI_ARRAY_OBJ_DECLARE(
  gemm_26_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 256, AI_STATIC)

/* Array#308 */
AI_ARRAY_OBJ_DECLARE(
  gemm_18_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 65536, AI_STATIC)

/* Array#309 */
AI_ARRAY_OBJ_DECLARE(
  gemm_13_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 65536, AI_STATIC)

/* Array#310 */
AI_ARRAY_OBJ_DECLARE(
  gemm_42_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 65536, AI_STATIC)

/* Array#311 */
AI_ARRAY_OBJ_DECLARE(
  reduce_46_Mul_scale_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)

/* Array#312 */
AI_ARRAY_OBJ_DECLARE(
  reduce_46_Mul_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)

/* Array#313 */
AI_ARRAY_OBJ_DECLARE(
  gemm_64_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 131072, AI_STATIC)

/* Array#314 */
AI_ARRAY_OBJ_DECLARE(
  gemm_64_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 512, AI_STATIC)

/* Array#315 */
AI_ARRAY_OBJ_DECLARE(
  gemm_75_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 131072, AI_STATIC)

/* Array#316 */
AI_ARRAY_OBJ_DECLARE(
  gemm_113_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 65536, AI_STATIC)

/* Array#317 */
AI_ARRAY_OBJ_DECLARE(
  gemm_105_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 65536, AI_STATIC)

/* Array#318 */
AI_ARRAY_OBJ_DECLARE(
  gemm_100_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 65536, AI_STATIC)

/* Array#319 */
AI_ARRAY_OBJ_DECLARE(
  gemm_129_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 65536, AI_STATIC)

/* Array#320 */
AI_ARRAY_OBJ_DECLARE(
  gemm_151_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 131072, AI_STATIC)

/* Array#321 */
AI_ARRAY_OBJ_DECLARE(
  gemm_162_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 131072, AI_STATIC)

/* Array#322 */
AI_ARRAY_OBJ_DECLARE(
  gemm_200_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 65536, AI_STATIC)

/* Array#323 */
AI_ARRAY_OBJ_DECLARE(
  gemm_192_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 65536, AI_STATIC)

/* Array#324 */
AI_ARRAY_OBJ_DECLARE(
  gemm_187_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 65536, AI_STATIC)

/* Array#325 */
AI_ARRAY_OBJ_DECLARE(
  gemm_216_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 65536, AI_STATIC)

/* Array#326 */
AI_ARRAY_OBJ_DECLARE(
  gemm_238_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 131072, AI_STATIC)

/* Array#327 */
AI_ARRAY_OBJ_DECLARE(
  gemm_249_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 131072, AI_STATIC)

/* Array#328 */
AI_ARRAY_OBJ_DECLARE(
  gemm_287_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 65536, AI_STATIC)

/* Array#329 */
AI_ARRAY_OBJ_DECLARE(
  gemm_279_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 65536, AI_STATIC)

/* Array#330 */
AI_ARRAY_OBJ_DECLARE(
  gemm_274_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 65536, AI_STATIC)

/* Array#331 */
AI_ARRAY_OBJ_DECLARE(
  gemm_303_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 65536, AI_STATIC)

/* Array#332 */
AI_ARRAY_OBJ_DECLARE(
  gemm_325_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 131072, AI_STATIC)

/* Array#333 */
AI_ARRAY_OBJ_DECLARE(
  gemm_336_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 131072, AI_STATIC)

/* Array#334 */
AI_ARRAY_OBJ_DECLARE(
  gemm_358_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16896, AI_STATIC)

/* Array#335 */
AI_ARRAY_OBJ_DECLARE(
  gemm_358_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 66, AI_STATIC)

/* Array#336 */
AI_ARRAY_OBJ_DECLARE(
  gemm_26_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1024, AI_STATIC)

/* Array#337 */
AI_ARRAY_OBJ_DECLARE(
  gemm_18_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1024, AI_STATIC)

/* Array#338 */
AI_ARRAY_OBJ_DECLARE(
  gemm_13_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1024, AI_STATIC)

/* Array#339 */
AI_ARRAY_OBJ_DECLARE(
  nl_25_scratch0_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 7, AI_STATIC)

/* Array#340 */
AI_ARRAY_OBJ_DECLARE(
  gemm_42_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1024, AI_STATIC)

/* Array#341 */
AI_ARRAY_OBJ_DECLARE(
  gemm_64_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1024, AI_STATIC)

/* Array#342 */
AI_ARRAY_OBJ_DECLARE(
  gemm_75_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2048, AI_STATIC)

/* Array#343 */
AI_ARRAY_OBJ_DECLARE(
  gemm_113_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1024, AI_STATIC)

/* Array#344 */
AI_ARRAY_OBJ_DECLARE(
  gemm_105_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1024, AI_STATIC)

/* Array#345 */
AI_ARRAY_OBJ_DECLARE(
  gemm_100_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1024, AI_STATIC)

/* Array#346 */
AI_ARRAY_OBJ_DECLARE(
  nl_112_scratch0_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 62, AI_STATIC)

/* Array#347 */
AI_ARRAY_OBJ_DECLARE(
  gemm_129_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1024, AI_STATIC)

/* Array#348 */
AI_ARRAY_OBJ_DECLARE(
  gemm_151_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1024, AI_STATIC)

/* Array#349 */
AI_ARRAY_OBJ_DECLARE(
  gemm_162_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2048, AI_STATIC)

/* Array#350 */
AI_ARRAY_OBJ_DECLARE(
  gemm_200_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1024, AI_STATIC)

/* Array#351 */
AI_ARRAY_OBJ_DECLARE(
  gemm_192_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1024, AI_STATIC)

/* Array#352 */
AI_ARRAY_OBJ_DECLARE(
  gemm_187_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1024, AI_STATIC)

/* Array#353 */
AI_ARRAY_OBJ_DECLARE(
  nl_199_scratch0_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 31, AI_STATIC)

/* Array#354 */
AI_ARRAY_OBJ_DECLARE(
  gemm_216_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1024, AI_STATIC)

/* Array#355 */
AI_ARRAY_OBJ_DECLARE(
  gemm_238_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1024, AI_STATIC)

/* Array#356 */
AI_ARRAY_OBJ_DECLARE(
  gemm_249_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2048, AI_STATIC)

/* Array#357 */
AI_ARRAY_OBJ_DECLARE(
  gemm_287_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1024, AI_STATIC)

/* Array#358 */
AI_ARRAY_OBJ_DECLARE(
  gemm_279_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1024, AI_STATIC)

/* Array#359 */
AI_ARRAY_OBJ_DECLARE(
  gemm_274_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1024, AI_STATIC)

/* Array#360 */
AI_ARRAY_OBJ_DECLARE(
  nl_286_scratch0_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 7, AI_STATIC)

/* Array#361 */
AI_ARRAY_OBJ_DECLARE(
  gemm_303_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1024, AI_STATIC)

/* Array#362 */
AI_ARRAY_OBJ_DECLARE(
  gemm_325_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1024, AI_STATIC)

/* Array#363 */
AI_ARRAY_OBJ_DECLARE(
  gemm_336_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2048, AI_STATIC)

/* Array#364 */
AI_ARRAY_OBJ_DECLARE(
  gemm_358_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1024, AI_STATIC)

/**  Array metadata declarations section  *************************************/
/* Int quant #0 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.025706449523568153f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #1 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_102_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.057787321507930756f),
    AI_PACK_INTQ_ZP(-12)))

/* Int quant #2 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_107_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.09996993094682693f),
    AI_PACK_INTQ_ZP(-3)))

/* Int quant #3 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_111_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.4037426710128784f),
    AI_PACK_INTQ_ZP(103)))

/* Int quant #4 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_115_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.1256372034549713f),
    AI_PACK_INTQ_ZP(1)))

/* Int quant #5 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_131_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.8139194250106812f),
    AI_PACK_INTQ_ZP(-5)))

/* Int quant #6 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_132_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.8124402761459351f),
    AI_PACK_INTQ_ZP(-5)))

/* Int quant #7 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_136_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(9.411563873291016f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #8 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_138_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(9.092828258872032e-05f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #9 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_139_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.017929712310433388f),
    AI_PACK_INTQ_ZP(-4)))

/* Int quant #10 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_140_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(8.65034235175699e-06f),
    AI_PACK_INTQ_ZP(127)))

/* Int quant #11 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_141_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0008392363088205457f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #12 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_142_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.01861877366900444f),
    AI_PACK_INTQ_ZP(-5)))

/* Int quant #13 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_153_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.1538243442773819f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #14 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_15_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.1098199412226677f),
    AI_PACK_INTQ_ZP(-10)))

/* Int quant #15 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_164_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(1.4196467399597168f),
    AI_PACK_INTQ_ZP(-4)))

/* Int quant #16 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_165_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(1.4268971681594849f),
    AI_PACK_INTQ_ZP(-4)))

/* Int quant #17 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_169_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(31.192527770996094f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #18 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_171_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(4.826235817745328e-05f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #19 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_172_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.01712680049240589f),
    AI_PACK_INTQ_ZP(-5)))

/* Int quant #20 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_173_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(1.7469867088948376e-05f),
    AI_PACK_INTQ_ZP(127)))

/* Int quant #21 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_174_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0008286628290079534f),
    AI_PACK_INTQ_ZP(-12)))

/* Int quant #22 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_175_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.017797354608774185f),
    AI_PACK_INTQ_ZP(-6)))

/* Int quant #23 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_189_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.08979139477014542f),
    AI_PACK_INTQ_ZP(9)))

/* Int quant #24 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_194_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.1136530190706253f),
    AI_PACK_INTQ_ZP(-8)))

/* Int quant #25 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_198_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.946876049041748f),
    AI_PACK_INTQ_ZP(-58)))

/* Int quant #26 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_1_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.030732745304703712f),
    AI_PACK_INTQ_ZP(-6)))

/* Int quant #27 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_202_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.13711415231227875f),
    AI_PACK_INTQ_ZP(-10)))

/* Int quant #28 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_20_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.10581283271312714f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #29 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_218_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.6782305836677551f),
    AI_PACK_INTQ_ZP(5)))

/* Int quant #30 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_219_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.690091609954834f),
    AI_PACK_INTQ_ZP(6)))

/* Int quant #31 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_223_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(9.204447746276855f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #32 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_225_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(9.429892816115171e-05f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #33 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_226_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.015534243546426296f),
    AI_PACK_INTQ_ZP(5)))

/* Int quant #34 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_227_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00010158392251469195f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #35 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_228_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0006992782582528889f),
    AI_PACK_INTQ_ZP(25)))

/* Int quant #36 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_229_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.016174277290701866f),
    AI_PACK_INTQ_ZP(6)))

/* Int quant #37 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_240_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0933648943901062f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #38 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_24_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(2.437107801437378f),
    AI_PACK_INTQ_ZP(-24)))

/* Int quant #39 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_251_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.7780804634094238f),
    AI_PACK_INTQ_ZP(15)))

/* Int quant #40 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_252_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.7747630476951599f),
    AI_PACK_INTQ_ZP(15)))

/* Int quant #41 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_256_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(6.855565547943115f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #42 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_258_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00010713700612541288f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #43 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_259_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02043519914150238f),
    AI_PACK_INTQ_ZP(15)))

/* Int quant #44 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_260_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(7.962021300045308e-06f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #45 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_261_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0005562662845477462f),
    AI_PACK_INTQ_ZP(2)))

/* Int quant #46 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_262_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.020991463214159012f),
    AI_PACK_INTQ_ZP(14)))

/* Int quant #47 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_276_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.1260143369436264f),
    AI_PACK_INTQ_ZP(-11)))

/* Int quant #48 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_281_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.14364778995513916f),
    AI_PACK_INTQ_ZP(-6)))

/* Int quant #49 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_285_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(2.4836690425872803f),
    AI_PACK_INTQ_ZP(127)))

/* Int quant #50 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_289_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.21361419558525085f),
    AI_PACK_INTQ_ZP(-2)))

/* Int quant #51 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_28_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.13490118086338043f),
    AI_PACK_INTQ_ZP(-2)))

/* Int quant #52 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_305_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(1.9216004610061646f),
    AI_PACK_INTQ_ZP(12)))

/* Int quant #53 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_306_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(1.9108195304870605f),
    AI_PACK_INTQ_ZP(12)))

/* Int quant #54 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_310_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(46.68927764892578f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #55 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_312_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(4.566111965687014e-05f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #56 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_313_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02051856927573681f),
    AI_PACK_INTQ_ZP(10)))

/* Int quant #57 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_314_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(6.300689619820332e-06f),
    AI_PACK_INTQ_ZP(127)))

/* Int quant #58 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_315_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0020774921867996454f),
    AI_PACK_INTQ_ZP(-5)))

/* Int quant #59 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_316_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02247421070933342f),
    AI_PACK_INTQ_ZP(8)))

/* Int quant #60 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_327_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.512325644493103f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #61 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_338_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(27.55299949645996f),
    AI_PACK_INTQ_ZP(9)))

/* Int quant #62 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_339_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(27.55480194091797f),
    AI_PACK_INTQ_ZP(9)))

/* Int quant #63 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_343_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(966.4218139648438f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #64 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_345_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(6.211575509951217e-06f),
    AI_PACK_INTQ_ZP(-103)))

/* Int quant #65 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_346_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.01643681898713112f),
    AI_PACK_INTQ_ZP(-34)))

/* Int quant #66 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_347_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(4.5645218051504344e-05f),
    AI_PACK_INTQ_ZP(-103)))

/* Int quant #67 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_348_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.004172159358859062f),
    AI_PACK_INTQ_ZP(4)))

/* Int quant #68 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_349_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.013832950964570045f),
    AI_PACK_INTQ_ZP(-32)))

/* Int quant #69 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_360_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.049303341656923294f),
    AI_PACK_INTQ_ZP(74)))

/* Int quant #70 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_44_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.7943773865699768f),
    AI_PACK_INTQ_ZP(4)))

/* Int quant #71 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_45_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.7950785160064697f),
    AI_PACK_INTQ_ZP(4)))

/* Int quant #72 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_49_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(9.29905891418457f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #73 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_51_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0010268354089930654f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #74 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_52_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.023193450644612312f),
    AI_PACK_INTQ_ZP(-12)))

/* Int quant #75 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_53_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0005586555344052613f),
    AI_PACK_INTQ_ZP(-39)))

/* Int quant #76 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_54_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0010431462433189154f),
    AI_PACK_INTQ_ZP(18)))

/* Int quant #77 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_55_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.023503223434090614f),
    AI_PACK_INTQ_ZP(-10)))

/* Int quant #78 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_66_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.06448160856962204f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #79 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_77_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(1.1295177936553955f),
    AI_PACK_INTQ_ZP(1)))

/* Int quant #80 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_78_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(1.1365560293197632f),
    AI_PACK_INTQ_ZP(1)))

/* Int quant #81 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_82_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(19.320114135742188f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #82 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_84_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00012324837734922767f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #83 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_85_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.018005622550845146f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #84 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_86_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(6.057717837393284e-05f),
    AI_PACK_INTQ_ZP(127)))

/* Int quant #85 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_87_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0008771343273110688f),
    AI_PACK_INTQ_ZP(-20)))

/* Int quant #86 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_88_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.01878371089696884f),
    AI_PACK_INTQ_ZP(-3)))

/* Int quant #87 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_100_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.05778725445270538f),
    AI_PACK_INTQ_ZP(-12)))

/* Int quant #88 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_100_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0016486539971083403f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #89 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_105_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.09944001585245132f),
    AI_PACK_INTQ_ZP(-3)))

/* Int quant #90 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_105_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0019136159680783749f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #91 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_110_0_0_eltwise_111_conversion_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(3.2299413681030273f),
    AI_PACK_INTQ_ZP(103)))

/* Int quant #92 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_110_bias_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(3.2299413681030273f),
    AI_PACK_INTQ_ZP(103)))

/* Int quant #93 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_113_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.12496710568666458f),
    AI_PACK_INTQ_ZP(1)))

/* Int quant #94 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_113_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0017305412329733372f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #95 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_118_0_0_transpose_119_conversion_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.12557001411914825f),
    AI_PACK_INTQ_ZP(1)))

/* Int quant #96 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_118_bias_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.12557001411914825f),
    AI_PACK_INTQ_ZP(1)))

/* Int quant #97 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_129_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.8133624196052551f),
    AI_PACK_INTQ_ZP(-5)))

/* Int quant #98 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_129_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0017849564319476485f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #99 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_13_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.10981569439172745f),
    AI_PACK_INTQ_ZP(-10)))

/* Int quant #100 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_13_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0018929282668977976f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #101 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_151_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.23654940724372864f),
    AI_PACK_INTQ_ZP(-38)))

/* Int quant #102 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_151_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.009155895560979843f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #103 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_162_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(1.419053554534912f),
    AI_PACK_INTQ_ZP(-4)))

/* Int quant #104 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_162_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.009357985109090805f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #105 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_187_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.08979574590921402f),
    AI_PACK_INTQ_ZP(9)))

/* Int quant #106 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_187_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0022518427576869726f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #107 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_18_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.10544240474700928f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #108 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_18_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.002914897631853819f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #109 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_192_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.11324325203895569f),
    AI_PACK_INTQ_ZP(-8)))

/* Int quant #110 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_192_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.001854496425949037f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #111 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_197_0_0_eltwise_198_conversion_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(7.575008392333984f),
    AI_PACK_INTQ_ZP(-58)))

/* Int quant #112 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_197_bias_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(7.575008392333984f),
    AI_PACK_INTQ_ZP(-58)))

/* Int quant #113 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_200_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.13645605742931366f),
    AI_PACK_INTQ_ZP(-10)))

/* Int quant #114 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_200_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0019889508839696646f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #115 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_205_0_0_transpose_206_conversion_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.13711419701576233f),
    AI_PACK_INTQ_ZP(-10)))

/* Int quant #116 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_205_bias_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.13711419701576233f),
    AI_PACK_INTQ_ZP(-10)))

/* Int quant #117 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_216_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.6776818633079529f),
    AI_PACK_INTQ_ZP(5)))

/* Int quant #118 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_216_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0019772157538682222f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #119 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_238_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.17399810254573822f),
    AI_PACK_INTQ_ZP(-9)))

/* Int quant #120 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_238_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.004047996364533901f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #121 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_23_0_0_eltwise_24_conversion_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(19.496862411499023f),
    AI_PACK_INTQ_ZP(-24)))

/* Int quant #122 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_23_bias_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(19.496862411499023f),
    AI_PACK_INTQ_ZP(-24)))

/* Int quant #123 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_249_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.7776262164115906f),
    AI_PACK_INTQ_ZP(15)))

/* Int quant #124 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_249_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.002718492178246379f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #125 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_26_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.13458198308944702f),
    AI_PACK_INTQ_ZP(-2)))

/* Int quant #126 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_26_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.002491940278559923f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #127 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_274_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.1260126680135727f),
    AI_PACK_INTQ_ZP(-11)))

/* Int quant #128 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_274_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.002221645787358284f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #129 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_279_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.14295925199985504f),
    AI_PACK_INTQ_ZP(-6)))

/* Int quant #130 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_279_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0022351876832544804f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #131 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_284_0_0_eltwise_285_conversion_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(19.869352340698242f),
    AI_PACK_INTQ_ZP(127)))

/* Int quant #132 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_284_bias_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(19.869352340698242f),
    AI_PACK_INTQ_ZP(127)))

/* Int quant #133 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_287_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.21294474601745605f),
    AI_PACK_INTQ_ZP(-2)))

/* Int quant #134 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_287_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.002494936576113105f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #135 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_292_0_0_transpose_293_conversion_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.21361443400382996f),
    AI_PACK_INTQ_ZP(-2)))

/* Int quant #136 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_292_bias_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.21361443400382996f),
    AI_PACK_INTQ_ZP(-2)))

/* Int quant #137 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_303_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(1.921013593673706f),
    AI_PACK_INTQ_ZP(12)))

/* Int quant #138 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_303_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0029276690911501646f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #139 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_31_0_0_transpose_32_conversion_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.13490113615989685f),
    AI_PACK_INTQ_ZP(-2)))

/* Int quant #140 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_31_bias_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.13490113615989685f),
    AI_PACK_INTQ_ZP(-2)))

/* Int quant #141 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_325_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.5606815814971924f),
    AI_PACK_INTQ_ZP(-105)))

/* Int quant #142 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_325_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.005390227772295475f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #143 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_336_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(27.548830032348633f),
    AI_PACK_INTQ_ZP(9)))

/* Int quant #144 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_336_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.009032238274812698f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #145 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_358_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.043946925550699234f),
    AI_PACK_INTQ_ZP(74)))

/* Int quant #146 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_358_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.01777946762740612f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #147 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_42_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.7943030595779419f),
    AI_PACK_INTQ_ZP(4)))

/* Int quant #148 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_42_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0017215185798704624f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #149 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_64_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.12841933965682983f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #150 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_64_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.003756167832762003f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #151 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_75_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(1.1289100646972656f),
    AI_PACK_INTQ_ZP(1)))

/* Int quant #152 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_75_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0029262362513691187f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #153 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_112_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00390625f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #154 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_137_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(8.149569475790486e-05f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #155 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_170_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(4.401422484079376e-05f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #156 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_199_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00390625f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #157 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_224_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(8.095167140709236e-05f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #158 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_257_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(9.379312541568652e-05f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #159 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_25_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00390625f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #160 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_286_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00390625f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #161 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_311_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(3.5940276575274765e-05f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #162 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_344_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(7.899625416030176e-06f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #163 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_50_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0009359540999867022f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #164 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_83_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00011120088311145082f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #165 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(reduce_133_Mul_0_1_eltwise_134_conversion_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.000379670353140682f),
    AI_PACK_INTQ_ZP(127)))

/* Int quant #166 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(reduce_135_Mul_0_0_eltwise_136_conversion_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(9.411563873291016f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #167 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(reduce_166_Mul_0_1_eltwise_167_conversion_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0014197807759046555f),
    AI_PACK_INTQ_ZP(127)))

/* Int quant #168 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(reduce_168_Mul_0_0_eltwise_169_conversion_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(31.192527770996094f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #169 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(reduce_220_Mul_0_1_eltwise_221_conversion_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.004224866162985563f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #170 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(reduce_222_Mul_0_0_eltwise_223_conversion_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(9.204447746276855f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #171 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(reduce_253_Mul_0_1_eltwise_254_conversion_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0002914363285526633f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #172 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(reduce_255_Mul_0_0_eltwise_256_conversion_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(6.855565547943115f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #173 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(reduce_307_Mul_0_1_eltwise_308_conversion_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0005411297897808254f),
    AI_PACK_INTQ_ZP(127)))

/* Int quant #174 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(reduce_309_Mul_0_0_eltwise_310_conversion_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(46.68927764892578f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #175 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(reduce_340_Mul_0_1_eltwise_341_conversion_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.028817307204008102f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #176 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(reduce_342_Mul_0_0_eltwise_343_conversion_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(966.4218139648438f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #177 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(reduce_46_Mul_0_1_eltwise_47_conversion_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.008404841646552086f),
    AI_PACK_INTQ_ZP(80)))

/* Int quant #178 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(reduce_48_Mul_0_0_eltwise_49_conversion_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(9.29905891418457f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #179 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(reduce_79_Mul_0_1_eltwise_80_conversion_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.003242065431550145f),
    AI_PACK_INTQ_ZP(127)))

/* Int quant #180 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(reduce_81_Mul_0_0_eltwise_82_conversion_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(19.320114135742188f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #181 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(serving_default_embedded_tokens0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0016066530952230096f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #182 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(tiny_bert_generator_enc_0_ffn_dense_24_BiasAdd_ReadVariableOp_3D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0006058535072952509f),
    AI_PACK_INTQ_ZP(-29)))

/* Int quant #183 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(tiny_bert_generator_enc_0_ffn_dense_25_BiasAdd_ReadVariableOp_3D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.000656914955470711f),
    AI_PACK_INTQ_ZP(2)))

/* Int quant #184 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(tiny_bert_generator_enc_0_ln1_batchnorm_ReadVariableOp_3D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0005488575552590191f),
    AI_PACK_INTQ_ZP(1)))

/* Int quant #185 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(tiny_bert_generator_enc_0_ln1_batchnorm_add_y_3D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(5.882353271147167e-09f),
    AI_PACK_INTQ_ZP(-43)))

/* Int quant #186 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(tiny_bert_generator_enc_0_ln1_batchnorm_mul_ReadVariableOp_3D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.004302353598177433f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #187 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(tiny_bert_generator_enc_0_ln2_batchnorm_ReadVariableOp_3D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0008352758595719934f),
    AI_PACK_INTQ_ZP(-9)))

/* Int quant #188 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(tiny_bert_generator_enc_0_ln2_batchnorm_mul_ReadVariableOp_3D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.004346431698650122f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #189 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(tiny_bert_generator_enc_0_mha_output_dense_BiasAdd_ReadVariableOp_3D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.000437113776570186f),
    AI_PACK_INTQ_ZP(-3)))

/* Int quant #190 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(tiny_bert_generator_enc_0_mha_wk_BiasAdd_ReadVariableOp_3D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(2.341756953683216e-05f),
    AI_PACK_INTQ_ZP(-35)))

/* Int quant #191 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(tiny_bert_generator_enc_0_mha_wq_BiasAdd_ReadVariableOp_3D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0005861893878318369f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #192 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(tiny_bert_generator_enc_0_mha_wv_BiasAdd_ReadVariableOp_3D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0005144912865944207f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #193 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(tiny_bert_generator_enc_1_ffn_dense_26_BiasAdd_ReadVariableOp_3D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0008236012072302401f),
    AI_PACK_INTQ_ZP(-61)))

/* Int quant #194 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(tiny_bert_generator_enc_1_ffn_dense_27_BiasAdd_ReadVariableOp_3D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0006995750009082258f),
    AI_PACK_INTQ_ZP(-3)))

/* Int quant #195 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(tiny_bert_generator_enc_1_ln1_batchnorm_ReadVariableOp_3D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0008380362996831536f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #196 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(tiny_bert_generator_enc_1_ln1_batchnorm_mul_ReadVariableOp_3D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.004375464282929897f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #197 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(tiny_bert_generator_enc_1_ln2_batchnorm_ReadVariableOp_3D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0008274564752355218f),
    AI_PACK_INTQ_ZP(-7)))

/* Int quant #198 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(tiny_bert_generator_enc_1_ln2_batchnorm_mul_ReadVariableOp_3D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.004300068132579327f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #199 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(tiny_bert_generator_enc_1_mha_output_dense_BiasAdd_ReadVariableOp_3D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0007089835708029568f),
    AI_PACK_INTQ_ZP(1)))

/* Int quant #200 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(tiny_bert_generator_enc_1_mha_wk_BiasAdd_ReadVariableOp_3D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(1.0524530807742849e-05f),
    AI_PACK_INTQ_ZP(8)))

/* Int quant #201 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(tiny_bert_generator_enc_1_mha_wq_BiasAdd_ReadVariableOp_3D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0005884980200789869f),
    AI_PACK_INTQ_ZP(-3)))

/* Int quant #202 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(tiny_bert_generator_enc_1_mha_wv_BiasAdd_ReadVariableOp_3D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0008321540080942214f),
    AI_PACK_INTQ_ZP(-3)))

/* Int quant #203 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(tiny_bert_generator_enc_2_ffn_dense_28_BiasAdd_ReadVariableOp_3D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0006950747920200229f),
    AI_PACK_INTQ_ZP(-18)))

/* Int quant #204 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(tiny_bert_generator_enc_2_ffn_dense_29_BiasAdd_ReadVariableOp_3D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0005606161430478096f),
    AI_PACK_INTQ_ZP(-11)))

/* Int quant #205 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(tiny_bert_generator_enc_2_ln1_batchnorm_ReadVariableOp_3D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0006968578672967851f),
    AI_PACK_INTQ_ZP(-9)))

/* Int quant #206 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(tiny_bert_generator_enc_2_ln1_batchnorm_mul_ReadVariableOp_3D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0045681544579565525f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #207 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(tiny_bert_generator_enc_2_ln2_batchnorm_ReadVariableOp_3D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0005562865408137441f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #208 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(tiny_bert_generator_enc_2_ln2_batchnorm_mul_ReadVariableOp_3D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0044794874265789986f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #209 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(tiny_bert_generator_enc_2_mha_output_dense_BiasAdd_ReadVariableOp_3D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0006575894076377153f),
    AI_PACK_INTQ_ZP(-8)))

/* Int quant #210 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(tiny_bert_generator_enc_2_mha_wk_BiasAdd_ReadVariableOp_3D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(1.0469741937413346e-05f),
    AI_PACK_INTQ_ZP(-32)))

/* Int quant #211 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(tiny_bert_generator_enc_2_mha_wq_BiasAdd_ReadVariableOp_3D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0004996975767426193f),
    AI_PACK_INTQ_ZP(8)))

/* Int quant #212 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(tiny_bert_generator_enc_2_mha_wv_BiasAdd_ReadVariableOp_3D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0007738408166915178f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #213 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(tiny_bert_generator_enc_3_ffn_dense_30_BiasAdd_ReadVariableOp_3D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0022734589874744415f),
    AI_PACK_INTQ_ZP(-85)))

/* Int quant #214 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(tiny_bert_generator_enc_3_ffn_dense_31_BiasAdd_ReadVariableOp_3D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.004167653154581785f),
    AI_PACK_INTQ_ZP(13)))

/* Int quant #215 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(tiny_bert_generator_enc_3_ln1_batchnorm_ReadVariableOp_3D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0020773466676473618f),
    AI_PACK_INTQ_ZP(-4)))

/* Int quant #216 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(tiny_bert_generator_enc_3_ln1_batchnorm_mul_ReadVariableOp_3D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00498224375769496f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #217 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(tiny_bert_generator_enc_3_ln2_batchnorm_ReadVariableOp_3D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.004190638195723295f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #218 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(tiny_bert_generator_enc_3_ln2_batchnorm_mul_ReadVariableOp_3D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0030835792422294617f),
    AI_PACK_INTQ_ZP(-103)))

/* Int quant #219 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(tiny_bert_generator_enc_3_mha_output_dense_BiasAdd_ReadVariableOp_3D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0006395551608875394f),
    AI_PACK_INTQ_ZP(1)))

/* Int quant #220 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(tiny_bert_generator_enc_3_mha_truedivtiny_bert_generator_enc_3_mha_Sqrt_4D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0004901961074210703f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #221 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(tiny_bert_generator_enc_3_mha_wk_BiasAdd_ReadVariableOp_3D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(2.0773254618688952e-06f),
    AI_PACK_INTQ_ZP(5)))

/* Int quant #222 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(tiny_bert_generator_enc_3_mha_wq_BiasAdd_ReadVariableOp_3D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0007187591982074082f),
    AI_PACK_INTQ_ZP(-17)))

/* Int quant #223 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(tiny_bert_generator_enc_3_mha_wv_BiasAdd_ReadVariableOp_3D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0007022032514214516f),
    AI_PACK_INTQ_ZP(8)))

/* Int quant #224 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(tiny_bert_generator_output_dense_BiasAdd_ReadVariableOp_3D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.005628753453493118f),
    AI_PACK_INTQ_ZP(76)))

/* Int quant #225 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(tiny_bert_generator_tf___operators___add_1_y_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #226 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(tiny_bert_generator_tf_math_multiply_1_Mul_y_3D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.062745101749897f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #227 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(transpose_109_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.09996993094682693f),
    AI_PACK_INTQ_ZP(-3)))

/* Int quant #228 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(transpose_117_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.1256372034549713f),
    AI_PACK_INTQ_ZP(1)))

/* Int quant #229 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(transpose_119_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.12557001411914825f),
    AI_PACK_INTQ_ZP(1)))

/* Int quant #230 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(transpose_196_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.1136530190706253f),
    AI_PACK_INTQ_ZP(-8)))

/* Int quant #231 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(transpose_204_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.13711415231227875f),
    AI_PACK_INTQ_ZP(-10)))

/* Int quant #232 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(transpose_206_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.13711419701576233f),
    AI_PACK_INTQ_ZP(-10)))

/* Int quant #233 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(transpose_22_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.10581283271312714f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #234 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(transpose_283_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.14364778995513916f),
    AI_PACK_INTQ_ZP(-6)))

/* Int quant #235 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(transpose_291_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.21361419558525085f),
    AI_PACK_INTQ_ZP(-2)))

/* Int quant #236 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(transpose_293_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.21361443400382996f),
    AI_PACK_INTQ_ZP(-2)))

/* Int quant #237 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(transpose_30_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.13490118086338043f),
    AI_PACK_INTQ_ZP(-2)))

/* Int quant #238 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(transpose_32_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.13490113615989685f),
    AI_PACK_INTQ_ZP(-2)))

/* Int quant #239 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(transpose_bgemm_110_out_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.057787321507930756f),
    AI_PACK_INTQ_ZP(-12)))

/* Int quant #240 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(transpose_bgemm_197_out_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.08979139477014542f),
    AI_PACK_INTQ_ZP(9)))

/* Int quant #241 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(transpose_bgemm_23_out_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.1098199412226677f),
    AI_PACK_INTQ_ZP(-10)))

/* Int quant #242 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(transpose_bgemm_284_out_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.1260143369436264f),
    AI_PACK_INTQ_ZP(-11)))

/**  Tensor declarations section  *********************************************/
/* Tensor #0 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_0_output, AI_STATIC,
  0, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_0_output_array, &eltwise_0_output_array_intq)

/* Tensor #1 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_102_output, AI_STATIC,
  1, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_102_output_array, &eltwise_102_output_array_intq)

/* Tensor #2 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_102_output0, AI_STATIC,
  2, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 4, 100), AI_STRIDE_INIT(4, 1, 1, 64, 256),
  1, &eltwise_102_output_array, &eltwise_102_output_array_intq)

/* Tensor #3 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_107_output, AI_STATIC,
  3, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_107_output_array, &eltwise_107_output_array_intq)

/* Tensor #4 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_107_output0, AI_STATIC,
  4, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 4, 100), AI_STRIDE_INIT(4, 1, 1, 64, 256),
  1, &eltwise_107_output_array, &eltwise_107_output_array_intq)

/* Tensor #5 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_111_output, AI_STATIC,
  5, 0x1,
  AI_SHAPE_INIT(4, 1, 100, 100, 4), AI_STRIDE_INIT(4, 1, 1, 100, 10000),
  1, &eltwise_111_output_array, &eltwise_111_output_array_intq)

/* Tensor #6 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_115_output, AI_STATIC,
  6, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_115_output_array, &eltwise_115_output_array_intq)

/* Tensor #7 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_115_output0, AI_STATIC,
  7, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 4, 100), AI_STRIDE_INIT(4, 1, 1, 64, 256),
  1, &eltwise_115_output_array, &eltwise_115_output_array_intq)

/* Tensor #8 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_131_output, AI_STATIC,
  8, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_131_output_array, &eltwise_131_output_array_intq)

/* Tensor #9 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_132_0_0_reduce_133_conversion_output, AI_STATIC,
  9, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 4, 4, 1024, 1024),
  1, &eltwise_132_0_0_reduce_133_conversion_output_array, NULL)

/* Tensor #10 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_132_output, AI_STATIC,
  10, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_132_output_array, &eltwise_132_output_array_intq)

/* Tensor #11 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_134_output, AI_STATIC,
  11, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 4, 4, 1024, 1024),
  1, &eltwise_134_output_array, NULL)

/* Tensor #12 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_136_output, AI_STATIC,
  12, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &eltwise_136_output_array, &eltwise_136_output_array_intq)

/* Tensor #13 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_138_output, AI_STATIC,
  13, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_138_output_array, &eltwise_138_output_array_intq)

/* Tensor #14 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_139_output, AI_STATIC,
  14, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_139_output_array, &eltwise_139_output_array_intq)

/* Tensor #15 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_140_output, AI_STATIC,
  15, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_140_output_array, &eltwise_140_output_array_intq)

/* Tensor #16 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_141_output, AI_STATIC,
  16, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_141_output_array, &eltwise_141_output_array_intq)

/* Tensor #17 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_142_output, AI_STATIC,
  17, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_142_output_array, &eltwise_142_output_array_intq)

/* Tensor #18 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_153_output, AI_STATIC,
  18, 0x1,
  AI_SHAPE_INIT(4, 1, 512, 1, 100), AI_STRIDE_INIT(4, 1, 1, 512, 512),
  1, &eltwise_153_output_array, &eltwise_153_output_array_intq)

/* Tensor #19 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_15_output, AI_STATIC,
  19, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_15_output_array, &eltwise_15_output_array_intq)

/* Tensor #20 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_15_output0, AI_STATIC,
  20, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 4, 100), AI_STRIDE_INIT(4, 1, 1, 64, 256),
  1, &eltwise_15_output_array, &eltwise_15_output_array_intq)

/* Tensor #21 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_164_output, AI_STATIC,
  21, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_164_output_array, &eltwise_164_output_array_intq)

/* Tensor #22 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_165_0_0_reduce_166_conversion_output, AI_STATIC,
  22, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 4, 4, 1024, 1024),
  1, &eltwise_165_0_0_reduce_166_conversion_output_array, NULL)

/* Tensor #23 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_165_output, AI_STATIC,
  23, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_165_output_array, &eltwise_165_output_array_intq)

/* Tensor #24 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_167_output, AI_STATIC,
  24, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 4, 4, 1024, 1024),
  1, &eltwise_167_output_array, NULL)

/* Tensor #25 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_169_output, AI_STATIC,
  25, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &eltwise_169_output_array, &eltwise_169_output_array_intq)

/* Tensor #26 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_171_output, AI_STATIC,
  26, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_171_output_array, &eltwise_171_output_array_intq)

/* Tensor #27 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_172_output, AI_STATIC,
  27, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_172_output_array, &eltwise_172_output_array_intq)

/* Tensor #28 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_173_output, AI_STATIC,
  28, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_173_output_array, &eltwise_173_output_array_intq)

/* Tensor #29 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_174_output, AI_STATIC,
  29, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_174_output_array, &eltwise_174_output_array_intq)

/* Tensor #30 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_175_output, AI_STATIC,
  30, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_175_output_array, &eltwise_175_output_array_intq)

/* Tensor #31 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_189_output, AI_STATIC,
  31, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_189_output_array, &eltwise_189_output_array_intq)

/* Tensor #32 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_189_output0, AI_STATIC,
  32, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 4, 100), AI_STRIDE_INIT(4, 1, 1, 64, 256),
  1, &eltwise_189_output_array, &eltwise_189_output_array_intq)

/* Tensor #33 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_194_output, AI_STATIC,
  33, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_194_output_array, &eltwise_194_output_array_intq)

/* Tensor #34 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_194_output0, AI_STATIC,
  34, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 4, 100), AI_STRIDE_INIT(4, 1, 1, 64, 256),
  1, &eltwise_194_output_array, &eltwise_194_output_array_intq)

/* Tensor #35 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_198_output, AI_STATIC,
  35, 0x1,
  AI_SHAPE_INIT(4, 1, 100, 100, 4), AI_STRIDE_INIT(4, 1, 1, 100, 10000),
  1, &eltwise_198_output_array, &eltwise_198_output_array_intq)

/* Tensor #36 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_1_output, AI_STATIC,
  36, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_1_output_array, &eltwise_1_output_array_intq)

/* Tensor #37 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_202_output, AI_STATIC,
  37, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_202_output_array, &eltwise_202_output_array_intq)

/* Tensor #38 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_202_output0, AI_STATIC,
  38, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 4, 100), AI_STRIDE_INIT(4, 1, 1, 64, 256),
  1, &eltwise_202_output_array, &eltwise_202_output_array_intq)

/* Tensor #39 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_20_output, AI_STATIC,
  39, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_20_output_array, &eltwise_20_output_array_intq)

/* Tensor #40 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_20_output0, AI_STATIC,
  40, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 4, 100), AI_STRIDE_INIT(4, 1, 1, 64, 256),
  1, &eltwise_20_output_array, &eltwise_20_output_array_intq)

/* Tensor #41 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_218_output, AI_STATIC,
  41, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_218_output_array, &eltwise_218_output_array_intq)

/* Tensor #42 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_219_0_0_reduce_220_conversion_output, AI_STATIC,
  42, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 4, 4, 1024, 1024),
  1, &eltwise_219_0_0_reduce_220_conversion_output_array, NULL)

/* Tensor #43 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_219_output, AI_STATIC,
  43, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_219_output_array, &eltwise_219_output_array_intq)

/* Tensor #44 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_221_output, AI_STATIC,
  44, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 4, 4, 1024, 1024),
  1, &eltwise_221_output_array, NULL)

/* Tensor #45 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_223_output, AI_STATIC,
  45, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &eltwise_223_output_array, &eltwise_223_output_array_intq)

/* Tensor #46 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_225_output, AI_STATIC,
  46, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_225_output_array, &eltwise_225_output_array_intq)

/* Tensor #47 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_226_output, AI_STATIC,
  47, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_226_output_array, &eltwise_226_output_array_intq)

/* Tensor #48 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_227_output, AI_STATIC,
  48, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_227_output_array, &eltwise_227_output_array_intq)

/* Tensor #49 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_228_output, AI_STATIC,
  49, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_228_output_array, &eltwise_228_output_array_intq)

/* Tensor #50 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_229_output, AI_STATIC,
  50, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_229_output_array, &eltwise_229_output_array_intq)

/* Tensor #51 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_240_output, AI_STATIC,
  51, 0x1,
  AI_SHAPE_INIT(4, 1, 512, 1, 100), AI_STRIDE_INIT(4, 1, 1, 512, 512),
  1, &eltwise_240_output_array, &eltwise_240_output_array_intq)

/* Tensor #52 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_24_output, AI_STATIC,
  52, 0x1,
  AI_SHAPE_INIT(4, 1, 100, 100, 4), AI_STRIDE_INIT(4, 1, 1, 100, 10000),
  1, &eltwise_24_output_array, &eltwise_24_output_array_intq)

/* Tensor #53 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_251_output, AI_STATIC,
  53, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_251_output_array, &eltwise_251_output_array_intq)

/* Tensor #54 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_252_0_0_reduce_253_conversion_output, AI_STATIC,
  54, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 4, 4, 1024, 1024),
  1, &eltwise_252_0_0_reduce_253_conversion_output_array, NULL)

/* Tensor #55 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_252_output, AI_STATIC,
  55, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_252_output_array, &eltwise_252_output_array_intq)

/* Tensor #56 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_254_output, AI_STATIC,
  56, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 4, 4, 1024, 1024),
  1, &eltwise_254_output_array, NULL)

/* Tensor #57 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_256_output, AI_STATIC,
  57, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &eltwise_256_output_array, &eltwise_256_output_array_intq)

/* Tensor #58 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_258_output, AI_STATIC,
  58, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_258_output_array, &eltwise_258_output_array_intq)

/* Tensor #59 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_259_output, AI_STATIC,
  59, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_259_output_array, &eltwise_259_output_array_intq)

/* Tensor #60 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_260_output, AI_STATIC,
  60, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_260_output_array, &eltwise_260_output_array_intq)

/* Tensor #61 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_261_output, AI_STATIC,
  61, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_261_output_array, &eltwise_261_output_array_intq)

/* Tensor #62 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_262_output, AI_STATIC,
  62, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_262_output_array, &eltwise_262_output_array_intq)

/* Tensor #63 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_276_output, AI_STATIC,
  63, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_276_output_array, &eltwise_276_output_array_intq)

/* Tensor #64 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_276_output0, AI_STATIC,
  64, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 4, 100), AI_STRIDE_INIT(4, 1, 1, 64, 256),
  1, &eltwise_276_output_array, &eltwise_276_output_array_intq)

/* Tensor #65 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_281_output, AI_STATIC,
  65, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_281_output_array, &eltwise_281_output_array_intq)

/* Tensor #66 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_281_output0, AI_STATIC,
  66, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 4, 100), AI_STRIDE_INIT(4, 1, 1, 64, 256),
  1, &eltwise_281_output_array, &eltwise_281_output_array_intq)

/* Tensor #67 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_285_output, AI_STATIC,
  67, 0x1,
  AI_SHAPE_INIT(4, 1, 100, 100, 4), AI_STRIDE_INIT(4, 1, 1, 100, 10000),
  1, &eltwise_285_output_array, &eltwise_285_output_array_intq)

/* Tensor #68 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_289_output, AI_STATIC,
  68, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_289_output_array, &eltwise_289_output_array_intq)

/* Tensor #69 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_289_output0, AI_STATIC,
  69, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 4, 100), AI_STRIDE_INIT(4, 1, 1, 64, 256),
  1, &eltwise_289_output_array, &eltwise_289_output_array_intq)

/* Tensor #70 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_28_output, AI_STATIC,
  70, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_28_output_array, &eltwise_28_output_array_intq)

/* Tensor #71 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_28_output0, AI_STATIC,
  71, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 4, 100), AI_STRIDE_INIT(4, 1, 1, 64, 256),
  1, &eltwise_28_output_array, &eltwise_28_output_array_intq)

/* Tensor #72 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_305_output, AI_STATIC,
  72, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_305_output_array, &eltwise_305_output_array_intq)

/* Tensor #73 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_306_0_0_reduce_307_conversion_output, AI_STATIC,
  73, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 4, 4, 1024, 1024),
  1, &eltwise_306_0_0_reduce_307_conversion_output_array, NULL)

/* Tensor #74 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_306_output, AI_STATIC,
  74, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_306_output_array, &eltwise_306_output_array_intq)

/* Tensor #75 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_308_output, AI_STATIC,
  75, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 4, 4, 1024, 1024),
  1, &eltwise_308_output_array, NULL)

/* Tensor #76 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_310_output, AI_STATIC,
  76, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &eltwise_310_output_array, &eltwise_310_output_array_intq)

/* Tensor #77 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_312_output, AI_STATIC,
  77, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_312_output_array, &eltwise_312_output_array_intq)

/* Tensor #78 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_313_output, AI_STATIC,
  78, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_313_output_array, &eltwise_313_output_array_intq)

/* Tensor #79 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_314_output, AI_STATIC,
  79, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_314_output_array, &eltwise_314_output_array_intq)

/* Tensor #80 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_315_output, AI_STATIC,
  80, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_315_output_array, &eltwise_315_output_array_intq)

/* Tensor #81 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_316_output, AI_STATIC,
  81, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_316_output_array, &eltwise_316_output_array_intq)

/* Tensor #82 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_327_output, AI_STATIC,
  82, 0x1,
  AI_SHAPE_INIT(4, 1, 512, 1, 100), AI_STRIDE_INIT(4, 1, 1, 512, 512),
  1, &eltwise_327_output_array, &eltwise_327_output_array_intq)

/* Tensor #83 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_338_output, AI_STATIC,
  83, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_338_output_array, &eltwise_338_output_array_intq)

/* Tensor #84 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_339_0_0_reduce_340_conversion_output, AI_STATIC,
  84, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 4, 4, 1024, 1024),
  1, &eltwise_339_0_0_reduce_340_conversion_output_array, NULL)

/* Tensor #85 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_339_output, AI_STATIC,
  85, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_339_output_array, &eltwise_339_output_array_intq)

/* Tensor #86 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_341_output, AI_STATIC,
  86, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 4, 4, 1024, 1024),
  1, &eltwise_341_output_array, NULL)

/* Tensor #87 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_343_output, AI_STATIC,
  87, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &eltwise_343_output_array, &eltwise_343_output_array_intq)

/* Tensor #88 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_345_output, AI_STATIC,
  88, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_345_output_array, &eltwise_345_output_array_intq)

/* Tensor #89 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_346_output, AI_STATIC,
  89, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_346_output_array, &eltwise_346_output_array_intq)

/* Tensor #90 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_347_output, AI_STATIC,
  90, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_347_output_array, &eltwise_347_output_array_intq)

/* Tensor #91 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_348_output, AI_STATIC,
  91, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_348_output_array, &eltwise_348_output_array_intq)

/* Tensor #92 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_349_output, AI_STATIC,
  92, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_349_output_array, &eltwise_349_output_array_intq)

/* Tensor #93 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_360_output, AI_STATIC,
  93, 0x1,
  AI_SHAPE_INIT(4, 1, 66, 1, 100), AI_STRIDE_INIT(4, 1, 1, 66, 66),
  1, &eltwise_360_output_array, &eltwise_360_output_array_intq)

/* Tensor #94 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_44_output, AI_STATIC,
  94, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_44_output_array, &eltwise_44_output_array_intq)

/* Tensor #95 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_45_0_0_reduce_46_conversion_output, AI_STATIC,
  95, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 4, 4, 1024, 1024),
  1, &eltwise_45_0_0_reduce_46_conversion_output_array, NULL)

/* Tensor #96 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_45_output, AI_STATIC,
  96, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_45_output_array, &eltwise_45_output_array_intq)

/* Tensor #97 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_47_output, AI_STATIC,
  97, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 4, 4, 1024, 1024),
  1, &eltwise_47_output_array, NULL)

/* Tensor #98 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_49_output, AI_STATIC,
  98, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &eltwise_49_output_array, &eltwise_49_output_array_intq)

/* Tensor #99 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_51_output, AI_STATIC,
  99, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_51_output_array, &eltwise_51_output_array_intq)

/* Tensor #100 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_52_output, AI_STATIC,
  100, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_52_output_array, &eltwise_52_output_array_intq)

/* Tensor #101 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_53_output, AI_STATIC,
  101, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_53_output_array, &eltwise_53_output_array_intq)

/* Tensor #102 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_54_output, AI_STATIC,
  102, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_54_output_array, &eltwise_54_output_array_intq)

/* Tensor #103 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_55_output, AI_STATIC,
  103, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_55_output_array, &eltwise_55_output_array_intq)

/* Tensor #104 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_66_output, AI_STATIC,
  104, 0x1,
  AI_SHAPE_INIT(4, 1, 512, 1, 100), AI_STRIDE_INIT(4, 1, 1, 512, 512),
  1, &eltwise_66_output_array, &eltwise_66_output_array_intq)

/* Tensor #105 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_77_output, AI_STATIC,
  105, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_77_output_array, &eltwise_77_output_array_intq)

/* Tensor #106 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_78_0_0_reduce_79_conversion_output, AI_STATIC,
  106, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 4, 4, 1024, 1024),
  1, &eltwise_78_0_0_reduce_79_conversion_output_array, NULL)

/* Tensor #107 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_78_output, AI_STATIC,
  107, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_78_output_array, &eltwise_78_output_array_intq)

/* Tensor #108 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_80_output, AI_STATIC,
  108, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 4, 4, 1024, 1024),
  1, &eltwise_80_output_array, NULL)

/* Tensor #109 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_82_output, AI_STATIC,
  109, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &eltwise_82_output_array, &eltwise_82_output_array_intq)

/* Tensor #110 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_84_output, AI_STATIC,
  110, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_84_output_array, &eltwise_84_output_array_intq)

/* Tensor #111 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_85_output, AI_STATIC,
  111, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_85_output_array, &eltwise_85_output_array_intq)

/* Tensor #112 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_86_output, AI_STATIC,
  112, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_86_output_array, &eltwise_86_output_array_intq)

/* Tensor #113 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_87_output, AI_STATIC,
  113, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_87_output_array, &eltwise_87_output_array_intq)

/* Tensor #114 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_88_output, AI_STATIC,
  114, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_88_output_array, &eltwise_88_output_array_intq)

/* Tensor #115 */
AI_TENSOR_OBJ_DECLARE(
  gemm_100_output, AI_STATIC,
  115, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &gemm_100_output_array, &gemm_100_output_array_intq)

/* Tensor #116 */
AI_TENSOR_OBJ_DECLARE(
  gemm_100_scratch0, AI_STATIC,
  116, 0x0,
  AI_SHAPE_INIT(4, 1, 1024, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1024, 1024),
  1, &gemm_100_scratch0_array, NULL)

/* Tensor #117 */
AI_TENSOR_OBJ_DECLARE(
  gemm_100_weights, AI_STATIC,
  117, 0x1,
  AI_SHAPE_INIT(4, 256, 1, 1, 256), AI_STRIDE_INIT(4, 1, 256, 65536, 65536),
  1, &gemm_100_weights_array, &gemm_100_weights_array_intq)

/* Tensor #118 */
AI_TENSOR_OBJ_DECLARE(
  gemm_105_output, AI_STATIC,
  118, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &gemm_105_output_array, &gemm_105_output_array_intq)

/* Tensor #119 */
AI_TENSOR_OBJ_DECLARE(
  gemm_105_scratch0, AI_STATIC,
  119, 0x0,
  AI_SHAPE_INIT(4, 1, 1024, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1024, 1024),
  1, &gemm_105_scratch0_array, NULL)

/* Tensor #120 */
AI_TENSOR_OBJ_DECLARE(
  gemm_105_weights, AI_STATIC,
  120, 0x1,
  AI_SHAPE_INIT(4, 256, 1, 1, 256), AI_STRIDE_INIT(4, 1, 256, 65536, 65536),
  1, &gemm_105_weights_array, &gemm_105_weights_array_intq)

/* Tensor #121 */
AI_TENSOR_OBJ_DECLARE(
  gemm_110_0_0_eltwise_111_conversion_output, AI_STATIC,
  121, 0x1,
  AI_SHAPE_INIT(4, 1, 100, 100, 4), AI_STRIDE_INIT(4, 1, 1, 100, 10000),
  1, &gemm_110_0_0_eltwise_111_conversion_output_array, &gemm_110_0_0_eltwise_111_conversion_output_array_intq)

/* Tensor #122 */
AI_TENSOR_OBJ_DECLARE(
  gemm_110_bias, AI_STATIC,
  122, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &gemm_110_bias_array, &gemm_110_bias_array_intq)

/* Tensor #123 */
AI_TENSOR_OBJ_DECLARE(
  gemm_110_bias_0_2_gemm_110_conversion_output, AI_STATIC,
  123, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &gemm_110_bias_0_2_gemm_110_conversion_output_array, NULL)

/* Tensor #124 */
AI_TENSOR_OBJ_DECLARE(
  gemm_110_output, AI_STATIC,
  124, 0x0,
  AI_SHAPE_INIT(4, 1, 100, 100, 4), AI_STRIDE_INIT(4, 4, 4, 400, 40000),
  1, &gemm_110_output_array, NULL)

/* Tensor #125 */
AI_TENSOR_OBJ_DECLARE(
  gemm_113_output, AI_STATIC,
  125, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &gemm_113_output_array, &gemm_113_output_array_intq)

/* Tensor #126 */
AI_TENSOR_OBJ_DECLARE(
  gemm_113_scratch0, AI_STATIC,
  126, 0x0,
  AI_SHAPE_INIT(4, 1, 1024, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1024, 1024),
  1, &gemm_113_scratch0_array, NULL)

/* Tensor #127 */
AI_TENSOR_OBJ_DECLARE(
  gemm_113_weights, AI_STATIC,
  127, 0x1,
  AI_SHAPE_INIT(4, 256, 1, 1, 256), AI_STRIDE_INIT(4, 1, 256, 65536, 65536),
  1, &gemm_113_weights_array, &gemm_113_weights_array_intq)

/* Tensor #128 */
AI_TENSOR_OBJ_DECLARE(
  gemm_118_0_0_transpose_119_conversion_output, AI_STATIC,
  128, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 100, 4), AI_STRIDE_INIT(4, 1, 1, 64, 6400),
  1, &gemm_118_0_0_transpose_119_conversion_output_array, &gemm_118_0_0_transpose_119_conversion_output_array_intq)

/* Tensor #129 */
AI_TENSOR_OBJ_DECLARE(
  gemm_118_bias, AI_STATIC,
  129, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &gemm_118_bias_array, &gemm_118_bias_array_intq)

/* Tensor #130 */
AI_TENSOR_OBJ_DECLARE(
  gemm_118_bias_0_2_gemm_118_conversion_output, AI_STATIC,
  130, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &gemm_118_bias_0_2_gemm_118_conversion_output_array, NULL)

/* Tensor #131 */
AI_TENSOR_OBJ_DECLARE(
  gemm_118_output, AI_STATIC,
  131, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 100, 4), AI_STRIDE_INIT(4, 4, 4, 256, 25600),
  1, &gemm_118_output_array, NULL)

/* Tensor #132 */
AI_TENSOR_OBJ_DECLARE(
  gemm_129_output, AI_STATIC,
  132, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &gemm_129_output_array, &gemm_129_output_array_intq)

/* Tensor #133 */
AI_TENSOR_OBJ_DECLARE(
  gemm_129_scratch0, AI_STATIC,
  133, 0x0,
  AI_SHAPE_INIT(4, 1, 1024, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1024, 1024),
  1, &gemm_129_scratch0_array, NULL)

/* Tensor #134 */
AI_TENSOR_OBJ_DECLARE(
  gemm_129_weights, AI_STATIC,
  134, 0x1,
  AI_SHAPE_INIT(4, 256, 1, 1, 256), AI_STRIDE_INIT(4, 1, 256, 65536, 65536),
  1, &gemm_129_weights_array, &gemm_129_weights_array_intq)

/* Tensor #135 */
AI_TENSOR_OBJ_DECLARE(
  gemm_13_output, AI_STATIC,
  135, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &gemm_13_output_array, &gemm_13_output_array_intq)

/* Tensor #136 */
AI_TENSOR_OBJ_DECLARE(
  gemm_13_scratch0, AI_STATIC,
  136, 0x0,
  AI_SHAPE_INIT(4, 1, 1024, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1024, 1024),
  1, &gemm_13_scratch0_array, NULL)

/* Tensor #137 */
AI_TENSOR_OBJ_DECLARE(
  gemm_13_weights, AI_STATIC,
  137, 0x1,
  AI_SHAPE_INIT(4, 256, 1, 1, 256), AI_STRIDE_INIT(4, 1, 256, 65536, 65536),
  1, &gemm_13_weights_array, &gemm_13_weights_array_intq)

/* Tensor #138 */
AI_TENSOR_OBJ_DECLARE(
  gemm_151_output, AI_STATIC,
  138, 0x1,
  AI_SHAPE_INIT(4, 1, 512, 1, 100), AI_STRIDE_INIT(4, 1, 1, 512, 512),
  1, &gemm_151_output_array, &gemm_151_output_array_intq)

/* Tensor #139 */
AI_TENSOR_OBJ_DECLARE(
  gemm_151_scratch0, AI_STATIC,
  139, 0x0,
  AI_SHAPE_INIT(4, 1, 1024, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1024, 1024),
  1, &gemm_151_scratch0_array, NULL)

/* Tensor #140 */
AI_TENSOR_OBJ_DECLARE(
  gemm_151_weights, AI_STATIC,
  140, 0x1,
  AI_SHAPE_INIT(4, 256, 1, 1, 512), AI_STRIDE_INIT(4, 1, 256, 131072, 131072),
  1, &gemm_151_weights_array, &gemm_151_weights_array_intq)

/* Tensor #141 */
AI_TENSOR_OBJ_DECLARE(
  gemm_162_output, AI_STATIC,
  141, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &gemm_162_output_array, &gemm_162_output_array_intq)

/* Tensor #142 */
AI_TENSOR_OBJ_DECLARE(
  gemm_162_scratch0, AI_STATIC,
  142, 0x0,
  AI_SHAPE_INIT(4, 1, 2048, 1, 1), AI_STRIDE_INIT(4, 1, 1, 2048, 2048),
  1, &gemm_162_scratch0_array, NULL)

/* Tensor #143 */
AI_TENSOR_OBJ_DECLARE(
  gemm_162_weights, AI_STATIC,
  143, 0x1,
  AI_SHAPE_INIT(4, 512, 1, 1, 256), AI_STRIDE_INIT(4, 1, 512, 131072, 131072),
  1, &gemm_162_weights_array, &gemm_162_weights_array_intq)

/* Tensor #144 */
AI_TENSOR_OBJ_DECLARE(
  gemm_187_output, AI_STATIC,
  144, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &gemm_187_output_array, &gemm_187_output_array_intq)

/* Tensor #145 */
AI_TENSOR_OBJ_DECLARE(
  gemm_187_scratch0, AI_STATIC,
  145, 0x0,
  AI_SHAPE_INIT(4, 1, 1024, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1024, 1024),
  1, &gemm_187_scratch0_array, NULL)

/* Tensor #146 */
AI_TENSOR_OBJ_DECLARE(
  gemm_187_weights, AI_STATIC,
  146, 0x1,
  AI_SHAPE_INIT(4, 256, 1, 1, 256), AI_STRIDE_INIT(4, 1, 256, 65536, 65536),
  1, &gemm_187_weights_array, &gemm_187_weights_array_intq)

/* Tensor #147 */
AI_TENSOR_OBJ_DECLARE(
  gemm_18_output, AI_STATIC,
  147, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &gemm_18_output_array, &gemm_18_output_array_intq)

/* Tensor #148 */
AI_TENSOR_OBJ_DECLARE(
  gemm_18_scratch0, AI_STATIC,
  148, 0x0,
  AI_SHAPE_INIT(4, 1, 1024, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1024, 1024),
  1, &gemm_18_scratch0_array, NULL)

/* Tensor #149 */
AI_TENSOR_OBJ_DECLARE(
  gemm_18_weights, AI_STATIC,
  149, 0x1,
  AI_SHAPE_INIT(4, 256, 1, 1, 256), AI_STRIDE_INIT(4, 1, 256, 65536, 65536),
  1, &gemm_18_weights_array, &gemm_18_weights_array_intq)

/* Tensor #150 */
AI_TENSOR_OBJ_DECLARE(
  gemm_192_output, AI_STATIC,
  150, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &gemm_192_output_array, &gemm_192_output_array_intq)

/* Tensor #151 */
AI_TENSOR_OBJ_DECLARE(
  gemm_192_scratch0, AI_STATIC,
  151, 0x0,
  AI_SHAPE_INIT(4, 1, 1024, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1024, 1024),
  1, &gemm_192_scratch0_array, NULL)

/* Tensor #152 */
AI_TENSOR_OBJ_DECLARE(
  gemm_192_weights, AI_STATIC,
  152, 0x1,
  AI_SHAPE_INIT(4, 256, 1, 1, 256), AI_STRIDE_INIT(4, 1, 256, 65536, 65536),
  1, &gemm_192_weights_array, &gemm_192_weights_array_intq)

/* Tensor #153 */
AI_TENSOR_OBJ_DECLARE(
  gemm_197_0_0_eltwise_198_conversion_output, AI_STATIC,
  153, 0x1,
  AI_SHAPE_INIT(4, 1, 100, 100, 4), AI_STRIDE_INIT(4, 1, 1, 100, 10000),
  1, &gemm_197_0_0_eltwise_198_conversion_output_array, &gemm_197_0_0_eltwise_198_conversion_output_array_intq)

/* Tensor #154 */
AI_TENSOR_OBJ_DECLARE(
  gemm_197_bias, AI_STATIC,
  154, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &gemm_197_bias_array, &gemm_197_bias_array_intq)

/* Tensor #155 */
AI_TENSOR_OBJ_DECLARE(
  gemm_197_bias_0_2_gemm_197_conversion_output, AI_STATIC,
  155, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &gemm_197_bias_0_2_gemm_197_conversion_output_array, NULL)

/* Tensor #156 */
AI_TENSOR_OBJ_DECLARE(
  gemm_197_output, AI_STATIC,
  156, 0x0,
  AI_SHAPE_INIT(4, 1, 100, 100, 4), AI_STRIDE_INIT(4, 4, 4, 400, 40000),
  1, &gemm_197_output_array, NULL)

/* Tensor #157 */
AI_TENSOR_OBJ_DECLARE(
  gemm_200_output, AI_STATIC,
  157, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &gemm_200_output_array, &gemm_200_output_array_intq)

/* Tensor #158 */
AI_TENSOR_OBJ_DECLARE(
  gemm_200_scratch0, AI_STATIC,
  158, 0x0,
  AI_SHAPE_INIT(4, 1, 1024, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1024, 1024),
  1, &gemm_200_scratch0_array, NULL)

/* Tensor #159 */
AI_TENSOR_OBJ_DECLARE(
  gemm_200_weights, AI_STATIC,
  159, 0x1,
  AI_SHAPE_INIT(4, 256, 1, 1, 256), AI_STRIDE_INIT(4, 1, 256, 65536, 65536),
  1, &gemm_200_weights_array, &gemm_200_weights_array_intq)

/* Tensor #160 */
AI_TENSOR_OBJ_DECLARE(
  gemm_205_0_0_transpose_206_conversion_output, AI_STATIC,
  160, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 100, 4), AI_STRIDE_INIT(4, 1, 1, 64, 6400),
  1, &gemm_205_0_0_transpose_206_conversion_output_array, &gemm_205_0_0_transpose_206_conversion_output_array_intq)

/* Tensor #161 */
AI_TENSOR_OBJ_DECLARE(
  gemm_205_bias, AI_STATIC,
  161, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &gemm_205_bias_array, &gemm_205_bias_array_intq)

/* Tensor #162 */
AI_TENSOR_OBJ_DECLARE(
  gemm_205_bias_0_2_gemm_205_conversion_output, AI_STATIC,
  162, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &gemm_205_bias_0_2_gemm_205_conversion_output_array, NULL)

/* Tensor #163 */
AI_TENSOR_OBJ_DECLARE(
  gemm_205_output, AI_STATIC,
  163, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 100, 4), AI_STRIDE_INIT(4, 4, 4, 256, 25600),
  1, &gemm_205_output_array, NULL)

/* Tensor #164 */
AI_TENSOR_OBJ_DECLARE(
  gemm_216_output, AI_STATIC,
  164, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &gemm_216_output_array, &gemm_216_output_array_intq)

/* Tensor #165 */
AI_TENSOR_OBJ_DECLARE(
  gemm_216_scratch0, AI_STATIC,
  165, 0x0,
  AI_SHAPE_INIT(4, 1, 1024, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1024, 1024),
  1, &gemm_216_scratch0_array, NULL)

/* Tensor #166 */
AI_TENSOR_OBJ_DECLARE(
  gemm_216_weights, AI_STATIC,
  166, 0x1,
  AI_SHAPE_INIT(4, 256, 1, 1, 256), AI_STRIDE_INIT(4, 1, 256, 65536, 65536),
  1, &gemm_216_weights_array, &gemm_216_weights_array_intq)

/* Tensor #167 */
AI_TENSOR_OBJ_DECLARE(
  gemm_238_output, AI_STATIC,
  167, 0x1,
  AI_SHAPE_INIT(4, 1, 512, 1, 100), AI_STRIDE_INIT(4, 1, 1, 512, 512),
  1, &gemm_238_output_array, &gemm_238_output_array_intq)

/* Tensor #168 */
AI_TENSOR_OBJ_DECLARE(
  gemm_238_scratch0, AI_STATIC,
  168, 0x0,
  AI_SHAPE_INIT(4, 1, 1024, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1024, 1024),
  1, &gemm_238_scratch0_array, NULL)

/* Tensor #169 */
AI_TENSOR_OBJ_DECLARE(
  gemm_238_weights, AI_STATIC,
  169, 0x1,
  AI_SHAPE_INIT(4, 256, 1, 1, 512), AI_STRIDE_INIT(4, 1, 256, 131072, 131072),
  1, &gemm_238_weights_array, &gemm_238_weights_array_intq)

/* Tensor #170 */
AI_TENSOR_OBJ_DECLARE(
  gemm_23_0_0_eltwise_24_conversion_output, AI_STATIC,
  170, 0x1,
  AI_SHAPE_INIT(4, 1, 100, 100, 4), AI_STRIDE_INIT(4, 1, 1, 100, 10000),
  1, &gemm_23_0_0_eltwise_24_conversion_output_array, &gemm_23_0_0_eltwise_24_conversion_output_array_intq)

/* Tensor #171 */
AI_TENSOR_OBJ_DECLARE(
  gemm_23_bias, AI_STATIC,
  171, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &gemm_23_bias_array, &gemm_23_bias_array_intq)

/* Tensor #172 */
AI_TENSOR_OBJ_DECLARE(
  gemm_23_bias_0_2_gemm_23_conversion_output, AI_STATIC,
  172, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &gemm_23_bias_0_2_gemm_23_conversion_output_array, NULL)

/* Tensor #173 */
AI_TENSOR_OBJ_DECLARE(
  gemm_23_output, AI_STATIC,
  173, 0x0,
  AI_SHAPE_INIT(4, 1, 100, 100, 4), AI_STRIDE_INIT(4, 4, 4, 400, 40000),
  1, &gemm_23_output_array, NULL)

/* Tensor #174 */
AI_TENSOR_OBJ_DECLARE(
  gemm_249_output, AI_STATIC,
  174, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &gemm_249_output_array, &gemm_249_output_array_intq)

/* Tensor #175 */
AI_TENSOR_OBJ_DECLARE(
  gemm_249_scratch0, AI_STATIC,
  175, 0x0,
  AI_SHAPE_INIT(4, 1, 2048, 1, 1), AI_STRIDE_INIT(4, 1, 1, 2048, 2048),
  1, &gemm_249_scratch0_array, NULL)

/* Tensor #176 */
AI_TENSOR_OBJ_DECLARE(
  gemm_249_weights, AI_STATIC,
  176, 0x1,
  AI_SHAPE_INIT(4, 512, 1, 1, 256), AI_STRIDE_INIT(4, 1, 512, 131072, 131072),
  1, &gemm_249_weights_array, &gemm_249_weights_array_intq)

/* Tensor #177 */
AI_TENSOR_OBJ_DECLARE(
  gemm_26_bias, AI_STATIC,
  177, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 4, 4, 1024, 1024),
  1, &gemm_26_bias_array, NULL)

/* Tensor #178 */
AI_TENSOR_OBJ_DECLARE(
  gemm_26_output, AI_STATIC,
  178, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &gemm_26_output_array, &gemm_26_output_array_intq)

/* Tensor #179 */
AI_TENSOR_OBJ_DECLARE(
  gemm_26_scratch0, AI_STATIC,
  179, 0x0,
  AI_SHAPE_INIT(4, 1, 1024, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1024, 1024),
  1, &gemm_26_scratch0_array, NULL)

/* Tensor #180 */
AI_TENSOR_OBJ_DECLARE(
  gemm_26_weights, AI_STATIC,
  180, 0x1,
  AI_SHAPE_INIT(4, 256, 1, 1, 256), AI_STRIDE_INIT(4, 1, 256, 65536, 65536),
  1, &gemm_26_weights_array, &gemm_26_weights_array_intq)

/* Tensor #181 */
AI_TENSOR_OBJ_DECLARE(
  gemm_274_output, AI_STATIC,
  181, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &gemm_274_output_array, &gemm_274_output_array_intq)

/* Tensor #182 */
AI_TENSOR_OBJ_DECLARE(
  gemm_274_scratch0, AI_STATIC,
  182, 0x0,
  AI_SHAPE_INIT(4, 1, 1024, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1024, 1024),
  1, &gemm_274_scratch0_array, NULL)

/* Tensor #183 */
AI_TENSOR_OBJ_DECLARE(
  gemm_274_weights, AI_STATIC,
  183, 0x1,
  AI_SHAPE_INIT(4, 256, 1, 1, 256), AI_STRIDE_INIT(4, 1, 256, 65536, 65536),
  1, &gemm_274_weights_array, &gemm_274_weights_array_intq)

/* Tensor #184 */
AI_TENSOR_OBJ_DECLARE(
  gemm_279_output, AI_STATIC,
  184, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &gemm_279_output_array, &gemm_279_output_array_intq)

/* Tensor #185 */
AI_TENSOR_OBJ_DECLARE(
  gemm_279_scratch0, AI_STATIC,
  185, 0x0,
  AI_SHAPE_INIT(4, 1, 1024, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1024, 1024),
  1, &gemm_279_scratch0_array, NULL)

/* Tensor #186 */
AI_TENSOR_OBJ_DECLARE(
  gemm_279_weights, AI_STATIC,
  186, 0x1,
  AI_SHAPE_INIT(4, 256, 1, 1, 256), AI_STRIDE_INIT(4, 1, 256, 65536, 65536),
  1, &gemm_279_weights_array, &gemm_279_weights_array_intq)

/* Tensor #187 */
AI_TENSOR_OBJ_DECLARE(
  gemm_284_0_0_eltwise_285_conversion_output, AI_STATIC,
  187, 0x1,
  AI_SHAPE_INIT(4, 1, 100, 100, 4), AI_STRIDE_INIT(4, 1, 1, 100, 10000),
  1, &gemm_284_0_0_eltwise_285_conversion_output_array, &gemm_284_0_0_eltwise_285_conversion_output_array_intq)

/* Tensor #188 */
AI_TENSOR_OBJ_DECLARE(
  gemm_284_bias, AI_STATIC,
  188, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &gemm_284_bias_array, &gemm_284_bias_array_intq)

/* Tensor #189 */
AI_TENSOR_OBJ_DECLARE(
  gemm_284_bias_0_2_gemm_284_conversion_output, AI_STATIC,
  189, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &gemm_284_bias_0_2_gemm_284_conversion_output_array, NULL)

/* Tensor #190 */
AI_TENSOR_OBJ_DECLARE(
  gemm_284_output, AI_STATIC,
  190, 0x0,
  AI_SHAPE_INIT(4, 1, 100, 100, 4), AI_STRIDE_INIT(4, 4, 4, 400, 40000),
  1, &gemm_284_output_array, NULL)

/* Tensor #191 */
AI_TENSOR_OBJ_DECLARE(
  gemm_287_output, AI_STATIC,
  191, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &gemm_287_output_array, &gemm_287_output_array_intq)

/* Tensor #192 */
AI_TENSOR_OBJ_DECLARE(
  gemm_287_scratch0, AI_STATIC,
  192, 0x0,
  AI_SHAPE_INIT(4, 1, 1024, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1024, 1024),
  1, &gemm_287_scratch0_array, NULL)

/* Tensor #193 */
AI_TENSOR_OBJ_DECLARE(
  gemm_287_weights, AI_STATIC,
  193, 0x1,
  AI_SHAPE_INIT(4, 256, 1, 1, 256), AI_STRIDE_INIT(4, 1, 256, 65536, 65536),
  1, &gemm_287_weights_array, &gemm_287_weights_array_intq)

/* Tensor #194 */
AI_TENSOR_OBJ_DECLARE(
  gemm_292_0_0_transpose_293_conversion_output, AI_STATIC,
  194, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 100, 4), AI_STRIDE_INIT(4, 1, 1, 64, 6400),
  1, &gemm_292_0_0_transpose_293_conversion_output_array, &gemm_292_0_0_transpose_293_conversion_output_array_intq)

/* Tensor #195 */
AI_TENSOR_OBJ_DECLARE(
  gemm_292_bias, AI_STATIC,
  195, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &gemm_292_bias_array, &gemm_292_bias_array_intq)

/* Tensor #196 */
AI_TENSOR_OBJ_DECLARE(
  gemm_292_bias_0_conversion_output, AI_STATIC,
  196, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &gemm_292_bias_0_conversion_output_array, NULL)

/* Tensor #197 */
AI_TENSOR_OBJ_DECLARE(
  gemm_292_output, AI_STATIC,
  197, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 100, 4), AI_STRIDE_INIT(4, 4, 4, 256, 25600),
  1, &gemm_292_output_array, NULL)

/* Tensor #198 */
AI_TENSOR_OBJ_DECLARE(
  gemm_303_output, AI_STATIC,
  198, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &gemm_303_output_array, &gemm_303_output_array_intq)

/* Tensor #199 */
AI_TENSOR_OBJ_DECLARE(
  gemm_303_scratch0, AI_STATIC,
  199, 0x0,
  AI_SHAPE_INIT(4, 1, 1024, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1024, 1024),
  1, &gemm_303_scratch0_array, NULL)

/* Tensor #200 */
AI_TENSOR_OBJ_DECLARE(
  gemm_303_weights, AI_STATIC,
  200, 0x1,
  AI_SHAPE_INIT(4, 256, 1, 1, 256), AI_STRIDE_INIT(4, 1, 256, 65536, 65536),
  1, &gemm_303_weights_array, &gemm_303_weights_array_intq)

/* Tensor #201 */
AI_TENSOR_OBJ_DECLARE(
  gemm_31_0_0_transpose_32_conversion_output, AI_STATIC,
  201, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 100, 4), AI_STRIDE_INIT(4, 1, 1, 64, 6400),
  1, &gemm_31_0_0_transpose_32_conversion_output_array, &gemm_31_0_0_transpose_32_conversion_output_array_intq)

/* Tensor #202 */
AI_TENSOR_OBJ_DECLARE(
  gemm_31_bias, AI_STATIC,
  202, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &gemm_31_bias_array, &gemm_31_bias_array_intq)

/* Tensor #203 */
AI_TENSOR_OBJ_DECLARE(
  gemm_31_bias_0_2_gemm_31_conversion_output, AI_STATIC,
  203, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &gemm_31_bias_0_2_gemm_31_conversion_output_array, NULL)

/* Tensor #204 */
AI_TENSOR_OBJ_DECLARE(
  gemm_31_output, AI_STATIC,
  204, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 100, 4), AI_STRIDE_INIT(4, 4, 4, 256, 25600),
  1, &gemm_31_output_array, NULL)

/* Tensor #205 */
AI_TENSOR_OBJ_DECLARE(
  gemm_325_output, AI_STATIC,
  205, 0x1,
  AI_SHAPE_INIT(4, 1, 512, 1, 100), AI_STRIDE_INIT(4, 1, 1, 512, 512),
  1, &gemm_325_output_array, &gemm_325_output_array_intq)

/* Tensor #206 */
AI_TENSOR_OBJ_DECLARE(
  gemm_325_scratch0, AI_STATIC,
  206, 0x0,
  AI_SHAPE_INIT(4, 1, 1024, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1024, 1024),
  1, &gemm_325_scratch0_array, NULL)

/* Tensor #207 */
AI_TENSOR_OBJ_DECLARE(
  gemm_325_weights, AI_STATIC,
  207, 0x1,
  AI_SHAPE_INIT(4, 256, 1, 1, 512), AI_STRIDE_INIT(4, 1, 256, 131072, 131072),
  1, &gemm_325_weights_array, &gemm_325_weights_array_intq)

/* Tensor #208 */
AI_TENSOR_OBJ_DECLARE(
  gemm_336_output, AI_STATIC,
  208, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &gemm_336_output_array, &gemm_336_output_array_intq)

/* Tensor #209 */
AI_TENSOR_OBJ_DECLARE(
  gemm_336_scratch0, AI_STATIC,
  209, 0x0,
  AI_SHAPE_INIT(4, 1, 2048, 1, 1), AI_STRIDE_INIT(4, 1, 1, 2048, 2048),
  1, &gemm_336_scratch0_array, NULL)

/* Tensor #210 */
AI_TENSOR_OBJ_DECLARE(
  gemm_336_weights, AI_STATIC,
  210, 0x1,
  AI_SHAPE_INIT(4, 512, 1, 1, 256), AI_STRIDE_INIT(4, 1, 512, 131072, 131072),
  1, &gemm_336_weights_array, &gemm_336_weights_array_intq)

/* Tensor #211 */
AI_TENSOR_OBJ_DECLARE(
  gemm_358_bias, AI_STATIC,
  211, 0x0,
  AI_SHAPE_INIT(4, 1, 66, 1, 1), AI_STRIDE_INIT(4, 4, 4, 264, 264),
  1, &gemm_358_bias_array, NULL)

/* Tensor #212 */
AI_TENSOR_OBJ_DECLARE(
  gemm_358_output, AI_STATIC,
  212, 0x1,
  AI_SHAPE_INIT(4, 1, 66, 1, 100), AI_STRIDE_INIT(4, 1, 1, 66, 66),
  1, &gemm_358_output_array, &gemm_358_output_array_intq)

/* Tensor #213 */
AI_TENSOR_OBJ_DECLARE(
  gemm_358_scratch0, AI_STATIC,
  213, 0x0,
  AI_SHAPE_INIT(4, 1, 1024, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1024, 1024),
  1, &gemm_358_scratch0_array, NULL)

/* Tensor #214 */
AI_TENSOR_OBJ_DECLARE(
  gemm_358_weights, AI_STATIC,
  214, 0x1,
  AI_SHAPE_INIT(4, 256, 1, 1, 66), AI_STRIDE_INIT(4, 1, 256, 16896, 16896),
  1, &gemm_358_weights_array, &gemm_358_weights_array_intq)

/* Tensor #215 */
AI_TENSOR_OBJ_DECLARE(
  gemm_42_output, AI_STATIC,
  215, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &gemm_42_output_array, &gemm_42_output_array_intq)

/* Tensor #216 */
AI_TENSOR_OBJ_DECLARE(
  gemm_42_scratch0, AI_STATIC,
  216, 0x0,
  AI_SHAPE_INIT(4, 1, 1024, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1024, 1024),
  1, &gemm_42_scratch0_array, NULL)

/* Tensor #217 */
AI_TENSOR_OBJ_DECLARE(
  gemm_42_weights, AI_STATIC,
  217, 0x1,
  AI_SHAPE_INIT(4, 256, 1, 1, 256), AI_STRIDE_INIT(4, 1, 256, 65536, 65536),
  1, &gemm_42_weights_array, &gemm_42_weights_array_intq)

/* Tensor #218 */
AI_TENSOR_OBJ_DECLARE(
  gemm_64_bias, AI_STATIC,
  218, 0x0,
  AI_SHAPE_INIT(4, 1, 512, 1, 1), AI_STRIDE_INIT(4, 4, 4, 2048, 2048),
  1, &gemm_64_bias_array, NULL)

/* Tensor #219 */
AI_TENSOR_OBJ_DECLARE(
  gemm_64_output, AI_STATIC,
  219, 0x1,
  AI_SHAPE_INIT(4, 1, 512, 1, 100), AI_STRIDE_INIT(4, 1, 1, 512, 512),
  1, &gemm_64_output_array, &gemm_64_output_array_intq)

/* Tensor #220 */
AI_TENSOR_OBJ_DECLARE(
  gemm_64_scratch0, AI_STATIC,
  220, 0x0,
  AI_SHAPE_INIT(4, 1, 1024, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1024, 1024),
  1, &gemm_64_scratch0_array, NULL)

/* Tensor #221 */
AI_TENSOR_OBJ_DECLARE(
  gemm_64_weights, AI_STATIC,
  221, 0x1,
  AI_SHAPE_INIT(4, 256, 1, 1, 512), AI_STRIDE_INIT(4, 1, 256, 131072, 131072),
  1, &gemm_64_weights_array, &gemm_64_weights_array_intq)

/* Tensor #222 */
AI_TENSOR_OBJ_DECLARE(
  gemm_75_output, AI_STATIC,
  222, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &gemm_75_output_array, &gemm_75_output_array_intq)

/* Tensor #223 */
AI_TENSOR_OBJ_DECLARE(
  gemm_75_scratch0, AI_STATIC,
  223, 0x0,
  AI_SHAPE_INIT(4, 1, 2048, 1, 1), AI_STRIDE_INIT(4, 1, 1, 2048, 2048),
  1, &gemm_75_scratch0_array, NULL)

/* Tensor #224 */
AI_TENSOR_OBJ_DECLARE(
  gemm_75_weights, AI_STATIC,
  224, 0x1,
  AI_SHAPE_INIT(4, 512, 1, 1, 256), AI_STRIDE_INIT(4, 1, 512, 131072, 131072),
  1, &gemm_75_weights_array, &gemm_75_weights_array_intq)

/* Tensor #225 */
AI_TENSOR_OBJ_DECLARE(
  nl_112_0_0_gemm_118_conversion_output, AI_STATIC,
  225, 0x0,
  AI_SHAPE_INIT(4, 1, 100, 100, 4), AI_STRIDE_INIT(4, 4, 4, 400, 40000),
  1, &nl_112_0_0_gemm_118_conversion_output_array, NULL)

/* Tensor #226 */
AI_TENSOR_OBJ_DECLARE(
  nl_112_output, AI_STATIC,
  226, 0x1,
  AI_SHAPE_INIT(4, 1, 100, 100, 4), AI_STRIDE_INIT(4, 1, 1, 100, 10000),
  1, &nl_112_output_array, &nl_112_output_array_intq)

/* Tensor #227 */
AI_TENSOR_OBJ_DECLARE(
  nl_112_scratch0, AI_STATIC,
  227, 0x0,
  AI_SHAPE_INIT(4, 1, 62, 1, 1), AI_STRIDE_INIT(4, 4, 4, 248, 248),
  1, &nl_112_scratch0_array, NULL)

/* Tensor #228 */
AI_TENSOR_OBJ_DECLARE(
  nl_137_output, AI_STATIC,
  228, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &nl_137_output_array, &nl_137_output_array_intq)

/* Tensor #229 */
AI_TENSOR_OBJ_DECLARE(
  nl_170_output, AI_STATIC,
  229, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &nl_170_output_array, &nl_170_output_array_intq)

/* Tensor #230 */
AI_TENSOR_OBJ_DECLARE(
  nl_199_0_0_gemm_205_conversion_output, AI_STATIC,
  230, 0x0,
  AI_SHAPE_INIT(4, 1, 100, 100, 4), AI_STRIDE_INIT(4, 4, 4, 400, 40000),
  1, &nl_199_0_0_gemm_205_conversion_output_array, NULL)

/* Tensor #231 */
AI_TENSOR_OBJ_DECLARE(
  nl_199_output, AI_STATIC,
  231, 0x1,
  AI_SHAPE_INIT(4, 1, 100, 100, 4), AI_STRIDE_INIT(4, 1, 1, 100, 10000),
  1, &nl_199_output_array, &nl_199_output_array_intq)

/* Tensor #232 */
AI_TENSOR_OBJ_DECLARE(
  nl_199_scratch0, AI_STATIC,
  232, 0x0,
  AI_SHAPE_INIT(4, 1, 31, 1, 1), AI_STRIDE_INIT(4, 4, 4, 124, 124),
  1, &nl_199_scratch0_array, NULL)

/* Tensor #233 */
AI_TENSOR_OBJ_DECLARE(
  nl_224_output, AI_STATIC,
  233, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &nl_224_output_array, &nl_224_output_array_intq)

/* Tensor #234 */
AI_TENSOR_OBJ_DECLARE(
  nl_257_output, AI_STATIC,
  234, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &nl_257_output_array, &nl_257_output_array_intq)

/* Tensor #235 */
AI_TENSOR_OBJ_DECLARE(
  nl_25_0_0_gemm_31_conversion_output, AI_STATIC,
  235, 0x0,
  AI_SHAPE_INIT(4, 1, 100, 100, 4), AI_STRIDE_INIT(4, 4, 4, 400, 40000),
  1, &nl_25_0_0_gemm_31_conversion_output_array, NULL)

/* Tensor #236 */
AI_TENSOR_OBJ_DECLARE(
  nl_25_output, AI_STATIC,
  236, 0x1,
  AI_SHAPE_INIT(4, 1, 100, 100, 4), AI_STRIDE_INIT(4, 1, 1, 100, 10000),
  1, &nl_25_output_array, &nl_25_output_array_intq)

/* Tensor #237 */
AI_TENSOR_OBJ_DECLARE(
  nl_25_scratch0, AI_STATIC,
  237, 0x0,
  AI_SHAPE_INIT(4, 1, 7, 1, 1), AI_STRIDE_INIT(4, 4, 4, 28, 28),
  1, &nl_25_scratch0_array, NULL)

/* Tensor #238 */
AI_TENSOR_OBJ_DECLARE(
  nl_286_0_0_gemm_292_conversion_output, AI_STATIC,
  238, 0x0,
  AI_SHAPE_INIT(4, 1, 100, 100, 4), AI_STRIDE_INIT(4, 4, 4, 400, 40000),
  1, &nl_286_0_0_gemm_292_conversion_output_array, NULL)

/* Tensor #239 */
AI_TENSOR_OBJ_DECLARE(
  nl_286_output, AI_STATIC,
  239, 0x1,
  AI_SHAPE_INIT(4, 1, 100, 100, 4), AI_STRIDE_INIT(4, 1, 1, 100, 10000),
  1, &nl_286_output_array, &nl_286_output_array_intq)

/* Tensor #240 */
AI_TENSOR_OBJ_DECLARE(
  nl_286_scratch0, AI_STATIC,
  240, 0x0,
  AI_SHAPE_INIT(4, 1, 7, 1, 1), AI_STRIDE_INIT(4, 4, 4, 28, 28),
  1, &nl_286_scratch0_array, NULL)

/* Tensor #241 */
AI_TENSOR_OBJ_DECLARE(
  nl_311_output, AI_STATIC,
  241, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &nl_311_output_array, &nl_311_output_array_intq)

/* Tensor #242 */
AI_TENSOR_OBJ_DECLARE(
  nl_344_output, AI_STATIC,
  242, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &nl_344_output_array, &nl_344_output_array_intq)

/* Tensor #243 */
AI_TENSOR_OBJ_DECLARE(
  nl_50_output, AI_STATIC,
  243, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &nl_50_output_array, &nl_50_output_array_intq)

/* Tensor #244 */
AI_TENSOR_OBJ_DECLARE(
  nl_83_output, AI_STATIC,
  244, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &nl_83_output_array, &nl_83_output_array_intq)

/* Tensor #245 */
AI_TENSOR_OBJ_DECLARE(
  reduce_133_Mul_0_1_eltwise_134_conversion_0_1_eltwise_134_conversion_output, AI_STATIC,
  245, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &reduce_133_Mul_0_1_eltwise_134_conversion_0_1_eltwise_134_conversion_output_array, NULL)

/* Tensor #246 */
AI_TENSOR_OBJ_DECLARE(
  reduce_133_Mul_0_1_eltwise_134_conversion_output, AI_STATIC,
  246, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &reduce_133_Mul_0_1_eltwise_134_conversion_output_array, &reduce_133_Mul_0_1_eltwise_134_conversion_output_array_intq)

/* Tensor #247 */
AI_TENSOR_OBJ_DECLARE(
  reduce_133_Mul_output, AI_STATIC,
  247, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &reduce_133_Mul_output_array, NULL)

/* Tensor #248 */
AI_TENSOR_OBJ_DECLARE(
  reduce_133_output, AI_STATIC,
  248, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &reduce_133_output_array, NULL)

/* Tensor #249 */
AI_TENSOR_OBJ_DECLARE(
  reduce_135_Mul_0_0_eltwise_136_conversion_output, AI_STATIC,
  249, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &reduce_135_Mul_0_0_eltwise_136_conversion_output_array, &reduce_135_Mul_0_0_eltwise_136_conversion_output_array_intq)

/* Tensor #250 */
AI_TENSOR_OBJ_DECLARE(
  reduce_135_Mul_output, AI_STATIC,
  250, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &reduce_135_Mul_output_array, NULL)

/* Tensor #251 */
AI_TENSOR_OBJ_DECLARE(
  reduce_135_output, AI_STATIC,
  251, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &reduce_135_output_array, NULL)

/* Tensor #252 */
AI_TENSOR_OBJ_DECLARE(
  reduce_166_Mul_0_1_eltwise_167_conversion_0_1_eltwise_167_conversion_output, AI_STATIC,
  252, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &reduce_166_Mul_0_1_eltwise_167_conversion_0_1_eltwise_167_conversion_output_array, NULL)

/* Tensor #253 */
AI_TENSOR_OBJ_DECLARE(
  reduce_166_Mul_0_1_eltwise_167_conversion_output, AI_STATIC,
  253, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &reduce_166_Mul_0_1_eltwise_167_conversion_output_array, &reduce_166_Mul_0_1_eltwise_167_conversion_output_array_intq)

/* Tensor #254 */
AI_TENSOR_OBJ_DECLARE(
  reduce_166_Mul_output, AI_STATIC,
  254, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &reduce_166_Mul_output_array, NULL)

/* Tensor #255 */
AI_TENSOR_OBJ_DECLARE(
  reduce_166_output, AI_STATIC,
  255, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &reduce_166_output_array, NULL)

/* Tensor #256 */
AI_TENSOR_OBJ_DECLARE(
  reduce_168_Mul_0_0_eltwise_169_conversion_output, AI_STATIC,
  256, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &reduce_168_Mul_0_0_eltwise_169_conversion_output_array, &reduce_168_Mul_0_0_eltwise_169_conversion_output_array_intq)

/* Tensor #257 */
AI_TENSOR_OBJ_DECLARE(
  reduce_168_Mul_output, AI_STATIC,
  257, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &reduce_168_Mul_output_array, NULL)

/* Tensor #258 */
AI_TENSOR_OBJ_DECLARE(
  reduce_168_output, AI_STATIC,
  258, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &reduce_168_output_array, NULL)

/* Tensor #259 */
AI_TENSOR_OBJ_DECLARE(
  reduce_220_Mul_0_1_eltwise_221_conversion_0_1_eltwise_221_conversion_output, AI_STATIC,
  259, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &reduce_220_Mul_0_1_eltwise_221_conversion_0_1_eltwise_221_conversion_output_array, NULL)

/* Tensor #260 */
AI_TENSOR_OBJ_DECLARE(
  reduce_220_Mul_0_1_eltwise_221_conversion_output, AI_STATIC,
  260, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &reduce_220_Mul_0_1_eltwise_221_conversion_output_array, &reduce_220_Mul_0_1_eltwise_221_conversion_output_array_intq)

/* Tensor #261 */
AI_TENSOR_OBJ_DECLARE(
  reduce_220_Mul_output, AI_STATIC,
  261, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &reduce_220_Mul_output_array, NULL)

/* Tensor #262 */
AI_TENSOR_OBJ_DECLARE(
  reduce_220_output, AI_STATIC,
  262, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &reduce_220_output_array, NULL)

/* Tensor #263 */
AI_TENSOR_OBJ_DECLARE(
  reduce_222_Mul_0_0_eltwise_223_conversion_output, AI_STATIC,
  263, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &reduce_222_Mul_0_0_eltwise_223_conversion_output_array, &reduce_222_Mul_0_0_eltwise_223_conversion_output_array_intq)

/* Tensor #264 */
AI_TENSOR_OBJ_DECLARE(
  reduce_222_Mul_output, AI_STATIC,
  264, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &reduce_222_Mul_output_array, NULL)

/* Tensor #265 */
AI_TENSOR_OBJ_DECLARE(
  reduce_222_output, AI_STATIC,
  265, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &reduce_222_output_array, NULL)

/* Tensor #266 */
AI_TENSOR_OBJ_DECLARE(
  reduce_253_Mul_0_1_eltwise_254_conversion_0_1_eltwise_254_conversion_output, AI_STATIC,
  266, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &reduce_253_Mul_0_1_eltwise_254_conversion_0_1_eltwise_254_conversion_output_array, NULL)

/* Tensor #267 */
AI_TENSOR_OBJ_DECLARE(
  reduce_253_Mul_0_1_eltwise_254_conversion_output, AI_STATIC,
  267, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &reduce_253_Mul_0_1_eltwise_254_conversion_output_array, &reduce_253_Mul_0_1_eltwise_254_conversion_output_array_intq)

/* Tensor #268 */
AI_TENSOR_OBJ_DECLARE(
  reduce_253_Mul_output, AI_STATIC,
  268, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &reduce_253_Mul_output_array, NULL)

/* Tensor #269 */
AI_TENSOR_OBJ_DECLARE(
  reduce_253_output, AI_STATIC,
  269, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &reduce_253_output_array, NULL)

/* Tensor #270 */
AI_TENSOR_OBJ_DECLARE(
  reduce_255_Mul_0_0_eltwise_256_conversion_output, AI_STATIC,
  270, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &reduce_255_Mul_0_0_eltwise_256_conversion_output_array, &reduce_255_Mul_0_0_eltwise_256_conversion_output_array_intq)

/* Tensor #271 */
AI_TENSOR_OBJ_DECLARE(
  reduce_255_Mul_output, AI_STATIC,
  271, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &reduce_255_Mul_output_array, NULL)

/* Tensor #272 */
AI_TENSOR_OBJ_DECLARE(
  reduce_255_output, AI_STATIC,
  272, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &reduce_255_output_array, NULL)

/* Tensor #273 */
AI_TENSOR_OBJ_DECLARE(
  reduce_307_Mul_0_1_eltwise_308_conversion_0_1_eltwise_308_conversion_output, AI_STATIC,
  273, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &reduce_307_Mul_0_1_eltwise_308_conversion_0_1_eltwise_308_conversion_output_array, NULL)

/* Tensor #274 */
AI_TENSOR_OBJ_DECLARE(
  reduce_307_Mul_0_1_eltwise_308_conversion_output, AI_STATIC,
  274, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &reduce_307_Mul_0_1_eltwise_308_conversion_output_array, &reduce_307_Mul_0_1_eltwise_308_conversion_output_array_intq)

/* Tensor #275 */
AI_TENSOR_OBJ_DECLARE(
  reduce_307_Mul_output, AI_STATIC,
  275, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &reduce_307_Mul_output_array, NULL)

/* Tensor #276 */
AI_TENSOR_OBJ_DECLARE(
  reduce_307_output, AI_STATIC,
  276, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &reduce_307_output_array, NULL)

/* Tensor #277 */
AI_TENSOR_OBJ_DECLARE(
  reduce_309_Mul_0_0_eltwise_310_conversion_output, AI_STATIC,
  277, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &reduce_309_Mul_0_0_eltwise_310_conversion_output_array, &reduce_309_Mul_0_0_eltwise_310_conversion_output_array_intq)

/* Tensor #278 */
AI_TENSOR_OBJ_DECLARE(
  reduce_309_Mul_output, AI_STATIC,
  278, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &reduce_309_Mul_output_array, NULL)

/* Tensor #279 */
AI_TENSOR_OBJ_DECLARE(
  reduce_309_output, AI_STATIC,
  279, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &reduce_309_output_array, NULL)

/* Tensor #280 */
AI_TENSOR_OBJ_DECLARE(
  reduce_340_Mul_0_1_eltwise_341_conversion_0_1_eltwise_341_conversion_output, AI_STATIC,
  280, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &reduce_340_Mul_0_1_eltwise_341_conversion_0_1_eltwise_341_conversion_output_array, NULL)

/* Tensor #281 */
AI_TENSOR_OBJ_DECLARE(
  reduce_340_Mul_0_1_eltwise_341_conversion_output, AI_STATIC,
  281, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &reduce_340_Mul_0_1_eltwise_341_conversion_output_array, &reduce_340_Mul_0_1_eltwise_341_conversion_output_array_intq)

/* Tensor #282 */
AI_TENSOR_OBJ_DECLARE(
  reduce_340_Mul_output, AI_STATIC,
  282, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &reduce_340_Mul_output_array, NULL)

/* Tensor #283 */
AI_TENSOR_OBJ_DECLARE(
  reduce_340_output, AI_STATIC,
  283, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &reduce_340_output_array, NULL)

/* Tensor #284 */
AI_TENSOR_OBJ_DECLARE(
  reduce_342_Mul_0_0_eltwise_343_conversion_output, AI_STATIC,
  284, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &reduce_342_Mul_0_0_eltwise_343_conversion_output_array, &reduce_342_Mul_0_0_eltwise_343_conversion_output_array_intq)

/* Tensor #285 */
AI_TENSOR_OBJ_DECLARE(
  reduce_342_Mul_output, AI_STATIC,
  285, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &reduce_342_Mul_output_array, NULL)

/* Tensor #286 */
AI_TENSOR_OBJ_DECLARE(
  reduce_342_output, AI_STATIC,
  286, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &reduce_342_output_array, NULL)

/* Tensor #287 */
AI_TENSOR_OBJ_DECLARE(
  reduce_46_Mul_0_1_eltwise_47_conversion_0_1_eltwise_47_conversion_output, AI_STATIC,
  287, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &reduce_46_Mul_0_1_eltwise_47_conversion_0_1_eltwise_47_conversion_output_array, NULL)

/* Tensor #288 */
AI_TENSOR_OBJ_DECLARE(
  reduce_46_Mul_0_1_eltwise_47_conversion_output, AI_STATIC,
  288, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &reduce_46_Mul_0_1_eltwise_47_conversion_output_array, &reduce_46_Mul_0_1_eltwise_47_conversion_output_array_intq)

/* Tensor #289 */
AI_TENSOR_OBJ_DECLARE(
  reduce_46_Mul_bias, AI_STATIC,
  289, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &reduce_46_Mul_bias_array, NULL)

/* Tensor #290 */
AI_TENSOR_OBJ_DECLARE(
  reduce_46_Mul_output, AI_STATIC,
  290, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &reduce_46_Mul_output_array, NULL)

/* Tensor #291 */
AI_TENSOR_OBJ_DECLARE(
  reduce_46_Mul_scale, AI_STATIC,
  291, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &reduce_46_Mul_scale_array, NULL)

/* Tensor #292 */
AI_TENSOR_OBJ_DECLARE(
  reduce_46_output, AI_STATIC,
  292, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &reduce_46_output_array, NULL)

/* Tensor #293 */
AI_TENSOR_OBJ_DECLARE(
  reduce_48_Mul_0_0_eltwise_49_conversion_output, AI_STATIC,
  293, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &reduce_48_Mul_0_0_eltwise_49_conversion_output_array, &reduce_48_Mul_0_0_eltwise_49_conversion_output_array_intq)

/* Tensor #294 */
AI_TENSOR_OBJ_DECLARE(
  reduce_48_Mul_output, AI_STATIC,
  294, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &reduce_48_Mul_output_array, NULL)

/* Tensor #295 */
AI_TENSOR_OBJ_DECLARE(
  reduce_48_output, AI_STATIC,
  295, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &reduce_48_output_array, NULL)

/* Tensor #296 */
AI_TENSOR_OBJ_DECLARE(
  reduce_79_Mul_0_1_eltwise_80_conversion_0_1_eltwise_80_conversion_output, AI_STATIC,
  296, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &reduce_79_Mul_0_1_eltwise_80_conversion_0_1_eltwise_80_conversion_output_array, NULL)

/* Tensor #297 */
AI_TENSOR_OBJ_DECLARE(
  reduce_79_Mul_0_1_eltwise_80_conversion_output, AI_STATIC,
  297, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &reduce_79_Mul_0_1_eltwise_80_conversion_output_array, &reduce_79_Mul_0_1_eltwise_80_conversion_output_array_intq)

/* Tensor #298 */
AI_TENSOR_OBJ_DECLARE(
  reduce_79_Mul_output, AI_STATIC,
  298, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &reduce_79_Mul_output_array, NULL)

/* Tensor #299 */
AI_TENSOR_OBJ_DECLARE(
  reduce_79_output, AI_STATIC,
  299, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &reduce_79_output_array, NULL)

/* Tensor #300 */
AI_TENSOR_OBJ_DECLARE(
  reduce_81_Mul_0_0_eltwise_82_conversion_output, AI_STATIC,
  300, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &reduce_81_Mul_0_0_eltwise_82_conversion_output_array, &reduce_81_Mul_0_0_eltwise_82_conversion_output_array_intq)

/* Tensor #301 */
AI_TENSOR_OBJ_DECLARE(
  reduce_81_Mul_output, AI_STATIC,
  301, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &reduce_81_Mul_output_array, NULL)

/* Tensor #302 */
AI_TENSOR_OBJ_DECLARE(
  reduce_81_output, AI_STATIC,
  302, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &reduce_81_output_array, NULL)

/* Tensor #303 */
AI_TENSOR_OBJ_DECLARE(
  serving_default_embedded_tokens0_output, AI_STATIC,
  303, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &serving_default_embedded_tokens0_output_array, &serving_default_embedded_tokens0_output_array_intq)

/* Tensor #304 */
AI_TENSOR_OBJ_DECLARE(
  tiny_bert_generator_enc_0_ffn_dense_24_BiasAdd_ReadVariableOp_3D, AI_STATIC,
  304, 0x1,
  AI_SHAPE_INIT(4, 1, 512, 1, 1), AI_STRIDE_INIT(4, 1, 1, 512, 512),
  1, &tiny_bert_generator_enc_0_ffn_dense_24_BiasAdd_ReadVariableOp_3D_array, &tiny_bert_generator_enc_0_ffn_dense_24_BiasAdd_ReadVariableOp_3D_array_intq)

/* Tensor #305 */
AI_TENSOR_OBJ_DECLARE(
  tiny_bert_generator_enc_0_ffn_dense_25_BiasAdd_ReadVariableOp_3D, AI_STATIC,
  305, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &tiny_bert_generator_enc_0_ffn_dense_25_BiasAdd_ReadVariableOp_3D_array, &tiny_bert_generator_enc_0_ffn_dense_25_BiasAdd_ReadVariableOp_3D_array_intq)

/* Tensor #306 */
AI_TENSOR_OBJ_DECLARE(
  tiny_bert_generator_enc_0_ln1_batchnorm_ReadVariableOp_3D, AI_STATIC,
  306, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &tiny_bert_generator_enc_0_ln1_batchnorm_ReadVariableOp_3D_array, &tiny_bert_generator_enc_0_ln1_batchnorm_ReadVariableOp_3D_array_intq)

/* Tensor #307 */
AI_TENSOR_OBJ_DECLARE(
  tiny_bert_generator_enc_0_ln1_batchnorm_add_y_3D, AI_STATIC,
  307, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &tiny_bert_generator_enc_0_ln1_batchnorm_add_y_3D_array, &tiny_bert_generator_enc_0_ln1_batchnorm_add_y_3D_array_intq)

/* Tensor #308 */
AI_TENSOR_OBJ_DECLARE(
  tiny_bert_generator_enc_0_ln1_batchnorm_mul_ReadVariableOp_3D, AI_STATIC,
  308, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &tiny_bert_generator_enc_0_ln1_batchnorm_mul_ReadVariableOp_3D_array, &tiny_bert_generator_enc_0_ln1_batchnorm_mul_ReadVariableOp_3D_array_intq)

/* Tensor #309 */
AI_TENSOR_OBJ_DECLARE(
  tiny_bert_generator_enc_0_ln2_batchnorm_ReadVariableOp_3D, AI_STATIC,
  309, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &tiny_bert_generator_enc_0_ln2_batchnorm_ReadVariableOp_3D_array, &tiny_bert_generator_enc_0_ln2_batchnorm_ReadVariableOp_3D_array_intq)

/* Tensor #310 */
AI_TENSOR_OBJ_DECLARE(
  tiny_bert_generator_enc_0_ln2_batchnorm_mul_ReadVariableOp_3D, AI_STATIC,
  310, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &tiny_bert_generator_enc_0_ln2_batchnorm_mul_ReadVariableOp_3D_array, &tiny_bert_generator_enc_0_ln2_batchnorm_mul_ReadVariableOp_3D_array_intq)

/* Tensor #311 */
AI_TENSOR_OBJ_DECLARE(
  tiny_bert_generator_enc_0_mha_output_dense_BiasAdd_ReadVariableOp_3D, AI_STATIC,
  311, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &tiny_bert_generator_enc_0_mha_output_dense_BiasAdd_ReadVariableOp_3D_array, &tiny_bert_generator_enc_0_mha_output_dense_BiasAdd_ReadVariableOp_3D_array_intq)

/* Tensor #312 */
AI_TENSOR_OBJ_DECLARE(
  tiny_bert_generator_enc_0_mha_wk_BiasAdd_ReadVariableOp_3D, AI_STATIC,
  312, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &tiny_bert_generator_enc_0_mha_wk_BiasAdd_ReadVariableOp_3D_array, &tiny_bert_generator_enc_0_mha_wk_BiasAdd_ReadVariableOp_3D_array_intq)

/* Tensor #313 */
AI_TENSOR_OBJ_DECLARE(
  tiny_bert_generator_enc_0_mha_wq_BiasAdd_ReadVariableOp_3D, AI_STATIC,
  313, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &tiny_bert_generator_enc_0_mha_wq_BiasAdd_ReadVariableOp_3D_array, &tiny_bert_generator_enc_0_mha_wq_BiasAdd_ReadVariableOp_3D_array_intq)

/* Tensor #314 */
AI_TENSOR_OBJ_DECLARE(
  tiny_bert_generator_enc_0_mha_wv_BiasAdd_ReadVariableOp_3D, AI_STATIC,
  314, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &tiny_bert_generator_enc_0_mha_wv_BiasAdd_ReadVariableOp_3D_array, &tiny_bert_generator_enc_0_mha_wv_BiasAdd_ReadVariableOp_3D_array_intq)

/* Tensor #315 */
AI_TENSOR_OBJ_DECLARE(
  tiny_bert_generator_enc_1_ffn_dense_26_BiasAdd_ReadVariableOp_3D, AI_STATIC,
  315, 0x1,
  AI_SHAPE_INIT(4, 1, 512, 1, 1), AI_STRIDE_INIT(4, 1, 1, 512, 512),
  1, &tiny_bert_generator_enc_1_ffn_dense_26_BiasAdd_ReadVariableOp_3D_array, &tiny_bert_generator_enc_1_ffn_dense_26_BiasAdd_ReadVariableOp_3D_array_intq)

/* Tensor #316 */
AI_TENSOR_OBJ_DECLARE(
  tiny_bert_generator_enc_1_ffn_dense_27_BiasAdd_ReadVariableOp_3D, AI_STATIC,
  316, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &tiny_bert_generator_enc_1_ffn_dense_27_BiasAdd_ReadVariableOp_3D_array, &tiny_bert_generator_enc_1_ffn_dense_27_BiasAdd_ReadVariableOp_3D_array_intq)

/* Tensor #317 */
AI_TENSOR_OBJ_DECLARE(
  tiny_bert_generator_enc_1_ln1_batchnorm_ReadVariableOp_3D, AI_STATIC,
  317, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &tiny_bert_generator_enc_1_ln1_batchnorm_ReadVariableOp_3D_array, &tiny_bert_generator_enc_1_ln1_batchnorm_ReadVariableOp_3D_array_intq)

/* Tensor #318 */
AI_TENSOR_OBJ_DECLARE(
  tiny_bert_generator_enc_1_ln1_batchnorm_mul_ReadVariableOp_3D, AI_STATIC,
  318, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &tiny_bert_generator_enc_1_ln1_batchnorm_mul_ReadVariableOp_3D_array, &tiny_bert_generator_enc_1_ln1_batchnorm_mul_ReadVariableOp_3D_array_intq)

/* Tensor #319 */
AI_TENSOR_OBJ_DECLARE(
  tiny_bert_generator_enc_1_ln2_batchnorm_ReadVariableOp_3D, AI_STATIC,
  319, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &tiny_bert_generator_enc_1_ln2_batchnorm_ReadVariableOp_3D_array, &tiny_bert_generator_enc_1_ln2_batchnorm_ReadVariableOp_3D_array_intq)

/* Tensor #320 */
AI_TENSOR_OBJ_DECLARE(
  tiny_bert_generator_enc_1_ln2_batchnorm_mul_ReadVariableOp_3D, AI_STATIC,
  320, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &tiny_bert_generator_enc_1_ln2_batchnorm_mul_ReadVariableOp_3D_array, &tiny_bert_generator_enc_1_ln2_batchnorm_mul_ReadVariableOp_3D_array_intq)

/* Tensor #321 */
AI_TENSOR_OBJ_DECLARE(
  tiny_bert_generator_enc_1_mha_output_dense_BiasAdd_ReadVariableOp_3D, AI_STATIC,
  321, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &tiny_bert_generator_enc_1_mha_output_dense_BiasAdd_ReadVariableOp_3D_array, &tiny_bert_generator_enc_1_mha_output_dense_BiasAdd_ReadVariableOp_3D_array_intq)

/* Tensor #322 */
AI_TENSOR_OBJ_DECLARE(
  tiny_bert_generator_enc_1_mha_wk_BiasAdd_ReadVariableOp_3D, AI_STATIC,
  322, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &tiny_bert_generator_enc_1_mha_wk_BiasAdd_ReadVariableOp_3D_array, &tiny_bert_generator_enc_1_mha_wk_BiasAdd_ReadVariableOp_3D_array_intq)

/* Tensor #323 */
AI_TENSOR_OBJ_DECLARE(
  tiny_bert_generator_enc_1_mha_wq_BiasAdd_ReadVariableOp_3D, AI_STATIC,
  323, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &tiny_bert_generator_enc_1_mha_wq_BiasAdd_ReadVariableOp_3D_array, &tiny_bert_generator_enc_1_mha_wq_BiasAdd_ReadVariableOp_3D_array_intq)

/* Tensor #324 */
AI_TENSOR_OBJ_DECLARE(
  tiny_bert_generator_enc_1_mha_wv_BiasAdd_ReadVariableOp_3D, AI_STATIC,
  324, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &tiny_bert_generator_enc_1_mha_wv_BiasAdd_ReadVariableOp_3D_array, &tiny_bert_generator_enc_1_mha_wv_BiasAdd_ReadVariableOp_3D_array_intq)

/* Tensor #325 */
AI_TENSOR_OBJ_DECLARE(
  tiny_bert_generator_enc_2_ffn_dense_28_BiasAdd_ReadVariableOp_3D, AI_STATIC,
  325, 0x1,
  AI_SHAPE_INIT(4, 1, 512, 1, 1), AI_STRIDE_INIT(4, 1, 1, 512, 512),
  1, &tiny_bert_generator_enc_2_ffn_dense_28_BiasAdd_ReadVariableOp_3D_array, &tiny_bert_generator_enc_2_ffn_dense_28_BiasAdd_ReadVariableOp_3D_array_intq)

/* Tensor #326 */
AI_TENSOR_OBJ_DECLARE(
  tiny_bert_generator_enc_2_ffn_dense_29_BiasAdd_ReadVariableOp_3D, AI_STATIC,
  326, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &tiny_bert_generator_enc_2_ffn_dense_29_BiasAdd_ReadVariableOp_3D_array, &tiny_bert_generator_enc_2_ffn_dense_29_BiasAdd_ReadVariableOp_3D_array_intq)

/* Tensor #327 */
AI_TENSOR_OBJ_DECLARE(
  tiny_bert_generator_enc_2_ln1_batchnorm_ReadVariableOp_3D, AI_STATIC,
  327, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &tiny_bert_generator_enc_2_ln1_batchnorm_ReadVariableOp_3D_array, &tiny_bert_generator_enc_2_ln1_batchnorm_ReadVariableOp_3D_array_intq)

/* Tensor #328 */
AI_TENSOR_OBJ_DECLARE(
  tiny_bert_generator_enc_2_ln1_batchnorm_mul_ReadVariableOp_3D, AI_STATIC,
  328, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &tiny_bert_generator_enc_2_ln1_batchnorm_mul_ReadVariableOp_3D_array, &tiny_bert_generator_enc_2_ln1_batchnorm_mul_ReadVariableOp_3D_array_intq)

/* Tensor #329 */
AI_TENSOR_OBJ_DECLARE(
  tiny_bert_generator_enc_2_ln2_batchnorm_ReadVariableOp_3D, AI_STATIC,
  329, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &tiny_bert_generator_enc_2_ln2_batchnorm_ReadVariableOp_3D_array, &tiny_bert_generator_enc_2_ln2_batchnorm_ReadVariableOp_3D_array_intq)

/* Tensor #330 */
AI_TENSOR_OBJ_DECLARE(
  tiny_bert_generator_enc_2_ln2_batchnorm_mul_ReadVariableOp_3D, AI_STATIC,
  330, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &tiny_bert_generator_enc_2_ln2_batchnorm_mul_ReadVariableOp_3D_array, &tiny_bert_generator_enc_2_ln2_batchnorm_mul_ReadVariableOp_3D_array_intq)

/* Tensor #331 */
AI_TENSOR_OBJ_DECLARE(
  tiny_bert_generator_enc_2_mha_output_dense_BiasAdd_ReadVariableOp_3D, AI_STATIC,
  331, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &tiny_bert_generator_enc_2_mha_output_dense_BiasAdd_ReadVariableOp_3D_array, &tiny_bert_generator_enc_2_mha_output_dense_BiasAdd_ReadVariableOp_3D_array_intq)

/* Tensor #332 */
AI_TENSOR_OBJ_DECLARE(
  tiny_bert_generator_enc_2_mha_wk_BiasAdd_ReadVariableOp_3D, AI_STATIC,
  332, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &tiny_bert_generator_enc_2_mha_wk_BiasAdd_ReadVariableOp_3D_array, &tiny_bert_generator_enc_2_mha_wk_BiasAdd_ReadVariableOp_3D_array_intq)

/* Tensor #333 */
AI_TENSOR_OBJ_DECLARE(
  tiny_bert_generator_enc_2_mha_wq_BiasAdd_ReadVariableOp_3D, AI_STATIC,
  333, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &tiny_bert_generator_enc_2_mha_wq_BiasAdd_ReadVariableOp_3D_array, &tiny_bert_generator_enc_2_mha_wq_BiasAdd_ReadVariableOp_3D_array_intq)

/* Tensor #334 */
AI_TENSOR_OBJ_DECLARE(
  tiny_bert_generator_enc_2_mha_wv_BiasAdd_ReadVariableOp_3D, AI_STATIC,
  334, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &tiny_bert_generator_enc_2_mha_wv_BiasAdd_ReadVariableOp_3D_array, &tiny_bert_generator_enc_2_mha_wv_BiasAdd_ReadVariableOp_3D_array_intq)

/* Tensor #335 */
AI_TENSOR_OBJ_DECLARE(
  tiny_bert_generator_enc_3_ffn_dense_30_BiasAdd_ReadVariableOp_3D, AI_STATIC,
  335, 0x1,
  AI_SHAPE_INIT(4, 1, 512, 1, 1), AI_STRIDE_INIT(4, 1, 1, 512, 512),
  1, &tiny_bert_generator_enc_3_ffn_dense_30_BiasAdd_ReadVariableOp_3D_array, &tiny_bert_generator_enc_3_ffn_dense_30_BiasAdd_ReadVariableOp_3D_array_intq)

/* Tensor #336 */
AI_TENSOR_OBJ_DECLARE(
  tiny_bert_generator_enc_3_ffn_dense_31_BiasAdd_ReadVariableOp_3D, AI_STATIC,
  336, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &tiny_bert_generator_enc_3_ffn_dense_31_BiasAdd_ReadVariableOp_3D_array, &tiny_bert_generator_enc_3_ffn_dense_31_BiasAdd_ReadVariableOp_3D_array_intq)

/* Tensor #337 */
AI_TENSOR_OBJ_DECLARE(
  tiny_bert_generator_enc_3_ln1_batchnorm_ReadVariableOp_3D, AI_STATIC,
  337, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &tiny_bert_generator_enc_3_ln1_batchnorm_ReadVariableOp_3D_array, &tiny_bert_generator_enc_3_ln1_batchnorm_ReadVariableOp_3D_array_intq)

/* Tensor #338 */
AI_TENSOR_OBJ_DECLARE(
  tiny_bert_generator_enc_3_ln1_batchnorm_mul_ReadVariableOp_3D, AI_STATIC,
  338, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &tiny_bert_generator_enc_3_ln1_batchnorm_mul_ReadVariableOp_3D_array, &tiny_bert_generator_enc_3_ln1_batchnorm_mul_ReadVariableOp_3D_array_intq)

/* Tensor #339 */
AI_TENSOR_OBJ_DECLARE(
  tiny_bert_generator_enc_3_ln2_batchnorm_ReadVariableOp_3D, AI_STATIC,
  339, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &tiny_bert_generator_enc_3_ln2_batchnorm_ReadVariableOp_3D_array, &tiny_bert_generator_enc_3_ln2_batchnorm_ReadVariableOp_3D_array_intq)

/* Tensor #340 */
AI_TENSOR_OBJ_DECLARE(
  tiny_bert_generator_enc_3_ln2_batchnorm_mul_ReadVariableOp_3D, AI_STATIC,
  340, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &tiny_bert_generator_enc_3_ln2_batchnorm_mul_ReadVariableOp_3D_array, &tiny_bert_generator_enc_3_ln2_batchnorm_mul_ReadVariableOp_3D_array_intq)

/* Tensor #341 */
AI_TENSOR_OBJ_DECLARE(
  tiny_bert_generator_enc_3_mha_output_dense_BiasAdd_ReadVariableOp_3D, AI_STATIC,
  341, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &tiny_bert_generator_enc_3_mha_output_dense_BiasAdd_ReadVariableOp_3D_array, &tiny_bert_generator_enc_3_mha_output_dense_BiasAdd_ReadVariableOp_3D_array_intq)

/* Tensor #342 */
AI_TENSOR_OBJ_DECLARE(
  tiny_bert_generator_enc_3_mha_truedivtiny_bert_generator_enc_3_mha_Sqrt_4D, AI_STATIC,
  342, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &tiny_bert_generator_enc_3_mha_truedivtiny_bert_generator_enc_3_mha_Sqrt_4D_array, &tiny_bert_generator_enc_3_mha_truedivtiny_bert_generator_enc_3_mha_Sqrt_4D_array_intq)

/* Tensor #343 */
AI_TENSOR_OBJ_DECLARE(
  tiny_bert_generator_enc_3_mha_wk_BiasAdd_ReadVariableOp_3D, AI_STATIC,
  343, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &tiny_bert_generator_enc_3_mha_wk_BiasAdd_ReadVariableOp_3D_array, &tiny_bert_generator_enc_3_mha_wk_BiasAdd_ReadVariableOp_3D_array_intq)

/* Tensor #344 */
AI_TENSOR_OBJ_DECLARE(
  tiny_bert_generator_enc_3_mha_wq_BiasAdd_ReadVariableOp_3D, AI_STATIC,
  344, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &tiny_bert_generator_enc_3_mha_wq_BiasAdd_ReadVariableOp_3D_array, &tiny_bert_generator_enc_3_mha_wq_BiasAdd_ReadVariableOp_3D_array_intq)

/* Tensor #345 */
AI_TENSOR_OBJ_DECLARE(
  tiny_bert_generator_enc_3_mha_wv_BiasAdd_ReadVariableOp_3D, AI_STATIC,
  345, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &tiny_bert_generator_enc_3_mha_wv_BiasAdd_ReadVariableOp_3D_array, &tiny_bert_generator_enc_3_mha_wv_BiasAdd_ReadVariableOp_3D_array_intq)

/* Tensor #346 */
AI_TENSOR_OBJ_DECLARE(
  tiny_bert_generator_output_dense_BiasAdd_ReadVariableOp_3D, AI_STATIC,
  346, 0x1,
  AI_SHAPE_INIT(4, 1, 66, 1, 1), AI_STRIDE_INIT(4, 1, 1, 66, 66),
  1, &tiny_bert_generator_output_dense_BiasAdd_ReadVariableOp_3D_array, &tiny_bert_generator_output_dense_BiasAdd_ReadVariableOp_3D_array_intq)

/* Tensor #347 */
AI_TENSOR_OBJ_DECLARE(
  tiny_bert_generator_tf___operators___add_1_y, AI_STATIC,
  347, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &tiny_bert_generator_tf___operators___add_1_y_array, &tiny_bert_generator_tf___operators___add_1_y_array_intq)

/* Tensor #348 */
AI_TENSOR_OBJ_DECLARE(
  tiny_bert_generator_tf_math_multiply_1_Mul_y_3D, AI_STATIC,
  348, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &tiny_bert_generator_tf_math_multiply_1_Mul_y_3D_array, &tiny_bert_generator_tf_math_multiply_1_Mul_y_3D_array_intq)

/* Tensor #349 */
AI_TENSOR_OBJ_DECLARE(
  transpose_109_0_0_gemm_110_conversion_output, AI_STATIC,
  349, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 100, 4), AI_STRIDE_INIT(4, 4, 4, 256, 25600),
  1, &transpose_109_0_0_gemm_110_conversion_output_array, NULL)

/* Tensor #350 */
AI_TENSOR_OBJ_DECLARE(
  transpose_109_output, AI_STATIC,
  350, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 100, 4), AI_STRIDE_INIT(4, 1, 1, 64, 6400),
  1, &transpose_109_output_array, &transpose_109_output_array_intq)

/* Tensor #351 */
AI_TENSOR_OBJ_DECLARE(
  transpose_117_0_1_gemm_118_conversion_output, AI_STATIC,
  351, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 100, 4), AI_STRIDE_INIT(4, 4, 4, 256, 25600),
  1, &transpose_117_0_1_gemm_118_conversion_output_array, NULL)

/* Tensor #352 */
AI_TENSOR_OBJ_DECLARE(
  transpose_117_output, AI_STATIC,
  352, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 100, 4), AI_STRIDE_INIT(4, 1, 1, 64, 6400),
  1, &transpose_117_output_array, &transpose_117_output_array_intq)

/* Tensor #353 */
AI_TENSOR_OBJ_DECLARE(
  transpose_119_output, AI_STATIC,
  353, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 4, 100), AI_STRIDE_INIT(4, 1, 1, 64, 256),
  1, &transpose_119_output_array, &transpose_119_output_array_intq)

/* Tensor #354 */
AI_TENSOR_OBJ_DECLARE(
  transpose_119_output0, AI_STATIC,
  354, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &transpose_119_output_array, &transpose_119_output_array_intq)

/* Tensor #355 */
AI_TENSOR_OBJ_DECLARE(
  transpose_196_0_0_gemm_197_conversion_output, AI_STATIC,
  355, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 100, 4), AI_STRIDE_INIT(4, 4, 4, 256, 25600),
  1, &transpose_196_0_0_gemm_197_conversion_output_array, NULL)

/* Tensor #356 */
AI_TENSOR_OBJ_DECLARE(
  transpose_196_output, AI_STATIC,
  356, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 100, 4), AI_STRIDE_INIT(4, 1, 1, 64, 6400),
  1, &transpose_196_output_array, &transpose_196_output_array_intq)

/* Tensor #357 */
AI_TENSOR_OBJ_DECLARE(
  transpose_204_0_1_gemm_205_conversion_output, AI_STATIC,
  357, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 100, 4), AI_STRIDE_INIT(4, 4, 4, 256, 25600),
  1, &transpose_204_0_1_gemm_205_conversion_output_array, NULL)

/* Tensor #358 */
AI_TENSOR_OBJ_DECLARE(
  transpose_204_output, AI_STATIC,
  358, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 100, 4), AI_STRIDE_INIT(4, 1, 1, 64, 6400),
  1, &transpose_204_output_array, &transpose_204_output_array_intq)

/* Tensor #359 */
AI_TENSOR_OBJ_DECLARE(
  transpose_206_output, AI_STATIC,
  359, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 4, 100), AI_STRIDE_INIT(4, 1, 1, 64, 256),
  1, &transpose_206_output_array, &transpose_206_output_array_intq)

/* Tensor #360 */
AI_TENSOR_OBJ_DECLARE(
  transpose_206_output0, AI_STATIC,
  360, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &transpose_206_output_array, &transpose_206_output_array_intq)

/* Tensor #361 */
AI_TENSOR_OBJ_DECLARE(
  transpose_22_0_0_gemm_23_conversion_output, AI_STATIC,
  361, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 100, 4), AI_STRIDE_INIT(4, 4, 4, 256, 25600),
  1, &transpose_22_0_0_gemm_23_conversion_output_array, NULL)

/* Tensor #362 */
AI_TENSOR_OBJ_DECLARE(
  transpose_22_output, AI_STATIC,
  362, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 100, 4), AI_STRIDE_INIT(4, 1, 1, 64, 6400),
  1, &transpose_22_output_array, &transpose_22_output_array_intq)

/* Tensor #363 */
AI_TENSOR_OBJ_DECLARE(
  transpose_283_0_0_gemm_284_conversion_output, AI_STATIC,
  363, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 100, 4), AI_STRIDE_INIT(4, 4, 4, 256, 25600),
  1, &transpose_283_0_0_gemm_284_conversion_output_array, NULL)

/* Tensor #364 */
AI_TENSOR_OBJ_DECLARE(
  transpose_283_output, AI_STATIC,
  364, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 100, 4), AI_STRIDE_INIT(4, 1, 1, 64, 6400),
  1, &transpose_283_output_array, &transpose_283_output_array_intq)

/* Tensor #365 */
AI_TENSOR_OBJ_DECLARE(
  transpose_291_0_1_gemm_292_conversion_output, AI_STATIC,
  365, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 100, 4), AI_STRIDE_INIT(4, 4, 4, 256, 25600),
  1, &transpose_291_0_1_gemm_292_conversion_output_array, NULL)

/* Tensor #366 */
AI_TENSOR_OBJ_DECLARE(
  transpose_291_output, AI_STATIC,
  366, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 100, 4), AI_STRIDE_INIT(4, 1, 1, 64, 6400),
  1, &transpose_291_output_array, &transpose_291_output_array_intq)

/* Tensor #367 */
AI_TENSOR_OBJ_DECLARE(
  transpose_293_output, AI_STATIC,
  367, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 4, 100), AI_STRIDE_INIT(4, 1, 1, 64, 256),
  1, &transpose_293_output_array, &transpose_293_output_array_intq)

/* Tensor #368 */
AI_TENSOR_OBJ_DECLARE(
  transpose_293_output0, AI_STATIC,
  368, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &transpose_293_output_array, &transpose_293_output_array_intq)

/* Tensor #369 */
AI_TENSOR_OBJ_DECLARE(
  transpose_30_0_1_gemm_31_conversion_output, AI_STATIC,
  369, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 100, 4), AI_STRIDE_INIT(4, 4, 4, 256, 25600),
  1, &transpose_30_0_1_gemm_31_conversion_output_array, NULL)

/* Tensor #370 */
AI_TENSOR_OBJ_DECLARE(
  transpose_30_output, AI_STATIC,
  370, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 100, 4), AI_STRIDE_INIT(4, 1, 1, 64, 6400),
  1, &transpose_30_output_array, &transpose_30_output_array_intq)

/* Tensor #371 */
AI_TENSOR_OBJ_DECLARE(
  transpose_32_output, AI_STATIC,
  371, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 4, 100), AI_STRIDE_INIT(4, 1, 1, 64, 256),
  1, &transpose_32_output_array, &transpose_32_output_array_intq)

/* Tensor #372 */
AI_TENSOR_OBJ_DECLARE(
  transpose_32_output0, AI_STATIC,
  372, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &transpose_32_output_array, &transpose_32_output_array_intq)

/* Tensor #373 */
AI_TENSOR_OBJ_DECLARE(
  transpose_bgemm_110_out_0_1_gemm_110_conversion_output, AI_STATIC,
  373, 0x0,
  AI_SHAPE_INIT(4, 1, 100, 64, 4), AI_STRIDE_INIT(4, 4, 4, 400, 25600),
  1, &transpose_bgemm_110_out_0_1_gemm_110_conversion_output_array, NULL)

/* Tensor #374 */
AI_TENSOR_OBJ_DECLARE(
  transpose_bgemm_110_out_output, AI_STATIC,
  374, 0x1,
  AI_SHAPE_INIT(4, 1, 100, 64, 4), AI_STRIDE_INIT(4, 1, 1, 100, 6400),
  1, &transpose_bgemm_110_out_output_array, &transpose_bgemm_110_out_output_array_intq)

/* Tensor #375 */
AI_TENSOR_OBJ_DECLARE(
  transpose_bgemm_197_out_0_1_gemm_197_conversion_output, AI_STATIC,
  375, 0x0,
  AI_SHAPE_INIT(4, 1, 100, 64, 4), AI_STRIDE_INIT(4, 4, 4, 400, 25600),
  1, &transpose_bgemm_197_out_0_1_gemm_197_conversion_output_array, NULL)

/* Tensor #376 */
AI_TENSOR_OBJ_DECLARE(
  transpose_bgemm_197_out_output, AI_STATIC,
  376, 0x1,
  AI_SHAPE_INIT(4, 1, 100, 64, 4), AI_STRIDE_INIT(4, 1, 1, 100, 6400),
  1, &transpose_bgemm_197_out_output_array, &transpose_bgemm_197_out_output_array_intq)

/* Tensor #377 */
AI_TENSOR_OBJ_DECLARE(
  transpose_bgemm_23_out_0_1_gemm_23_conversion_output, AI_STATIC,
  377, 0x0,
  AI_SHAPE_INIT(4, 1, 100, 64, 4), AI_STRIDE_INIT(4, 4, 4, 400, 25600),
  1, &transpose_bgemm_23_out_0_1_gemm_23_conversion_output_array, NULL)

/* Tensor #378 */
AI_TENSOR_OBJ_DECLARE(
  transpose_bgemm_23_out_output, AI_STATIC,
  378, 0x1,
  AI_SHAPE_INIT(4, 1, 100, 64, 4), AI_STRIDE_INIT(4, 1, 1, 100, 6400),
  1, &transpose_bgemm_23_out_output_array, &transpose_bgemm_23_out_output_array_intq)

/* Tensor #379 */
AI_TENSOR_OBJ_DECLARE(
  transpose_bgemm_284_out_0_1_gemm_284_conversion_output, AI_STATIC,
  379, 0x0,
  AI_SHAPE_INIT(4, 1, 100, 64, 4), AI_STRIDE_INIT(4, 4, 4, 400, 25600),
  1, &transpose_bgemm_284_out_0_1_gemm_284_conversion_output_array, NULL)

/* Tensor #380 */
AI_TENSOR_OBJ_DECLARE(
  transpose_bgemm_284_out_output, AI_STATIC,
  380, 0x1,
  AI_SHAPE_INIT(4, 1, 100, 64, 4), AI_STRIDE_INIT(4, 1, 1, 100, 6400),
  1, &transpose_bgemm_284_out_output_array, &transpose_bgemm_284_out_output_array_intq)



/**  Layer declarations section  **********************************************/


AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_360_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_358_output, &tiny_bert_generator_output_dense_BiasAdd_ReadVariableOp_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_360_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_360_layer, 360,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_360_chain,
  NULL, &eltwise_360_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_358_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_349_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_358_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &gemm_358_weights, &gemm_358_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_358_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_358_layer, 358,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &gemm_358_chain,
  NULL, &eltwise_360_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_349_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_346_output, &eltwise_348_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_349_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_349_layer, 349,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_349_chain,
  NULL, &gemm_358_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_346_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_339_output, &eltwise_345_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_346_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_346_layer, 346,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_346_chain,
  NULL, &eltwise_349_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_348_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &tiny_bert_generator_enc_3_ln2_batchnorm_ReadVariableOp_3D, &eltwise_347_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_348_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_348_layer, 348,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_348_chain,
  NULL, &eltwise_346_layer, AI_STATIC, 
  .operation = ai_sub_f32, 
  .buffer_operation = ai_sub_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_347_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &reduce_340_Mul_0_1_eltwise_341_conversion_output, &eltwise_345_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_347_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_347_layer, 347,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_347_chain,
  NULL, &eltwise_348_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_345_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &nl_344_output, &tiny_bert_generator_enc_3_ln2_batchnorm_mul_ReadVariableOp_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_345_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_345_layer, 345,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_345_chain,
  NULL, &eltwise_347_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)


AI_STATIC_CONST ai_i8 nl_344_nl_params_data[] = { -128, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_344_nl_params, AI_ARRAY_FORMAT_S8,
    nl_344_nl_params_data, nl_344_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_344_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_343_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_344_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_344_layer, 344,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_344_chain,
  NULL, &eltwise_345_layer, AI_STATIC, 
  .nl_params = &nl_344_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_343_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &reduce_342_Mul_0_0_eltwise_343_conversion_output, &tiny_bert_generator_enc_0_ln1_batchnorm_add_y_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_343_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_343_layer, 343,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_343_chain,
  NULL, &nl_344_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_342_Mul_0_0_eltwise_343_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_342_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_342_Mul_0_0_eltwise_343_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_342_Mul_0_0_eltwise_343_conversion_layer, 342,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &reduce_342_Mul_0_0_eltwise_343_conversion_chain,
  NULL, &eltwise_343_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_342_Mul_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_342_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_342_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &reduce_46_Mul_scale, &reduce_46_Mul_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_342_Mul_layer, 342,
  BN_TYPE, 0x0, NULL,
  bn, forward_bn,
  &reduce_342_Mul_chain,
  NULL, &reduce_342_Mul_0_0_eltwise_343_conversion_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_float reduce_342_neutral_value_data[] = { 0.0f };
AI_ARRAY_OBJ_DECLARE(
    reduce_342_neutral_value, AI_ARRAY_FORMAT_FLOAT,
    reduce_342_neutral_value_data, reduce_342_neutral_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_342_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_341_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_342_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_342_layer, 342,
  REDUCE_TYPE, 0x0, NULL,
  reduce, forward_reduce,
  &reduce_342_chain,
  NULL, &reduce_342_Mul_layer, AI_STATIC, 
  .operation = ai_sum, 
  .neutral_value = &reduce_342_neutral_value, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_341_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_339_0_0_reduce_340_conversion_output, &reduce_340_Mul_0_1_eltwise_341_conversion_0_1_eltwise_341_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_341_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_341_layer, 341,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_341_chain,
  NULL, &reduce_342_layer, AI_STATIC, 
  .operation = ai_squared_diff, 
  .buffer_operation = ai_squared_diff_buffer, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_340_Mul_0_1_eltwise_341_conversion_0_1_eltwise_341_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_340_Mul_0_1_eltwise_341_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_340_Mul_0_1_eltwise_341_conversion_0_1_eltwise_341_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_340_Mul_0_1_eltwise_341_conversion_0_1_eltwise_341_conversion_layer, 340,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &reduce_340_Mul_0_1_eltwise_341_conversion_0_1_eltwise_341_conversion_chain,
  NULL, &eltwise_341_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_340_Mul_0_1_eltwise_341_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_340_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_340_Mul_0_1_eltwise_341_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_340_Mul_0_1_eltwise_341_conversion_layer, 340,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &reduce_340_Mul_0_1_eltwise_341_conversion_chain,
  NULL, &reduce_340_Mul_0_1_eltwise_341_conversion_0_1_eltwise_341_conversion_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_340_Mul_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_340_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_340_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &reduce_46_Mul_scale, &reduce_46_Mul_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_340_Mul_layer, 340,
  BN_TYPE, 0x0, NULL,
  bn, forward_bn,
  &reduce_340_Mul_chain,
  NULL, &reduce_340_Mul_0_1_eltwise_341_conversion_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_float reduce_340_neutral_value_data[] = { 0.0f };
AI_ARRAY_OBJ_DECLARE(
    reduce_340_neutral_value, AI_ARRAY_FORMAT_FLOAT,
    reduce_340_neutral_value_data, reduce_340_neutral_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_340_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_339_0_0_reduce_340_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_340_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_340_layer, 340,
  REDUCE_TYPE, 0x0, NULL,
  reduce, forward_reduce,
  &reduce_340_chain,
  NULL, &reduce_340_Mul_layer, AI_STATIC, 
  .operation = ai_sum, 
  .neutral_value = &reduce_340_neutral_value, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_339_0_0_reduce_340_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_339_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_339_0_0_reduce_340_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_339_0_0_reduce_340_conversion_layer, 339,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &eltwise_339_0_0_reduce_340_conversion_chain,
  NULL, &reduce_340_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_339_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_316_output, &eltwise_338_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_339_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_339_layer, 339,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_339_chain,
  NULL, &eltwise_339_0_0_reduce_340_conversion_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_338_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_336_output, &tiny_bert_generator_enc_3_ffn_dense_31_BiasAdd_ReadVariableOp_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_338_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_338_layer, 338,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_338_chain,
  NULL, &eltwise_339_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_336_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_327_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_336_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &gemm_336_weights, &gemm_26_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_336_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_336_layer, 336,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &gemm_336_chain,
  NULL, &eltwise_338_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_327_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_325_output, &tiny_bert_generator_enc_3_ffn_dense_30_BiasAdd_ReadVariableOp_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_327_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_327_layer, 327,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_327_chain,
  NULL, &gemm_336_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_325_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_316_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_325_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &gemm_325_weights, &gemm_64_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_325_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_325_layer, 325,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &gemm_325_chain,
  NULL, &eltwise_327_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_316_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_313_output, &eltwise_315_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_316_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_316_layer, 316,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_316_chain,
  NULL, &gemm_325_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_313_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_306_output, &eltwise_312_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_313_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_313_layer, 313,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_313_chain,
  NULL, &eltwise_316_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_315_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &tiny_bert_generator_enc_3_ln1_batchnorm_ReadVariableOp_3D, &eltwise_314_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_315_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_315_layer, 315,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_315_chain,
  NULL, &eltwise_313_layer, AI_STATIC, 
  .operation = ai_sub_f32, 
  .buffer_operation = ai_sub_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_314_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &reduce_307_Mul_0_1_eltwise_308_conversion_output, &eltwise_312_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_314_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_314_layer, 314,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_314_chain,
  NULL, &eltwise_315_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_312_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &nl_311_output, &tiny_bert_generator_enc_3_ln1_batchnorm_mul_ReadVariableOp_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_312_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_312_layer, 312,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_312_chain,
  NULL, &eltwise_314_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)


AI_STATIC_CONST ai_i8 nl_311_nl_params_data[] = { -128, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_311_nl_params, AI_ARRAY_FORMAT_S8,
    nl_311_nl_params_data, nl_311_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_311_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_310_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_311_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_311_layer, 311,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_311_chain,
  NULL, &eltwise_312_layer, AI_STATIC, 
  .nl_params = &nl_311_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_310_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &reduce_309_Mul_0_0_eltwise_310_conversion_output, &tiny_bert_generator_enc_0_ln1_batchnorm_add_y_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_310_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_310_layer, 310,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_310_chain,
  NULL, &nl_311_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_309_Mul_0_0_eltwise_310_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_309_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_309_Mul_0_0_eltwise_310_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_309_Mul_0_0_eltwise_310_conversion_layer, 309,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &reduce_309_Mul_0_0_eltwise_310_conversion_chain,
  NULL, &eltwise_310_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_309_Mul_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_309_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_309_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &reduce_46_Mul_scale, &reduce_46_Mul_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_309_Mul_layer, 309,
  BN_TYPE, 0x0, NULL,
  bn, forward_bn,
  &reduce_309_Mul_chain,
  NULL, &reduce_309_Mul_0_0_eltwise_310_conversion_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_float reduce_309_neutral_value_data[] = { 0.0f };
AI_ARRAY_OBJ_DECLARE(
    reduce_309_neutral_value, AI_ARRAY_FORMAT_FLOAT,
    reduce_309_neutral_value_data, reduce_309_neutral_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_309_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_308_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_309_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_309_layer, 309,
  REDUCE_TYPE, 0x0, NULL,
  reduce, forward_reduce,
  &reduce_309_chain,
  NULL, &reduce_309_Mul_layer, AI_STATIC, 
  .operation = ai_sum, 
  .neutral_value = &reduce_309_neutral_value, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_308_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_306_0_0_reduce_307_conversion_output, &reduce_307_Mul_0_1_eltwise_308_conversion_0_1_eltwise_308_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_308_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_308_layer, 308,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_308_chain,
  NULL, &reduce_309_layer, AI_STATIC, 
  .operation = ai_squared_diff, 
  .buffer_operation = ai_squared_diff_buffer, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_307_Mul_0_1_eltwise_308_conversion_0_1_eltwise_308_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_307_Mul_0_1_eltwise_308_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_307_Mul_0_1_eltwise_308_conversion_0_1_eltwise_308_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_307_Mul_0_1_eltwise_308_conversion_0_1_eltwise_308_conversion_layer, 307,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &reduce_307_Mul_0_1_eltwise_308_conversion_0_1_eltwise_308_conversion_chain,
  NULL, &eltwise_308_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_307_Mul_0_1_eltwise_308_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_307_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_307_Mul_0_1_eltwise_308_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_307_Mul_0_1_eltwise_308_conversion_layer, 307,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &reduce_307_Mul_0_1_eltwise_308_conversion_chain,
  NULL, &reduce_307_Mul_0_1_eltwise_308_conversion_0_1_eltwise_308_conversion_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_307_Mul_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_307_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_307_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &reduce_46_Mul_scale, &reduce_46_Mul_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_307_Mul_layer, 307,
  BN_TYPE, 0x0, NULL,
  bn, forward_bn,
  &reduce_307_Mul_chain,
  NULL, &reduce_307_Mul_0_1_eltwise_308_conversion_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_float reduce_307_neutral_value_data[] = { 0.0f };
AI_ARRAY_OBJ_DECLARE(
    reduce_307_neutral_value, AI_ARRAY_FORMAT_FLOAT,
    reduce_307_neutral_value_data, reduce_307_neutral_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_307_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_306_0_0_reduce_307_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_307_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_307_layer, 307,
  REDUCE_TYPE, 0x0, NULL,
  reduce, forward_reduce,
  &reduce_307_chain,
  NULL, &reduce_307_Mul_layer, AI_STATIC, 
  .operation = ai_sum, 
  .neutral_value = &reduce_307_neutral_value, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_306_0_0_reduce_307_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_306_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_306_0_0_reduce_307_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_306_0_0_reduce_307_conversion_layer, 306,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &eltwise_306_0_0_reduce_307_conversion_chain,
  NULL, &reduce_307_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_306_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_262_output, &eltwise_305_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_306_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_306_layer, 306,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_306_chain,
  NULL, &eltwise_306_0_0_reduce_307_conversion_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_305_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_303_output, &tiny_bert_generator_enc_3_mha_output_dense_BiasAdd_ReadVariableOp_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_305_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_305_layer, 305,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_305_chain,
  NULL, &eltwise_306_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_303_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_293_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_303_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &gemm_303_weights, &gemm_26_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_303_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_303_layer, 303,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &gemm_303_chain,
  NULL, &eltwise_305_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  transpose_293_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_292_0_0_transpose_293_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_293_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  transpose_293_layer, 293,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &transpose_293_chain,
  NULL, &gemm_303_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_CHANNEL, AI_SHAPE_HEIGHT, AI_SHAPE_WIDTH, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_292_0_0_transpose_293_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_292_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_292_0_0_transpose_293_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_292_0_0_transpose_293_conversion_layer, 292,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &gemm_292_0_0_transpose_293_conversion_chain,
  NULL, &transpose_293_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_292_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &nl_286_0_0_gemm_292_conversion_output, &transpose_291_0_1_gemm_292_conversion_output, &gemm_292_bias_0_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_292_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_292_layer, 292,
  MATMUL_TYPE, 0x0, NULL,
  matmul, forward_matmul,
  &gemm_292_chain,
  NULL, &gemm_292_0_0_transpose_293_conversion_layer, AI_STATIC, 
  .alpha = 1.0, 
  .beta = 1.0, 
  .tA = 0, 
  .tB = 0, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_286_0_0_gemm_292_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_286_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_286_0_0_gemm_292_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_286_0_0_gemm_292_conversion_layer, 286,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &nl_286_0_0_gemm_292_conversion_chain,
  NULL, &gemm_292_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i32 nl_286_nl_params_data[] = { 1333409664, 28, -7 };
AI_ARRAY_OBJ_DECLARE(
    nl_286_nl_params, AI_ARRAY_FORMAT_S32,
    nl_286_nl_params_data, nl_286_nl_params_data, 3, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_286_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_285_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_286_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_286_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  nl_286_layer, 286,
  SM_TYPE, 0x0, NULL,
  sm, forward_sm_integer,
  &nl_286_chain,
  NULL, &nl_286_0_0_gemm_292_conversion_layer, AI_STATIC, 
  .nl_params = &nl_286_nl_params, 
  .axis = AI_SHAPE_CHANNEL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_285_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_284_0_0_eltwise_285_conversion_output, &tiny_bert_generator_enc_3_mha_truedivtiny_bert_generator_enc_3_mha_Sqrt_4D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_285_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_285_layer, 285,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_285_chain,
  NULL, &nl_286_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_284_0_0_eltwise_285_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_284_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_284_0_0_eltwise_285_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_284_0_0_eltwise_285_conversion_layer, 284,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &gemm_284_0_0_eltwise_285_conversion_chain,
  NULL, &eltwise_285_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_284_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &transpose_283_0_0_gemm_284_conversion_output, &transpose_bgemm_284_out_0_1_gemm_284_conversion_output, &gemm_284_bias_0_2_gemm_284_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_284_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_284_layer, 284,
  MATMUL_TYPE, 0x0, NULL,
  matmul, forward_matmul,
  &gemm_284_chain,
  NULL, &gemm_284_0_0_eltwise_285_conversion_layer, AI_STATIC, 
  .alpha = 1.0, 
  .beta = 1.0, 
  .tA = 0, 
  .tB = 0, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  transpose_bgemm_284_out_0_1_gemm_284_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_bgemm_284_out_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_bgemm_284_out_0_1_gemm_284_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  transpose_bgemm_284_out_0_1_gemm_284_conversion_layer, 284,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &transpose_bgemm_284_out_0_1_gemm_284_conversion_chain,
  NULL, &gemm_284_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  transpose_bgemm_284_out_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_276_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_bgemm_284_out_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  transpose_bgemm_284_out_layer, 284,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &transpose_bgemm_284_out_chain,
  NULL, &transpose_bgemm_284_out_0_1_gemm_284_conversion_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_HEIGHT, AI_SHAPE_CHANNEL, AI_SHAPE_WIDTH, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_276_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_274_output, &tiny_bert_generator_enc_3_mha_wk_BiasAdd_ReadVariableOp_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_276_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_276_layer, 276,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_276_chain,
  NULL, &transpose_bgemm_284_out_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_274_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_262_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_274_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &gemm_274_weights, &gemm_26_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_274_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_274_layer, 274,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &gemm_274_chain,
  NULL, &eltwise_276_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  transpose_283_0_0_gemm_284_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_283_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_283_0_0_gemm_284_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  transpose_283_0_0_gemm_284_conversion_layer, 283,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &transpose_283_0_0_gemm_284_conversion_chain,
  NULL, &gemm_274_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  transpose_283_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_281_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_283_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  transpose_283_layer, 283,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &transpose_283_chain,
  NULL, &transpose_283_0_0_gemm_284_conversion_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_CHANNEL, AI_SHAPE_HEIGHT, AI_SHAPE_WIDTH, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_281_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_279_output, &tiny_bert_generator_enc_3_mha_wq_BiasAdd_ReadVariableOp_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_281_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_281_layer, 281,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_281_chain,
  NULL, &transpose_283_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_279_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_262_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_279_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &gemm_279_weights, &gemm_26_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_279_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_279_layer, 279,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &gemm_279_chain,
  NULL, &eltwise_281_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  transpose_291_0_1_gemm_292_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_291_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_291_0_1_gemm_292_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  transpose_291_0_1_gemm_292_conversion_layer, 291,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &transpose_291_0_1_gemm_292_conversion_chain,
  NULL, &gemm_279_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  transpose_291_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_289_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_291_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  transpose_291_layer, 291,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &transpose_291_chain,
  NULL, &transpose_291_0_1_gemm_292_conversion_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_CHANNEL, AI_SHAPE_HEIGHT, AI_SHAPE_WIDTH, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_289_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_287_output, &tiny_bert_generator_enc_3_mha_wv_BiasAdd_ReadVariableOp_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_289_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_289_layer, 289,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_289_chain,
  NULL, &transpose_291_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_287_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_262_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_287_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &gemm_287_weights, &gemm_26_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_287_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_287_layer, 287,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &gemm_287_chain,
  NULL, &eltwise_289_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_262_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_259_output, &eltwise_261_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_262_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_262_layer, 262,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_262_chain,
  NULL, &gemm_287_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_259_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_252_output, &eltwise_258_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_259_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_259_layer, 259,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_259_chain,
  NULL, &eltwise_262_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_261_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &tiny_bert_generator_enc_2_ln2_batchnorm_ReadVariableOp_3D, &eltwise_260_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_261_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_261_layer, 261,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_261_chain,
  NULL, &eltwise_259_layer, AI_STATIC, 
  .operation = ai_sub_f32, 
  .buffer_operation = ai_sub_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_260_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &reduce_253_Mul_0_1_eltwise_254_conversion_output, &eltwise_258_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_260_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_260_layer, 260,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_260_chain,
  NULL, &eltwise_261_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_258_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &nl_257_output, &tiny_bert_generator_enc_2_ln2_batchnorm_mul_ReadVariableOp_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_258_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_258_layer, 258,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_258_chain,
  NULL, &eltwise_260_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)


AI_STATIC_CONST ai_i8 nl_257_nl_params_data[] = { -128, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_257_nl_params, AI_ARRAY_FORMAT_S8,
    nl_257_nl_params_data, nl_257_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_257_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_256_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_257_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_257_layer, 257,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_257_chain,
  NULL, &eltwise_258_layer, AI_STATIC, 
  .nl_params = &nl_257_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_256_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &reduce_255_Mul_0_0_eltwise_256_conversion_output, &tiny_bert_generator_enc_0_ln1_batchnorm_add_y_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_256_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_256_layer, 256,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_256_chain,
  NULL, &nl_257_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_255_Mul_0_0_eltwise_256_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_255_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_255_Mul_0_0_eltwise_256_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_255_Mul_0_0_eltwise_256_conversion_layer, 255,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &reduce_255_Mul_0_0_eltwise_256_conversion_chain,
  NULL, &eltwise_256_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_255_Mul_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_255_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_255_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &reduce_46_Mul_scale, &reduce_46_Mul_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_255_Mul_layer, 255,
  BN_TYPE, 0x0, NULL,
  bn, forward_bn,
  &reduce_255_Mul_chain,
  NULL, &reduce_255_Mul_0_0_eltwise_256_conversion_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_float reduce_255_neutral_value_data[] = { 0.0f };
AI_ARRAY_OBJ_DECLARE(
    reduce_255_neutral_value, AI_ARRAY_FORMAT_FLOAT,
    reduce_255_neutral_value_data, reduce_255_neutral_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_255_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_254_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_255_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_255_layer, 255,
  REDUCE_TYPE, 0x0, NULL,
  reduce, forward_reduce,
  &reduce_255_chain,
  NULL, &reduce_255_Mul_layer, AI_STATIC, 
  .operation = ai_sum, 
  .neutral_value = &reduce_255_neutral_value, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_254_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_252_0_0_reduce_253_conversion_output, &reduce_253_Mul_0_1_eltwise_254_conversion_0_1_eltwise_254_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_254_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_254_layer, 254,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_254_chain,
  NULL, &reduce_255_layer, AI_STATIC, 
  .operation = ai_squared_diff, 
  .buffer_operation = ai_squared_diff_buffer, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_253_Mul_0_1_eltwise_254_conversion_0_1_eltwise_254_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_253_Mul_0_1_eltwise_254_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_253_Mul_0_1_eltwise_254_conversion_0_1_eltwise_254_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_253_Mul_0_1_eltwise_254_conversion_0_1_eltwise_254_conversion_layer, 253,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &reduce_253_Mul_0_1_eltwise_254_conversion_0_1_eltwise_254_conversion_chain,
  NULL, &eltwise_254_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_253_Mul_0_1_eltwise_254_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_253_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_253_Mul_0_1_eltwise_254_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_253_Mul_0_1_eltwise_254_conversion_layer, 253,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &reduce_253_Mul_0_1_eltwise_254_conversion_chain,
  NULL, &reduce_253_Mul_0_1_eltwise_254_conversion_0_1_eltwise_254_conversion_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_253_Mul_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_253_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_253_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &reduce_46_Mul_scale, &reduce_46_Mul_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_253_Mul_layer, 253,
  BN_TYPE, 0x0, NULL,
  bn, forward_bn,
  &reduce_253_Mul_chain,
  NULL, &reduce_253_Mul_0_1_eltwise_254_conversion_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_float reduce_253_neutral_value_data[] = { 0.0f };
AI_ARRAY_OBJ_DECLARE(
    reduce_253_neutral_value, AI_ARRAY_FORMAT_FLOAT,
    reduce_253_neutral_value_data, reduce_253_neutral_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_253_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_252_0_0_reduce_253_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_253_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_253_layer, 253,
  REDUCE_TYPE, 0x0, NULL,
  reduce, forward_reduce,
  &reduce_253_chain,
  NULL, &reduce_253_Mul_layer, AI_STATIC, 
  .operation = ai_sum, 
  .neutral_value = &reduce_253_neutral_value, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_252_0_0_reduce_253_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_252_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_252_0_0_reduce_253_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_252_0_0_reduce_253_conversion_layer, 252,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &eltwise_252_0_0_reduce_253_conversion_chain,
  NULL, &reduce_253_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_252_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_229_output, &eltwise_251_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_252_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_252_layer, 252,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_252_chain,
  NULL, &eltwise_252_0_0_reduce_253_conversion_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_251_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_249_output, &tiny_bert_generator_enc_2_ffn_dense_29_BiasAdd_ReadVariableOp_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_251_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_251_layer, 251,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_251_chain,
  NULL, &eltwise_252_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_249_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_240_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_249_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &gemm_249_weights, &gemm_26_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_249_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_249_layer, 249,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &gemm_249_chain,
  NULL, &eltwise_251_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_240_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_238_output, &tiny_bert_generator_enc_2_ffn_dense_28_BiasAdd_ReadVariableOp_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_240_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_240_layer, 240,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_240_chain,
  NULL, &gemm_249_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_238_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_229_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_238_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &gemm_238_weights, &gemm_64_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_238_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_238_layer, 238,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &gemm_238_chain,
  NULL, &eltwise_240_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_229_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_226_output, &eltwise_228_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_229_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_229_layer, 229,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_229_chain,
  NULL, &gemm_238_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_226_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_219_output, &eltwise_225_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_226_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_226_layer, 226,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_226_chain,
  NULL, &eltwise_229_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_228_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &tiny_bert_generator_enc_2_ln1_batchnorm_ReadVariableOp_3D, &eltwise_227_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_228_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_228_layer, 228,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_228_chain,
  NULL, &eltwise_226_layer, AI_STATIC, 
  .operation = ai_sub_f32, 
  .buffer_operation = ai_sub_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_227_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &reduce_220_Mul_0_1_eltwise_221_conversion_output, &eltwise_225_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_227_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_227_layer, 227,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_227_chain,
  NULL, &eltwise_228_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_225_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &nl_224_output, &tiny_bert_generator_enc_2_ln1_batchnorm_mul_ReadVariableOp_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_225_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_225_layer, 225,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_225_chain,
  NULL, &eltwise_227_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)


AI_STATIC_CONST ai_i8 nl_224_nl_params_data[] = { -128, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_224_nl_params, AI_ARRAY_FORMAT_S8,
    nl_224_nl_params_data, nl_224_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_224_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_223_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_224_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_224_layer, 224,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_224_chain,
  NULL, &eltwise_225_layer, AI_STATIC, 
  .nl_params = &nl_224_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_223_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &reduce_222_Mul_0_0_eltwise_223_conversion_output, &tiny_bert_generator_enc_0_ln1_batchnorm_add_y_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_223_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_223_layer, 223,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_223_chain,
  NULL, &nl_224_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_222_Mul_0_0_eltwise_223_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_222_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_222_Mul_0_0_eltwise_223_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_222_Mul_0_0_eltwise_223_conversion_layer, 222,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &reduce_222_Mul_0_0_eltwise_223_conversion_chain,
  NULL, &eltwise_223_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_222_Mul_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_222_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_222_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &reduce_46_Mul_scale, &reduce_46_Mul_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_222_Mul_layer, 222,
  BN_TYPE, 0x0, NULL,
  bn, forward_bn,
  &reduce_222_Mul_chain,
  NULL, &reduce_222_Mul_0_0_eltwise_223_conversion_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_float reduce_222_neutral_value_data[] = { 0.0f };
AI_ARRAY_OBJ_DECLARE(
    reduce_222_neutral_value, AI_ARRAY_FORMAT_FLOAT,
    reduce_222_neutral_value_data, reduce_222_neutral_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_222_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_221_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_222_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_222_layer, 222,
  REDUCE_TYPE, 0x0, NULL,
  reduce, forward_reduce,
  &reduce_222_chain,
  NULL, &reduce_222_Mul_layer, AI_STATIC, 
  .operation = ai_sum, 
  .neutral_value = &reduce_222_neutral_value, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_221_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_219_0_0_reduce_220_conversion_output, &reduce_220_Mul_0_1_eltwise_221_conversion_0_1_eltwise_221_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_221_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_221_layer, 221,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_221_chain,
  NULL, &reduce_222_layer, AI_STATIC, 
  .operation = ai_squared_diff, 
  .buffer_operation = ai_squared_diff_buffer, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_220_Mul_0_1_eltwise_221_conversion_0_1_eltwise_221_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_220_Mul_0_1_eltwise_221_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_220_Mul_0_1_eltwise_221_conversion_0_1_eltwise_221_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_220_Mul_0_1_eltwise_221_conversion_0_1_eltwise_221_conversion_layer, 220,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &reduce_220_Mul_0_1_eltwise_221_conversion_0_1_eltwise_221_conversion_chain,
  NULL, &eltwise_221_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_220_Mul_0_1_eltwise_221_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_220_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_220_Mul_0_1_eltwise_221_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_220_Mul_0_1_eltwise_221_conversion_layer, 220,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &reduce_220_Mul_0_1_eltwise_221_conversion_chain,
  NULL, &reduce_220_Mul_0_1_eltwise_221_conversion_0_1_eltwise_221_conversion_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_220_Mul_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_220_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_220_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &reduce_46_Mul_scale, &reduce_46_Mul_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_220_Mul_layer, 220,
  BN_TYPE, 0x0, NULL,
  bn, forward_bn,
  &reduce_220_Mul_chain,
  NULL, &reduce_220_Mul_0_1_eltwise_221_conversion_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_float reduce_220_neutral_value_data[] = { 0.0f };
AI_ARRAY_OBJ_DECLARE(
    reduce_220_neutral_value, AI_ARRAY_FORMAT_FLOAT,
    reduce_220_neutral_value_data, reduce_220_neutral_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_220_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_219_0_0_reduce_220_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_220_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_220_layer, 220,
  REDUCE_TYPE, 0x0, NULL,
  reduce, forward_reduce,
  &reduce_220_chain,
  NULL, &reduce_220_Mul_layer, AI_STATIC, 
  .operation = ai_sum, 
  .neutral_value = &reduce_220_neutral_value, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_219_0_0_reduce_220_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_219_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_219_0_0_reduce_220_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_219_0_0_reduce_220_conversion_layer, 219,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &eltwise_219_0_0_reduce_220_conversion_chain,
  NULL, &reduce_220_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_219_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_175_output, &eltwise_218_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_219_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_219_layer, 219,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_219_chain,
  NULL, &eltwise_219_0_0_reduce_220_conversion_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_218_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_216_output, &tiny_bert_generator_enc_2_mha_output_dense_BiasAdd_ReadVariableOp_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_218_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_218_layer, 218,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_218_chain,
  NULL, &eltwise_219_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_216_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_206_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_216_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &gemm_216_weights, &gemm_26_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_216_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_216_layer, 216,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &gemm_216_chain,
  NULL, &eltwise_218_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  transpose_206_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_205_0_0_transpose_206_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_206_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  transpose_206_layer, 206,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &transpose_206_chain,
  NULL, &gemm_216_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_CHANNEL, AI_SHAPE_HEIGHT, AI_SHAPE_WIDTH, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_205_0_0_transpose_206_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_205_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_205_0_0_transpose_206_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_205_0_0_transpose_206_conversion_layer, 205,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &gemm_205_0_0_transpose_206_conversion_chain,
  NULL, &transpose_206_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_205_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &nl_199_0_0_gemm_205_conversion_output, &transpose_204_0_1_gemm_205_conversion_output, &gemm_205_bias_0_2_gemm_205_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_205_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_205_layer, 205,
  MATMUL_TYPE, 0x0, NULL,
  matmul, forward_matmul,
  &gemm_205_chain,
  NULL, &gemm_205_0_0_transpose_206_conversion_layer, AI_STATIC, 
  .alpha = 1.0, 
  .beta = 1.0, 
  .tA = 0, 
  .tB = 0, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_199_0_0_gemm_205_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_199_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_199_0_0_gemm_205_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_199_0_0_gemm_205_conversion_layer, 199,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &nl_199_0_0_gemm_205_conversion_chain,
  NULL, &gemm_205_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i32 nl_199_nl_params_data[] = { 2033400832, 26, -31 };
AI_ARRAY_OBJ_DECLARE(
    nl_199_nl_params, AI_ARRAY_FORMAT_S32,
    nl_199_nl_params_data, nl_199_nl_params_data, 3, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_199_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_198_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_199_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_199_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  nl_199_layer, 199,
  SM_TYPE, 0x0, NULL,
  sm, forward_sm_integer,
  &nl_199_chain,
  NULL, &nl_199_0_0_gemm_205_conversion_layer, AI_STATIC, 
  .nl_params = &nl_199_nl_params, 
  .axis = AI_SHAPE_CHANNEL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_198_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_197_0_0_eltwise_198_conversion_output, &tiny_bert_generator_enc_3_mha_truedivtiny_bert_generator_enc_3_mha_Sqrt_4D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_198_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_198_layer, 198,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_198_chain,
  NULL, &nl_199_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_197_0_0_eltwise_198_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_197_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_197_0_0_eltwise_198_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_197_0_0_eltwise_198_conversion_layer, 197,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &gemm_197_0_0_eltwise_198_conversion_chain,
  NULL, &eltwise_198_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_197_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &transpose_196_0_0_gemm_197_conversion_output, &transpose_bgemm_197_out_0_1_gemm_197_conversion_output, &gemm_197_bias_0_2_gemm_197_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_197_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_197_layer, 197,
  MATMUL_TYPE, 0x0, NULL,
  matmul, forward_matmul,
  &gemm_197_chain,
  NULL, &gemm_197_0_0_eltwise_198_conversion_layer, AI_STATIC, 
  .alpha = 1.0, 
  .beta = 1.0, 
  .tA = 0, 
  .tB = 0, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  transpose_bgemm_197_out_0_1_gemm_197_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_bgemm_197_out_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_bgemm_197_out_0_1_gemm_197_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  transpose_bgemm_197_out_0_1_gemm_197_conversion_layer, 197,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &transpose_bgemm_197_out_0_1_gemm_197_conversion_chain,
  NULL, &gemm_197_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  transpose_bgemm_197_out_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_189_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_bgemm_197_out_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  transpose_bgemm_197_out_layer, 197,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &transpose_bgemm_197_out_chain,
  NULL, &transpose_bgemm_197_out_0_1_gemm_197_conversion_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_HEIGHT, AI_SHAPE_CHANNEL, AI_SHAPE_WIDTH, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_189_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_187_output, &tiny_bert_generator_enc_2_mha_wk_BiasAdd_ReadVariableOp_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_189_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_189_layer, 189,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_189_chain,
  NULL, &transpose_bgemm_197_out_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_187_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_175_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_187_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &gemm_187_weights, &gemm_26_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_187_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_187_layer, 187,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &gemm_187_chain,
  NULL, &eltwise_189_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  transpose_196_0_0_gemm_197_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_196_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_196_0_0_gemm_197_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  transpose_196_0_0_gemm_197_conversion_layer, 196,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &transpose_196_0_0_gemm_197_conversion_chain,
  NULL, &gemm_187_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  transpose_196_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_194_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_196_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  transpose_196_layer, 196,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &transpose_196_chain,
  NULL, &transpose_196_0_0_gemm_197_conversion_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_CHANNEL, AI_SHAPE_HEIGHT, AI_SHAPE_WIDTH, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_194_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_192_output, &tiny_bert_generator_enc_2_mha_wq_BiasAdd_ReadVariableOp_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_194_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_194_layer, 194,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_194_chain,
  NULL, &transpose_196_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_192_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_175_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_192_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &gemm_192_weights, &gemm_26_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_192_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_192_layer, 192,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &gemm_192_chain,
  NULL, &eltwise_194_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  transpose_204_0_1_gemm_205_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_204_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_204_0_1_gemm_205_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  transpose_204_0_1_gemm_205_conversion_layer, 204,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &transpose_204_0_1_gemm_205_conversion_chain,
  NULL, &gemm_192_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  transpose_204_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_202_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_204_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  transpose_204_layer, 204,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &transpose_204_chain,
  NULL, &transpose_204_0_1_gemm_205_conversion_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_CHANNEL, AI_SHAPE_HEIGHT, AI_SHAPE_WIDTH, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_202_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_200_output, &tiny_bert_generator_enc_2_mha_wv_BiasAdd_ReadVariableOp_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_202_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_202_layer, 202,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_202_chain,
  NULL, &transpose_204_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_200_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_175_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_200_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &gemm_200_weights, &gemm_26_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_200_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_200_layer, 200,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &gemm_200_chain,
  NULL, &eltwise_202_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_175_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_172_output, &eltwise_174_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_175_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_175_layer, 175,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_175_chain,
  NULL, &gemm_200_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_172_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_165_output, &eltwise_171_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_172_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_172_layer, 172,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_172_chain,
  NULL, &eltwise_175_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_174_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &tiny_bert_generator_enc_1_ln2_batchnorm_ReadVariableOp_3D, &eltwise_173_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_174_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_174_layer, 174,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_174_chain,
  NULL, &eltwise_172_layer, AI_STATIC, 
  .operation = ai_sub_f32, 
  .buffer_operation = ai_sub_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_173_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &reduce_166_Mul_0_1_eltwise_167_conversion_output, &eltwise_171_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_173_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_173_layer, 173,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_173_chain,
  NULL, &eltwise_174_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_171_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &nl_170_output, &tiny_bert_generator_enc_1_ln2_batchnorm_mul_ReadVariableOp_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_171_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_171_layer, 171,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_171_chain,
  NULL, &eltwise_173_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)


AI_STATIC_CONST ai_i8 nl_170_nl_params_data[] = { -128, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_170_nl_params, AI_ARRAY_FORMAT_S8,
    nl_170_nl_params_data, nl_170_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_170_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_169_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_170_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_170_layer, 170,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_170_chain,
  NULL, &eltwise_171_layer, AI_STATIC, 
  .nl_params = &nl_170_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_169_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &reduce_168_Mul_0_0_eltwise_169_conversion_output, &tiny_bert_generator_enc_0_ln1_batchnorm_add_y_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_169_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_169_layer, 169,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_169_chain,
  NULL, &nl_170_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_168_Mul_0_0_eltwise_169_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_168_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_168_Mul_0_0_eltwise_169_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_168_Mul_0_0_eltwise_169_conversion_layer, 168,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &reduce_168_Mul_0_0_eltwise_169_conversion_chain,
  NULL, &eltwise_169_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_168_Mul_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_168_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_168_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &reduce_46_Mul_scale, &reduce_46_Mul_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_168_Mul_layer, 168,
  BN_TYPE, 0x0, NULL,
  bn, forward_bn,
  &reduce_168_Mul_chain,
  NULL, &reduce_168_Mul_0_0_eltwise_169_conversion_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_float reduce_168_neutral_value_data[] = { 0.0f };
AI_ARRAY_OBJ_DECLARE(
    reduce_168_neutral_value, AI_ARRAY_FORMAT_FLOAT,
    reduce_168_neutral_value_data, reduce_168_neutral_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_168_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_167_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_168_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_168_layer, 168,
  REDUCE_TYPE, 0x0, NULL,
  reduce, forward_reduce,
  &reduce_168_chain,
  NULL, &reduce_168_Mul_layer, AI_STATIC, 
  .operation = ai_sum, 
  .neutral_value = &reduce_168_neutral_value, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_167_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_165_0_0_reduce_166_conversion_output, &reduce_166_Mul_0_1_eltwise_167_conversion_0_1_eltwise_167_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_167_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_167_layer, 167,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_167_chain,
  NULL, &reduce_168_layer, AI_STATIC, 
  .operation = ai_squared_diff, 
  .buffer_operation = ai_squared_diff_buffer, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_166_Mul_0_1_eltwise_167_conversion_0_1_eltwise_167_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_166_Mul_0_1_eltwise_167_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_166_Mul_0_1_eltwise_167_conversion_0_1_eltwise_167_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_166_Mul_0_1_eltwise_167_conversion_0_1_eltwise_167_conversion_layer, 166,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &reduce_166_Mul_0_1_eltwise_167_conversion_0_1_eltwise_167_conversion_chain,
  NULL, &eltwise_167_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_166_Mul_0_1_eltwise_167_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_166_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_166_Mul_0_1_eltwise_167_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_166_Mul_0_1_eltwise_167_conversion_layer, 166,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &reduce_166_Mul_0_1_eltwise_167_conversion_chain,
  NULL, &reduce_166_Mul_0_1_eltwise_167_conversion_0_1_eltwise_167_conversion_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_166_Mul_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_166_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_166_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &reduce_46_Mul_scale, &reduce_46_Mul_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_166_Mul_layer, 166,
  BN_TYPE, 0x0, NULL,
  bn, forward_bn,
  &reduce_166_Mul_chain,
  NULL, &reduce_166_Mul_0_1_eltwise_167_conversion_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_float reduce_166_neutral_value_data[] = { 0.0f };
AI_ARRAY_OBJ_DECLARE(
    reduce_166_neutral_value, AI_ARRAY_FORMAT_FLOAT,
    reduce_166_neutral_value_data, reduce_166_neutral_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_166_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_165_0_0_reduce_166_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_166_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_166_layer, 166,
  REDUCE_TYPE, 0x0, NULL,
  reduce, forward_reduce,
  &reduce_166_chain,
  NULL, &reduce_166_Mul_layer, AI_STATIC, 
  .operation = ai_sum, 
  .neutral_value = &reduce_166_neutral_value, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_165_0_0_reduce_166_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_165_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_165_0_0_reduce_166_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_165_0_0_reduce_166_conversion_layer, 165,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &eltwise_165_0_0_reduce_166_conversion_chain,
  NULL, &reduce_166_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_165_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_142_output, &eltwise_164_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_165_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_165_layer, 165,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_165_chain,
  NULL, &eltwise_165_0_0_reduce_166_conversion_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_164_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_162_output, &tiny_bert_generator_enc_1_ffn_dense_27_BiasAdd_ReadVariableOp_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_164_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_164_layer, 164,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_164_chain,
  NULL, &eltwise_165_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_162_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_153_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_162_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &gemm_162_weights, &gemm_26_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_162_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_162_layer, 162,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &gemm_162_chain,
  NULL, &eltwise_164_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_153_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_151_output, &tiny_bert_generator_enc_1_ffn_dense_26_BiasAdd_ReadVariableOp_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_153_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_153_layer, 153,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_153_chain,
  NULL, &gemm_162_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_151_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_142_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_151_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &gemm_151_weights, &gemm_64_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_151_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_151_layer, 151,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &gemm_151_chain,
  NULL, &eltwise_153_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_142_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_139_output, &eltwise_141_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_142_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_142_layer, 142,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_142_chain,
  NULL, &gemm_151_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_139_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_132_output, &eltwise_138_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_139_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_139_layer, 139,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_139_chain,
  NULL, &eltwise_142_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_141_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &tiny_bert_generator_enc_1_ln1_batchnorm_ReadVariableOp_3D, &eltwise_140_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_141_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_141_layer, 141,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_141_chain,
  NULL, &eltwise_139_layer, AI_STATIC, 
  .operation = ai_sub_f32, 
  .buffer_operation = ai_sub_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_140_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &reduce_133_Mul_0_1_eltwise_134_conversion_output, &eltwise_138_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_140_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_140_layer, 140,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_140_chain,
  NULL, &eltwise_141_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_138_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &nl_137_output, &tiny_bert_generator_enc_1_ln1_batchnorm_mul_ReadVariableOp_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_138_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_138_layer, 138,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_138_chain,
  NULL, &eltwise_140_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)


AI_STATIC_CONST ai_i8 nl_137_nl_params_data[] = { -128, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 126, 126, 125, 125, 124, 124, 123, 123, 122 };
AI_ARRAY_OBJ_DECLARE(
    nl_137_nl_params, AI_ARRAY_FORMAT_S8,
    nl_137_nl_params_data, nl_137_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_137_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_136_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_137_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_137_layer, 137,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_137_chain,
  NULL, &eltwise_138_layer, AI_STATIC, 
  .nl_params = &nl_137_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_136_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &reduce_135_Mul_0_0_eltwise_136_conversion_output, &tiny_bert_generator_enc_0_ln1_batchnorm_add_y_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_136_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_136_layer, 136,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_136_chain,
  NULL, &nl_137_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_135_Mul_0_0_eltwise_136_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_135_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_135_Mul_0_0_eltwise_136_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_135_Mul_0_0_eltwise_136_conversion_layer, 135,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &reduce_135_Mul_0_0_eltwise_136_conversion_chain,
  NULL, &eltwise_136_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_135_Mul_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_135_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_135_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &reduce_46_Mul_scale, &reduce_46_Mul_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_135_Mul_layer, 135,
  BN_TYPE, 0x0, NULL,
  bn, forward_bn,
  &reduce_135_Mul_chain,
  NULL, &reduce_135_Mul_0_0_eltwise_136_conversion_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_float reduce_135_neutral_value_data[] = { 0.0f };
AI_ARRAY_OBJ_DECLARE(
    reduce_135_neutral_value, AI_ARRAY_FORMAT_FLOAT,
    reduce_135_neutral_value_data, reduce_135_neutral_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_135_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_134_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_135_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_135_layer, 135,
  REDUCE_TYPE, 0x0, NULL,
  reduce, forward_reduce,
  &reduce_135_chain,
  NULL, &reduce_135_Mul_layer, AI_STATIC, 
  .operation = ai_sum, 
  .neutral_value = &reduce_135_neutral_value, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_134_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_132_0_0_reduce_133_conversion_output, &reduce_133_Mul_0_1_eltwise_134_conversion_0_1_eltwise_134_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_134_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_134_layer, 134,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_134_chain,
  NULL, &reduce_135_layer, AI_STATIC, 
  .operation = ai_squared_diff, 
  .buffer_operation = ai_squared_diff_buffer, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_133_Mul_0_1_eltwise_134_conversion_0_1_eltwise_134_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_133_Mul_0_1_eltwise_134_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_133_Mul_0_1_eltwise_134_conversion_0_1_eltwise_134_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_133_Mul_0_1_eltwise_134_conversion_0_1_eltwise_134_conversion_layer, 133,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &reduce_133_Mul_0_1_eltwise_134_conversion_0_1_eltwise_134_conversion_chain,
  NULL, &eltwise_134_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_133_Mul_0_1_eltwise_134_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_133_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_133_Mul_0_1_eltwise_134_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_133_Mul_0_1_eltwise_134_conversion_layer, 133,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &reduce_133_Mul_0_1_eltwise_134_conversion_chain,
  NULL, &reduce_133_Mul_0_1_eltwise_134_conversion_0_1_eltwise_134_conversion_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_133_Mul_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_133_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_133_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &reduce_46_Mul_scale, &reduce_46_Mul_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_133_Mul_layer, 133,
  BN_TYPE, 0x0, NULL,
  bn, forward_bn,
  &reduce_133_Mul_chain,
  NULL, &reduce_133_Mul_0_1_eltwise_134_conversion_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_float reduce_133_neutral_value_data[] = { 0.0f };
AI_ARRAY_OBJ_DECLARE(
    reduce_133_neutral_value, AI_ARRAY_FORMAT_FLOAT,
    reduce_133_neutral_value_data, reduce_133_neutral_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_133_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_132_0_0_reduce_133_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_133_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_133_layer, 133,
  REDUCE_TYPE, 0x0, NULL,
  reduce, forward_reduce,
  &reduce_133_chain,
  NULL, &reduce_133_Mul_layer, AI_STATIC, 
  .operation = ai_sum, 
  .neutral_value = &reduce_133_neutral_value, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_132_0_0_reduce_133_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_132_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_132_0_0_reduce_133_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_132_0_0_reduce_133_conversion_layer, 132,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &eltwise_132_0_0_reduce_133_conversion_chain,
  NULL, &reduce_133_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_132_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_88_output, &eltwise_131_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_132_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_132_layer, 132,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_132_chain,
  NULL, &eltwise_132_0_0_reduce_133_conversion_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_131_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_129_output, &tiny_bert_generator_enc_1_mha_output_dense_BiasAdd_ReadVariableOp_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_131_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_131_layer, 131,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_131_chain,
  NULL, &eltwise_132_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_129_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_119_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_129_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &gemm_129_weights, &gemm_26_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_129_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_129_layer, 129,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &gemm_129_chain,
  NULL, &eltwise_131_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  transpose_119_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_118_0_0_transpose_119_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_119_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  transpose_119_layer, 119,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &transpose_119_chain,
  NULL, &gemm_129_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_CHANNEL, AI_SHAPE_HEIGHT, AI_SHAPE_WIDTH, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_118_0_0_transpose_119_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_118_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_118_0_0_transpose_119_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_118_0_0_transpose_119_conversion_layer, 118,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &gemm_118_0_0_transpose_119_conversion_chain,
  NULL, &transpose_119_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_118_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &nl_112_0_0_gemm_118_conversion_output, &transpose_117_0_1_gemm_118_conversion_output, &gemm_118_bias_0_2_gemm_118_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_118_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_118_layer, 118,
  MATMUL_TYPE, 0x0, NULL,
  matmul, forward_matmul,
  &gemm_118_chain,
  NULL, &gemm_118_0_0_transpose_119_conversion_layer, AI_STATIC, 
  .alpha = 1.0, 
  .beta = 1.0, 
  .tA = 0, 
  .tB = 0, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_112_0_0_gemm_118_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_112_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_112_0_0_gemm_118_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_112_0_0_gemm_118_conversion_layer, 112,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &nl_112_0_0_gemm_118_conversion_chain,
  NULL, &gemm_118_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i32 nl_112_nl_params_data[] = { 1734061568, 25, -62 };
AI_ARRAY_OBJ_DECLARE(
    nl_112_nl_params, AI_ARRAY_FORMAT_S32,
    nl_112_nl_params_data, nl_112_nl_params_data, 3, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_112_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_111_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_112_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_112_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  nl_112_layer, 112,
  SM_TYPE, 0x0, NULL,
  sm, forward_sm_integer,
  &nl_112_chain,
  NULL, &nl_112_0_0_gemm_118_conversion_layer, AI_STATIC, 
  .nl_params = &nl_112_nl_params, 
  .axis = AI_SHAPE_CHANNEL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_111_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_110_0_0_eltwise_111_conversion_output, &tiny_bert_generator_enc_3_mha_truedivtiny_bert_generator_enc_3_mha_Sqrt_4D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_111_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_111_layer, 111,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_111_chain,
  NULL, &nl_112_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_110_0_0_eltwise_111_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_110_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_110_0_0_eltwise_111_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_110_0_0_eltwise_111_conversion_layer, 110,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &gemm_110_0_0_eltwise_111_conversion_chain,
  NULL, &eltwise_111_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_110_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &transpose_109_0_0_gemm_110_conversion_output, &transpose_bgemm_110_out_0_1_gemm_110_conversion_output, &gemm_110_bias_0_2_gemm_110_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_110_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_110_layer, 110,
  MATMUL_TYPE, 0x0, NULL,
  matmul, forward_matmul,
  &gemm_110_chain,
  NULL, &gemm_110_0_0_eltwise_111_conversion_layer, AI_STATIC, 
  .alpha = 1.0, 
  .beta = 1.0, 
  .tA = 0, 
  .tB = 0, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  transpose_bgemm_110_out_0_1_gemm_110_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_bgemm_110_out_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_bgemm_110_out_0_1_gemm_110_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  transpose_bgemm_110_out_0_1_gemm_110_conversion_layer, 110,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &transpose_bgemm_110_out_0_1_gemm_110_conversion_chain,
  NULL, &gemm_110_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  transpose_bgemm_110_out_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_102_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_bgemm_110_out_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  transpose_bgemm_110_out_layer, 110,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &transpose_bgemm_110_out_chain,
  NULL, &transpose_bgemm_110_out_0_1_gemm_110_conversion_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_HEIGHT, AI_SHAPE_CHANNEL, AI_SHAPE_WIDTH, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_102_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_100_output, &tiny_bert_generator_enc_1_mha_wk_BiasAdd_ReadVariableOp_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_102_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_102_layer, 102,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_102_chain,
  NULL, &transpose_bgemm_110_out_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_100_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_88_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_100_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &gemm_100_weights, &gemm_26_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_100_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_100_layer, 100,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &gemm_100_chain,
  NULL, &eltwise_102_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  transpose_109_0_0_gemm_110_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_109_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_109_0_0_gemm_110_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  transpose_109_0_0_gemm_110_conversion_layer, 109,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &transpose_109_0_0_gemm_110_conversion_chain,
  NULL, &gemm_100_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  transpose_109_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_107_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_109_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  transpose_109_layer, 109,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &transpose_109_chain,
  NULL, &transpose_109_0_0_gemm_110_conversion_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_CHANNEL, AI_SHAPE_HEIGHT, AI_SHAPE_WIDTH, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_107_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_105_output, &tiny_bert_generator_enc_1_mha_wq_BiasAdd_ReadVariableOp_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_107_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_107_layer, 107,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_107_chain,
  NULL, &transpose_109_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_105_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_88_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_105_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &gemm_105_weights, &gemm_26_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_105_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_105_layer, 105,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &gemm_105_chain,
  NULL, &eltwise_107_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  transpose_117_0_1_gemm_118_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_117_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_117_0_1_gemm_118_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  transpose_117_0_1_gemm_118_conversion_layer, 117,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &transpose_117_0_1_gemm_118_conversion_chain,
  NULL, &gemm_105_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  transpose_117_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_115_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_117_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  transpose_117_layer, 117,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &transpose_117_chain,
  NULL, &transpose_117_0_1_gemm_118_conversion_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_CHANNEL, AI_SHAPE_HEIGHT, AI_SHAPE_WIDTH, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_115_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_113_output, &tiny_bert_generator_enc_1_mha_wv_BiasAdd_ReadVariableOp_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_115_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_115_layer, 115,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_115_chain,
  NULL, &transpose_117_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_113_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_88_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_113_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &gemm_113_weights, &gemm_26_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_113_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_113_layer, 113,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &gemm_113_chain,
  NULL, &eltwise_115_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_88_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_85_output, &eltwise_87_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_88_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_88_layer, 88,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_88_chain,
  NULL, &gemm_113_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_85_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_78_output, &eltwise_84_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_85_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_85_layer, 85,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_85_chain,
  NULL, &eltwise_88_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_87_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &tiny_bert_generator_enc_0_ln2_batchnorm_ReadVariableOp_3D, &eltwise_86_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_87_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_87_layer, 87,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_87_chain,
  NULL, &eltwise_85_layer, AI_STATIC, 
  .operation = ai_sub_f32, 
  .buffer_operation = ai_sub_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_86_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &reduce_79_Mul_0_1_eltwise_80_conversion_output, &eltwise_84_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_86_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_86_layer, 86,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_86_chain,
  NULL, &eltwise_87_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_84_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &nl_83_output, &tiny_bert_generator_enc_0_ln2_batchnorm_mul_ReadVariableOp_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_84_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_84_layer, 84,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_84_chain,
  NULL, &eltwise_86_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)


AI_STATIC_CONST ai_i8 nl_83_nl_params_data[] = { -128, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 126, 124, 122, 120, 118, 117, 115, 113, 111, 110, 108, 107, 105, 104, 102, 101, 99, 98, 97, 95, 94, 93, 91, 90, 89, 88, 86, 85, 84, 83, 82, 81, 80, 79, 78, 77, 76, 75, 74, 73, 72, 71, 70, 69, 68, 67, 66, 65, 64, 64, 63, 62, 61, 60, 60, 59, 58, 57, 56, 56, 55, 54, 54, 53, 52, 51, 51, 50, 49, 49, 48, 47, 47, 46, 46, 45, 44, 44, 43, 42, 42, 41, 41, 40, 40, 39, 38, 38, 37, 37, 36, 36, 35, 35, 34, 34, 33, 33, 32, 32, 31, 31, 30, 30, 29, 29, 28, 28, 28, 27, 27, 26, 26, 25, 25, 24, 24, 24, 23, 23, 22, 22, 22, 21, 21, 20, 20, 20, 19, 19, 19, 18, 18, 17, 17, 17, 16, 16, 16, 15, 15, 15, 14, 14, 14, 13, 13, 13, 12, 12, 12, 11, 11, 11, 10, 10, 10, 9, 9, 9, 8, 8, 8, 7, 7, 7, 7, 6, 6, 6, 5, 5, 5, 5, 4, 4, 4, 4, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0 };
AI_ARRAY_OBJ_DECLARE(
    nl_83_nl_params, AI_ARRAY_FORMAT_S8,
    nl_83_nl_params_data, nl_83_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_83_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_82_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_83_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_83_layer, 83,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_83_chain,
  NULL, &eltwise_84_layer, AI_STATIC, 
  .nl_params = &nl_83_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_82_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &reduce_81_Mul_0_0_eltwise_82_conversion_output, &tiny_bert_generator_enc_0_ln1_batchnorm_add_y_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_82_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_82_layer, 82,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_82_chain,
  NULL, &nl_83_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_81_Mul_0_0_eltwise_82_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_81_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_81_Mul_0_0_eltwise_82_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_81_Mul_0_0_eltwise_82_conversion_layer, 81,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &reduce_81_Mul_0_0_eltwise_82_conversion_chain,
  NULL, &eltwise_82_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_81_Mul_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_81_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_81_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &reduce_46_Mul_scale, &reduce_46_Mul_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_81_Mul_layer, 81,
  BN_TYPE, 0x0, NULL,
  bn, forward_bn,
  &reduce_81_Mul_chain,
  NULL, &reduce_81_Mul_0_0_eltwise_82_conversion_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_float reduce_81_neutral_value_data[] = { 0.0f };
AI_ARRAY_OBJ_DECLARE(
    reduce_81_neutral_value, AI_ARRAY_FORMAT_FLOAT,
    reduce_81_neutral_value_data, reduce_81_neutral_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_81_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_80_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_81_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_81_layer, 81,
  REDUCE_TYPE, 0x0, NULL,
  reduce, forward_reduce,
  &reduce_81_chain,
  NULL, &reduce_81_Mul_layer, AI_STATIC, 
  .operation = ai_sum, 
  .neutral_value = &reduce_81_neutral_value, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_80_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_78_0_0_reduce_79_conversion_output, &reduce_79_Mul_0_1_eltwise_80_conversion_0_1_eltwise_80_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_80_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_80_layer, 80,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_80_chain,
  NULL, &reduce_81_layer, AI_STATIC, 
  .operation = ai_squared_diff, 
  .buffer_operation = ai_squared_diff_buffer, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_79_Mul_0_1_eltwise_80_conversion_0_1_eltwise_80_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_79_Mul_0_1_eltwise_80_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_79_Mul_0_1_eltwise_80_conversion_0_1_eltwise_80_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_79_Mul_0_1_eltwise_80_conversion_0_1_eltwise_80_conversion_layer, 79,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &reduce_79_Mul_0_1_eltwise_80_conversion_0_1_eltwise_80_conversion_chain,
  NULL, &eltwise_80_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_79_Mul_0_1_eltwise_80_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_79_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_79_Mul_0_1_eltwise_80_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_79_Mul_0_1_eltwise_80_conversion_layer, 79,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &reduce_79_Mul_0_1_eltwise_80_conversion_chain,
  NULL, &reduce_79_Mul_0_1_eltwise_80_conversion_0_1_eltwise_80_conversion_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_79_Mul_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_79_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_79_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &reduce_46_Mul_scale, &reduce_46_Mul_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_79_Mul_layer, 79,
  BN_TYPE, 0x0, NULL,
  bn, forward_bn,
  &reduce_79_Mul_chain,
  NULL, &reduce_79_Mul_0_1_eltwise_80_conversion_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_float reduce_79_neutral_value_data[] = { 0.0f };
AI_ARRAY_OBJ_DECLARE(
    reduce_79_neutral_value, AI_ARRAY_FORMAT_FLOAT,
    reduce_79_neutral_value_data, reduce_79_neutral_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_79_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_78_0_0_reduce_79_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_79_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_79_layer, 79,
  REDUCE_TYPE, 0x0, NULL,
  reduce, forward_reduce,
  &reduce_79_chain,
  NULL, &reduce_79_Mul_layer, AI_STATIC, 
  .operation = ai_sum, 
  .neutral_value = &reduce_79_neutral_value, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_78_0_0_reduce_79_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_78_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_78_0_0_reduce_79_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_78_0_0_reduce_79_conversion_layer, 78,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &eltwise_78_0_0_reduce_79_conversion_chain,
  NULL, &reduce_79_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_78_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_55_output, &eltwise_77_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_78_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_78_layer, 78,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_78_chain,
  NULL, &eltwise_78_0_0_reduce_79_conversion_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_77_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_75_output, &tiny_bert_generator_enc_0_ffn_dense_25_BiasAdd_ReadVariableOp_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_77_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_77_layer, 77,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_77_chain,
  NULL, &eltwise_78_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_75_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_66_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_75_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &gemm_75_weights, &gemm_26_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_75_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_75_layer, 75,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &gemm_75_chain,
  NULL, &eltwise_77_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_66_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_64_output, &tiny_bert_generator_enc_0_ffn_dense_24_BiasAdd_ReadVariableOp_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_66_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_66_layer, 66,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_66_chain,
  NULL, &gemm_75_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_64_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_55_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_64_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &gemm_64_weights, &gemm_64_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_64_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_64_layer, 64,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &gemm_64_chain,
  NULL, &eltwise_66_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_55_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_52_output, &eltwise_54_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_55_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_55_layer, 55,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_55_chain,
  NULL, &gemm_64_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_52_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_45_output, &eltwise_51_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_52_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_52_layer, 52,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_52_chain,
  NULL, &eltwise_55_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_54_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &tiny_bert_generator_enc_0_ln1_batchnorm_ReadVariableOp_3D, &eltwise_53_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_54_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_54_layer, 54,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_54_chain,
  NULL, &eltwise_52_layer, AI_STATIC, 
  .operation = ai_sub_f32, 
  .buffer_operation = ai_sub_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_53_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &reduce_46_Mul_0_1_eltwise_47_conversion_output, &eltwise_51_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_53_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_53_layer, 53,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_53_chain,
  NULL, &eltwise_54_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_51_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &nl_50_output, &tiny_bert_generator_enc_0_ln1_batchnorm_mul_ReadVariableOp_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_51_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_51_layer, 51,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_51_chain,
  NULL, &eltwise_53_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)


AI_STATIC_CONST ai_i8 nl_50_nl_params_data[] = { -128, 127, 120, 74, 47, 29, 15, 4, -4, -11, -17, -22, -27, -31, -34, -38, -40, -43, -45, -48, -50, -52, -53, -55, -56, -58, -59, -61, -62, -63, -64, -65, -66, -67, -68, -69, -70, -70, -71, -72, -73, -73, -74, -75, -75, -76, -76, -77, -77, -78, -78, -79, -79, -80, -80, -81, -81, -82, -82, -82, -83, -83, -84, -84, -84, -85, -85, -85, -86, -86, -86, -86, -87, -87, -87, -88, -88, -88, -88, -89, -89, -89, -89, -90, -90, -90, -90, -90, -91, -91, -91, -91, -91, -92, -92, -92, -92, -92, -93, -93, -93, -93, -93, -93, -94, -94, -94, -94, -94, -94, -95, -95, -95, -95, -95, -95, -95, -96, -96, -96, -96, -96, -96, -96, -97, -97, -97, -97, -97, -97, -97, -97, -98, -98, -98, -98, -98, -98, -98, -98, -98, -98, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -101, -101, -101, -101, -101, -101, -101, -101, -101, -101, -101, -101, -102, -102, -102, -102, -102, -102, -102, -102, -102, -102, -102, -102, -102, -102, -103, -103, -103, -103, -103, -103, -103, -103, -103, -103, -103, -103, -103, -103, -103, -103, -104, -104, -104, -104, -104, -104, -104, -104, -104, -104, -104, -104, -104, -104, -104, -104, -104, -104, -105, -105, -105, -105, -105, -105, -105, -105, -105, -105, -105, -105, -105, -105, -105, -105, -105, -105, -105, -105, -106, -106, -106, -106, -106, -106, -106, -106, -106, -106, -106, -106, -106 };
AI_ARRAY_OBJ_DECLARE(
    nl_50_nl_params, AI_ARRAY_FORMAT_S8,
    nl_50_nl_params_data, nl_50_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_50_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_49_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_50_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_50_layer, 50,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_50_chain,
  NULL, &eltwise_51_layer, AI_STATIC, 
  .nl_params = &nl_50_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_49_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &reduce_48_Mul_0_0_eltwise_49_conversion_output, &tiny_bert_generator_enc_0_ln1_batchnorm_add_y_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_49_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_49_layer, 49,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_49_chain,
  NULL, &nl_50_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_48_Mul_0_0_eltwise_49_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_48_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_48_Mul_0_0_eltwise_49_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_48_Mul_0_0_eltwise_49_conversion_layer, 48,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &reduce_48_Mul_0_0_eltwise_49_conversion_chain,
  NULL, &eltwise_49_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_48_Mul_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_48_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_48_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &reduce_46_Mul_scale, &reduce_46_Mul_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_48_Mul_layer, 48,
  BN_TYPE, 0x0, NULL,
  bn, forward_bn,
  &reduce_48_Mul_chain,
  NULL, &reduce_48_Mul_0_0_eltwise_49_conversion_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_float reduce_48_neutral_value_data[] = { 0.0f };
AI_ARRAY_OBJ_DECLARE(
    reduce_48_neutral_value, AI_ARRAY_FORMAT_FLOAT,
    reduce_48_neutral_value_data, reduce_48_neutral_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_48_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_47_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_48_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_48_layer, 48,
  REDUCE_TYPE, 0x0, NULL,
  reduce, forward_reduce,
  &reduce_48_chain,
  NULL, &reduce_48_Mul_layer, AI_STATIC, 
  .operation = ai_sum, 
  .neutral_value = &reduce_48_neutral_value, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_47_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_45_0_0_reduce_46_conversion_output, &reduce_46_Mul_0_1_eltwise_47_conversion_0_1_eltwise_47_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_47_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_47_layer, 47,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_47_chain,
  NULL, &reduce_48_layer, AI_STATIC, 
  .operation = ai_squared_diff, 
  .buffer_operation = ai_squared_diff_buffer, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_46_Mul_0_1_eltwise_47_conversion_0_1_eltwise_47_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_46_Mul_0_1_eltwise_47_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_46_Mul_0_1_eltwise_47_conversion_0_1_eltwise_47_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_46_Mul_0_1_eltwise_47_conversion_0_1_eltwise_47_conversion_layer, 46,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &reduce_46_Mul_0_1_eltwise_47_conversion_0_1_eltwise_47_conversion_chain,
  NULL, &eltwise_47_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_46_Mul_0_1_eltwise_47_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_46_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_46_Mul_0_1_eltwise_47_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_46_Mul_0_1_eltwise_47_conversion_layer, 46,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &reduce_46_Mul_0_1_eltwise_47_conversion_chain,
  NULL, &reduce_46_Mul_0_1_eltwise_47_conversion_0_1_eltwise_47_conversion_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_46_Mul_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_46_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_46_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &reduce_46_Mul_scale, &reduce_46_Mul_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_46_Mul_layer, 46,
  BN_TYPE, 0x0, NULL,
  bn, forward_bn,
  &reduce_46_Mul_chain,
  NULL, &reduce_46_Mul_0_1_eltwise_47_conversion_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_float reduce_46_neutral_value_data[] = { 0.0f };
AI_ARRAY_OBJ_DECLARE(
    reduce_46_neutral_value, AI_ARRAY_FORMAT_FLOAT,
    reduce_46_neutral_value_data, reduce_46_neutral_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_46_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_45_0_0_reduce_46_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_46_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_46_layer, 46,
  REDUCE_TYPE, 0x0, NULL,
  reduce, forward_reduce,
  &reduce_46_chain,
  NULL, &reduce_46_Mul_layer, AI_STATIC, 
  .operation = ai_sum, 
  .neutral_value = &reduce_46_neutral_value, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_45_0_0_reduce_46_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_45_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_45_0_0_reduce_46_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_45_0_0_reduce_46_conversion_layer, 45,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &eltwise_45_0_0_reduce_46_conversion_chain,
  NULL, &reduce_46_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_45_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_1_output, &eltwise_44_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_45_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_45_layer, 45,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_45_chain,
  NULL, &eltwise_45_0_0_reduce_46_conversion_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_44_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_42_output, &tiny_bert_generator_enc_0_mha_output_dense_BiasAdd_ReadVariableOp_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_44_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_44_layer, 44,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_44_chain,
  NULL, &eltwise_45_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_42_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_32_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_42_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &gemm_42_weights, &gemm_26_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_42_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_42_layer, 42,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &gemm_42_chain,
  NULL, &eltwise_44_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  transpose_32_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_31_0_0_transpose_32_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_32_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  transpose_32_layer, 32,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &transpose_32_chain,
  NULL, &gemm_42_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_CHANNEL, AI_SHAPE_HEIGHT, AI_SHAPE_WIDTH, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_31_0_0_transpose_32_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_31_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_31_0_0_transpose_32_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_31_0_0_transpose_32_conversion_layer, 31,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &gemm_31_0_0_transpose_32_conversion_chain,
  NULL, &transpose_32_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_31_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &nl_25_0_0_gemm_31_conversion_output, &transpose_30_0_1_gemm_31_conversion_output, &gemm_31_bias_0_2_gemm_31_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_31_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_31_layer, 31,
  MATMUL_TYPE, 0x0, NULL,
  matmul, forward_matmul,
  &gemm_31_chain,
  NULL, &gemm_31_0_0_transpose_32_conversion_layer, AI_STATIC, 
  .alpha = 1.0, 
  .beta = 1.0, 
  .tA = 0, 
  .tB = 0, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_25_0_0_gemm_31_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_25_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_25_0_0_gemm_31_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_25_0_0_gemm_31_conversion_layer, 25,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &nl_25_0_0_gemm_31_conversion_chain,
  NULL, &gemm_31_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i32 nl_25_nl_params_data[] = { 1308412288, 28, -7 };
AI_ARRAY_OBJ_DECLARE(
    nl_25_nl_params, AI_ARRAY_FORMAT_S32,
    nl_25_nl_params_data, nl_25_nl_params_data, 3, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_25_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_24_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_25_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_25_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  nl_25_layer, 25,
  SM_TYPE, 0x0, NULL,
  sm, forward_sm_integer,
  &nl_25_chain,
  NULL, &nl_25_0_0_gemm_31_conversion_layer, AI_STATIC, 
  .nl_params = &nl_25_nl_params, 
  .axis = AI_SHAPE_CHANNEL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_24_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_23_0_0_eltwise_24_conversion_output, &tiny_bert_generator_enc_3_mha_truedivtiny_bert_generator_enc_3_mha_Sqrt_4D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_24_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_24_layer, 24,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_24_chain,
  NULL, &nl_25_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_23_0_0_eltwise_24_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_23_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_23_0_0_eltwise_24_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_23_0_0_eltwise_24_conversion_layer, 23,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &gemm_23_0_0_eltwise_24_conversion_chain,
  NULL, &eltwise_24_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_23_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &transpose_22_0_0_gemm_23_conversion_output, &transpose_bgemm_23_out_0_1_gemm_23_conversion_output, &gemm_23_bias_0_2_gemm_23_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_23_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_23_layer, 23,
  MATMUL_TYPE, 0x0, NULL,
  matmul, forward_matmul,
  &gemm_23_chain,
  NULL, &gemm_23_0_0_eltwise_24_conversion_layer, AI_STATIC, 
  .alpha = 1.0, 
  .beta = 1.0, 
  .tA = 0, 
  .tB = 0, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  transpose_bgemm_23_out_0_1_gemm_23_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_bgemm_23_out_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_bgemm_23_out_0_1_gemm_23_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  transpose_bgemm_23_out_0_1_gemm_23_conversion_layer, 23,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &transpose_bgemm_23_out_0_1_gemm_23_conversion_chain,
  NULL, &gemm_23_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  transpose_bgemm_23_out_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_15_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_bgemm_23_out_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  transpose_bgemm_23_out_layer, 23,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &transpose_bgemm_23_out_chain,
  NULL, &transpose_bgemm_23_out_0_1_gemm_23_conversion_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_HEIGHT, AI_SHAPE_CHANNEL, AI_SHAPE_WIDTH, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_15_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_13_output, &tiny_bert_generator_enc_0_mha_wk_BiasAdd_ReadVariableOp_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_15_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_15_layer, 15,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_15_chain,
  NULL, &transpose_bgemm_23_out_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_13_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_1_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_13_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &gemm_13_weights, &gemm_26_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_13_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_13_layer, 13,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &gemm_13_chain,
  NULL, &eltwise_15_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  transpose_22_0_0_gemm_23_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_22_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_22_0_0_gemm_23_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  transpose_22_0_0_gemm_23_conversion_layer, 22,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &transpose_22_0_0_gemm_23_conversion_chain,
  NULL, &gemm_13_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  transpose_22_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_20_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_22_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  transpose_22_layer, 22,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &transpose_22_chain,
  NULL, &transpose_22_0_0_gemm_23_conversion_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_CHANNEL, AI_SHAPE_HEIGHT, AI_SHAPE_WIDTH, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_20_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_18_output, &tiny_bert_generator_enc_0_mha_wq_BiasAdd_ReadVariableOp_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_20_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_20_layer, 20,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_20_chain,
  NULL, &transpose_22_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_18_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_1_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_18_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &gemm_18_weights, &gemm_26_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_18_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_18_layer, 18,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &gemm_18_chain,
  NULL, &eltwise_20_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  transpose_30_0_1_gemm_31_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_30_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_30_0_1_gemm_31_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  transpose_30_0_1_gemm_31_conversion_layer, 30,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &transpose_30_0_1_gemm_31_conversion_chain,
  NULL, &gemm_18_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  transpose_30_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_28_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_30_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  transpose_30_layer, 30,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &transpose_30_chain,
  NULL, &transpose_30_0_1_gemm_31_conversion_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_CHANNEL, AI_SHAPE_HEIGHT, AI_SHAPE_WIDTH, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_28_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_26_output, &tiny_bert_generator_enc_0_mha_wv_BiasAdd_ReadVariableOp_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_28_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_28_layer, 28,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_28_chain,
  NULL, &transpose_30_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_26_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_1_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_26_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &gemm_26_weights, &gemm_26_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_26_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_26_layer, 26,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &gemm_26_chain,
  NULL, &eltwise_28_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_1_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_0_output, &tiny_bert_generator_tf___operators___add_1_y),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_1_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_1_layer, 1,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_1_chain,
  NULL, &gemm_26_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &serving_default_embedded_tokens0_output, &tiny_bert_generator_tf_math_multiply_1_Mul_y_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_0_layer, 0,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_0_chain,
  NULL, &eltwise_1_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_23_bias_0_2_gemm_23_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_23_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_23_bias_0_2_gemm_23_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_23_bias_0_2_gemm_23_conversion_layer, 23,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &gemm_23_bias_0_2_gemm_23_conversion_chain,
  NULL, &eltwise_0_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_31_bias_0_2_gemm_31_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_31_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_31_bias_0_2_gemm_31_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_31_bias_0_2_gemm_31_conversion_layer, 31,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &gemm_31_bias_0_2_gemm_31_conversion_chain,
  NULL, &gemm_23_bias_0_2_gemm_23_conversion_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_110_bias_0_2_gemm_110_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_110_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_110_bias_0_2_gemm_110_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_110_bias_0_2_gemm_110_conversion_layer, 110,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &gemm_110_bias_0_2_gemm_110_conversion_chain,
  NULL, &gemm_31_bias_0_2_gemm_31_conversion_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_118_bias_0_2_gemm_118_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_118_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_118_bias_0_2_gemm_118_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_118_bias_0_2_gemm_118_conversion_layer, 118,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &gemm_118_bias_0_2_gemm_118_conversion_chain,
  NULL, &gemm_110_bias_0_2_gemm_110_conversion_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_197_bias_0_2_gemm_197_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_197_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_197_bias_0_2_gemm_197_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_197_bias_0_2_gemm_197_conversion_layer, 197,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &gemm_197_bias_0_2_gemm_197_conversion_chain,
  NULL, &gemm_118_bias_0_2_gemm_118_conversion_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_205_bias_0_2_gemm_205_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_205_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_205_bias_0_2_gemm_205_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_205_bias_0_2_gemm_205_conversion_layer, 205,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &gemm_205_bias_0_2_gemm_205_conversion_chain,
  NULL, &gemm_197_bias_0_2_gemm_197_conversion_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_284_bias_0_2_gemm_284_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_284_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_284_bias_0_2_gemm_284_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_284_bias_0_2_gemm_284_conversion_layer, 284,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &gemm_284_bias_0_2_gemm_284_conversion_chain,
  NULL, &gemm_205_bias_0_2_gemm_205_conversion_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_292_bias_0_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_292_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_292_bias_0_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_292_bias_0_conversion_layer, 292,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &gemm_292_bias_0_conversion_chain,
  NULL, &gemm_284_bias_0_2_gemm_284_conversion_layer, AI_STATIC, 
)


#if (AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 2154368, 1, 1),
    2154368, NULL, NULL),
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 545056, 1, 1),
    545056, NULL, NULL),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_IN_NUM, &serving_default_embedded_tokens0_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_OUT_NUM, &eltwise_360_output),
  &gemm_292_bias_0_conversion_layer, 0x17a8d7a1, NULL)

#else

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 2154368, 1, 1),
      2154368, NULL, NULL)
  ),
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 545056, 1, 1),
      545056, NULL, NULL)
  ),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_IN_NUM, &serving_default_embedded_tokens0_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_OUT_NUM, &eltwise_360_output),
  &gemm_292_bias_0_conversion_layer, 0x17a8d7a1, NULL)

#endif	/*(AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)*/



/******************************************************************************/
AI_DECLARE_STATIC
ai_bool network_configure_activations(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_activations_map(g_network_activations_map, 1, params)) {
    /* Updating activations (byte) offsets */
    
    serving_default_embedded_tokens0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    serving_default_embedded_tokens0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    gemm_292_bias_0_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 25600);
    gemm_292_bias_0_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 25600);
    gemm_284_bias_0_2_gemm_284_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 25604);
    gemm_284_bias_0_2_gemm_284_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 25604);
    gemm_205_bias_0_2_gemm_205_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 25608);
    gemm_205_bias_0_2_gemm_205_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 25608);
    gemm_197_bias_0_2_gemm_197_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 25612);
    gemm_197_bias_0_2_gemm_197_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 25612);
    gemm_118_bias_0_2_gemm_118_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 25616);
    gemm_118_bias_0_2_gemm_118_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 25616);
    gemm_110_bias_0_2_gemm_110_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 25620);
    gemm_110_bias_0_2_gemm_110_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 25620);
    gemm_31_bias_0_2_gemm_31_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 25624);
    gemm_31_bias_0_2_gemm_31_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 25624);
    gemm_23_bias_0_2_gemm_23_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 25628);
    gemm_23_bias_0_2_gemm_23_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 25628);
    eltwise_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    eltwise_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    eltwise_1_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    eltwise_1_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    gemm_26_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 25632);
    gemm_26_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 25632);
    gemm_26_output_array.data = AI_PTR(g_network_activations_map[0] + 26656);
    gemm_26_output_array.data_start = AI_PTR(g_network_activations_map[0] + 26656);
    eltwise_28_output_array.data = AI_PTR(g_network_activations_map[0] + 26656);
    eltwise_28_output_array.data_start = AI_PTR(g_network_activations_map[0] + 26656);
    transpose_30_output_array.data = AI_PTR(g_network_activations_map[0] + 52256);
    transpose_30_output_array.data_start = AI_PTR(g_network_activations_map[0] + 52256);
    transpose_30_0_1_gemm_31_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 77856);
    transpose_30_0_1_gemm_31_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 77856);
    gemm_18_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 25632);
    gemm_18_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 25632);
    gemm_18_output_array.data = AI_PTR(g_network_activations_map[0] + 26656);
    gemm_18_output_array.data_start = AI_PTR(g_network_activations_map[0] + 26656);
    eltwise_20_output_array.data = AI_PTR(g_network_activations_map[0] + 52256);
    eltwise_20_output_array.data_start = AI_PTR(g_network_activations_map[0] + 52256);
    transpose_22_output_array.data = AI_PTR(g_network_activations_map[0] + 25632);
    transpose_22_output_array.data_start = AI_PTR(g_network_activations_map[0] + 25632);
    transpose_22_0_0_gemm_23_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 180256);
    transpose_22_0_0_gemm_23_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 180256);
    gemm_13_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 25632);
    gemm_13_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 25632);
    gemm_13_output_array.data = AI_PTR(g_network_activations_map[0] + 26656);
    gemm_13_output_array.data_start = AI_PTR(g_network_activations_map[0] + 26656);
    eltwise_15_output_array.data = AI_PTR(g_network_activations_map[0] + 52256);
    eltwise_15_output_array.data_start = AI_PTR(g_network_activations_map[0] + 52256);
    transpose_bgemm_23_out_output_array.data = AI_PTR(g_network_activations_map[0] + 25632);
    transpose_bgemm_23_out_output_array.data_start = AI_PTR(g_network_activations_map[0] + 25632);
    transpose_bgemm_23_out_0_1_gemm_23_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 282656);
    transpose_bgemm_23_out_0_1_gemm_23_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 282656);
    gemm_23_output_array.data = AI_PTR(g_network_activations_map[0] + 385056);
    gemm_23_output_array.data_start = AI_PTR(g_network_activations_map[0] + 385056);
    gemm_23_0_0_eltwise_24_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 25628);
    gemm_23_0_0_eltwise_24_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 25628);
    eltwise_24_output_array.data = AI_PTR(g_network_activations_map[0] + 180256);
    eltwise_24_output_array.data_start = AI_PTR(g_network_activations_map[0] + 180256);
    nl_25_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 25628);
    nl_25_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 25628);
    nl_25_output_array.data = AI_PTR(g_network_activations_map[0] + 25656);
    nl_25_output_array.data_start = AI_PTR(g_network_activations_map[0] + 25656);
    nl_25_0_0_gemm_31_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 180256);
    nl_25_0_0_gemm_31_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 180256);
    gemm_31_output_array.data = AI_PTR(g_network_activations_map[0] + 340256);
    gemm_31_output_array.data_start = AI_PTR(g_network_activations_map[0] + 340256);
    gemm_31_0_0_transpose_32_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 25624);
    gemm_31_0_0_transpose_32_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 25624);
    transpose_32_output_array.data = AI_PTR(g_network_activations_map[0] + 51224);
    transpose_32_output_array.data_start = AI_PTR(g_network_activations_map[0] + 51224);
    gemm_42_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 25624);
    gemm_42_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 25624);
    gemm_42_output_array.data = AI_PTR(g_network_activations_map[0] + 76824);
    gemm_42_output_array.data_start = AI_PTR(g_network_activations_map[0] + 76824);
    eltwise_44_output_array.data = AI_PTR(g_network_activations_map[0] + 25624);
    eltwise_44_output_array.data_start = AI_PTR(g_network_activations_map[0] + 25624);
    eltwise_45_output_array.data = AI_PTR(g_network_activations_map[0] + 51224);
    eltwise_45_output_array.data_start = AI_PTR(g_network_activations_map[0] + 51224);
    eltwise_45_0_0_reduce_46_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 76824);
    eltwise_45_0_0_reduce_46_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 76824);
    reduce_46_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    reduce_46_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    reduce_46_Mul_output_array.data = AI_PTR(g_network_activations_map[0] + 400);
    reduce_46_Mul_output_array.data_start = AI_PTR(g_network_activations_map[0] + 400);
    reduce_46_Mul_0_1_eltwise_47_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    reduce_46_Mul_0_1_eltwise_47_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    reduce_46_Mul_0_1_eltwise_47_conversion_0_1_eltwise_47_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 100);
    reduce_46_Mul_0_1_eltwise_47_conversion_0_1_eltwise_47_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 100);
    eltwise_47_output_array.data = AI_PTR(g_network_activations_map[0] + 179224);
    eltwise_47_output_array.data_start = AI_PTR(g_network_activations_map[0] + 179224);
    reduce_48_output_array.data = AI_PTR(g_network_activations_map[0] + 100);
    reduce_48_output_array.data_start = AI_PTR(g_network_activations_map[0] + 100);
    reduce_48_Mul_output_array.data = AI_PTR(g_network_activations_map[0] + 500);
    reduce_48_Mul_output_array.data_start = AI_PTR(g_network_activations_map[0] + 500);
    reduce_48_Mul_0_0_eltwise_49_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 100);
    reduce_48_Mul_0_0_eltwise_49_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 100);
    eltwise_49_output_array.data = AI_PTR(g_network_activations_map[0] + 200);
    eltwise_49_output_array.data_start = AI_PTR(g_network_activations_map[0] + 200);
    nl_50_output_array.data = AI_PTR(g_network_activations_map[0] + 100);
    nl_50_output_array.data_start = AI_PTR(g_network_activations_map[0] + 100);
    eltwise_51_output_array.data = AI_PTR(g_network_activations_map[0] + 25624);
    eltwise_51_output_array.data_start = AI_PTR(g_network_activations_map[0] + 25624);
    eltwise_53_output_array.data = AI_PTR(g_network_activations_map[0] + 76824);
    eltwise_53_output_array.data_start = AI_PTR(g_network_activations_map[0] + 76824);
    eltwise_54_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    eltwise_54_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    eltwise_52_output_array.data = AI_PTR(g_network_activations_map[0] + 76824);
    eltwise_52_output_array.data_start = AI_PTR(g_network_activations_map[0] + 76824);
    eltwise_55_output_array.data = AI_PTR(g_network_activations_map[0] + 25624);
    eltwise_55_output_array.data_start = AI_PTR(g_network_activations_map[0] + 25624);
    gemm_64_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    gemm_64_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    gemm_64_output_array.data = AI_PTR(g_network_activations_map[0] + 51224);
    gemm_64_output_array.data_start = AI_PTR(g_network_activations_map[0] + 51224);
    eltwise_66_output_array.data = AI_PTR(g_network_activations_map[0] + 102424);
    eltwise_66_output_array.data_start = AI_PTR(g_network_activations_map[0] + 102424);
    gemm_75_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    gemm_75_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    gemm_75_output_array.data = AI_PTR(g_network_activations_map[0] + 51224);
    gemm_75_output_array.data_start = AI_PTR(g_network_activations_map[0] + 51224);
    eltwise_77_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    eltwise_77_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    eltwise_78_output_array.data = AI_PTR(g_network_activations_map[0] + 51224);
    eltwise_78_output_array.data_start = AI_PTR(g_network_activations_map[0] + 51224);
    eltwise_78_0_0_reduce_79_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 76824);
    eltwise_78_0_0_reduce_79_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 76824);
    reduce_79_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    reduce_79_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    reduce_79_Mul_output_array.data = AI_PTR(g_network_activations_map[0] + 400);
    reduce_79_Mul_output_array.data_start = AI_PTR(g_network_activations_map[0] + 400);
    reduce_79_Mul_0_1_eltwise_80_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    reduce_79_Mul_0_1_eltwise_80_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    reduce_79_Mul_0_1_eltwise_80_conversion_0_1_eltwise_80_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 100);
    reduce_79_Mul_0_1_eltwise_80_conversion_0_1_eltwise_80_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 100);
    eltwise_80_output_array.data = AI_PTR(g_network_activations_map[0] + 179224);
    eltwise_80_output_array.data_start = AI_PTR(g_network_activations_map[0] + 179224);
    reduce_81_output_array.data = AI_PTR(g_network_activations_map[0] + 100);
    reduce_81_output_array.data_start = AI_PTR(g_network_activations_map[0] + 100);
    reduce_81_Mul_output_array.data = AI_PTR(g_network_activations_map[0] + 500);
    reduce_81_Mul_output_array.data_start = AI_PTR(g_network_activations_map[0] + 500);
    reduce_81_Mul_0_0_eltwise_82_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 100);
    reduce_81_Mul_0_0_eltwise_82_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 100);
    eltwise_82_output_array.data = AI_PTR(g_network_activations_map[0] + 200);
    eltwise_82_output_array.data_start = AI_PTR(g_network_activations_map[0] + 200);
    nl_83_output_array.data = AI_PTR(g_network_activations_map[0] + 100);
    nl_83_output_array.data_start = AI_PTR(g_network_activations_map[0] + 100);
    eltwise_84_output_array.data = AI_PTR(g_network_activations_map[0] + 25624);
    eltwise_84_output_array.data_start = AI_PTR(g_network_activations_map[0] + 25624);
    eltwise_86_output_array.data = AI_PTR(g_network_activations_map[0] + 76824);
    eltwise_86_output_array.data_start = AI_PTR(g_network_activations_map[0] + 76824);
    eltwise_87_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    eltwise_87_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    eltwise_85_output_array.data = AI_PTR(g_network_activations_map[0] + 76824);
    eltwise_85_output_array.data_start = AI_PTR(g_network_activations_map[0] + 76824);
    eltwise_88_output_array.data = AI_PTR(g_network_activations_map[0] + 25624);
    eltwise_88_output_array.data_start = AI_PTR(g_network_activations_map[0] + 25624);
    gemm_113_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    gemm_113_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    gemm_113_output_array.data = AI_PTR(g_network_activations_map[0] + 51224);
    gemm_113_output_array.data_start = AI_PTR(g_network_activations_map[0] + 51224);
    eltwise_115_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    eltwise_115_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    transpose_117_output_array.data = AI_PTR(g_network_activations_map[0] + 51224);
    transpose_117_output_array.data_start = AI_PTR(g_network_activations_map[0] + 51224);
    transpose_117_0_1_gemm_118_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 76824);
    transpose_117_0_1_gemm_118_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 76824);
    gemm_105_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    gemm_105_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    gemm_105_output_array.data = AI_PTR(g_network_activations_map[0] + 51224);
    gemm_105_output_array.data_start = AI_PTR(g_network_activations_map[0] + 51224);
    eltwise_107_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    eltwise_107_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    transpose_109_output_array.data = AI_PTR(g_network_activations_map[0] + 51224);
    transpose_109_output_array.data_start = AI_PTR(g_network_activations_map[0] + 51224);
    transpose_109_0_0_gemm_110_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 179224);
    transpose_109_0_0_gemm_110_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 179224);
    gemm_100_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    gemm_100_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    gemm_100_output_array.data = AI_PTR(g_network_activations_map[0] + 51224);
    gemm_100_output_array.data_start = AI_PTR(g_network_activations_map[0] + 51224);
    eltwise_102_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    eltwise_102_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    transpose_bgemm_110_out_output_array.data = AI_PTR(g_network_activations_map[0] + 51224);
    transpose_bgemm_110_out_output_array.data_start = AI_PTR(g_network_activations_map[0] + 51224);
    transpose_bgemm_110_out_0_1_gemm_110_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 281624);
    transpose_bgemm_110_out_0_1_gemm_110_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 281624);
    gemm_110_output_array.data = AI_PTR(g_network_activations_map[0] + 384024);
    gemm_110_output_array.data_start = AI_PTR(g_network_activations_map[0] + 384024);
    gemm_110_0_0_eltwise_111_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 179224);
    gemm_110_0_0_eltwise_111_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 179224);
    eltwise_111_output_array.data = AI_PTR(g_network_activations_map[0] + 219224);
    eltwise_111_output_array.data_start = AI_PTR(g_network_activations_map[0] + 219224);
    nl_112_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    nl_112_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    nl_112_output_array.data = AI_PTR(g_network_activations_map[0] + 179224);
    nl_112_output_array.data_start = AI_PTR(g_network_activations_map[0] + 179224);
    nl_112_0_0_gemm_118_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 219224);
    nl_112_0_0_gemm_118_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 219224);
    gemm_118_output_array.data = AI_PTR(g_network_activations_map[0] + 379224);
    gemm_118_output_array.data_start = AI_PTR(g_network_activations_map[0] + 379224);
    gemm_118_0_0_transpose_119_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    gemm_118_0_0_transpose_119_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    transpose_119_output_array.data = AI_PTR(g_network_activations_map[0] + 51224);
    transpose_119_output_array.data_start = AI_PTR(g_network_activations_map[0] + 51224);
    gemm_129_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    gemm_129_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    gemm_129_output_array.data = AI_PTR(g_network_activations_map[0] + 76824);
    gemm_129_output_array.data_start = AI_PTR(g_network_activations_map[0] + 76824);
    eltwise_131_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    eltwise_131_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    eltwise_132_output_array.data = AI_PTR(g_network_activations_map[0] + 51224);
    eltwise_132_output_array.data_start = AI_PTR(g_network_activations_map[0] + 51224);
    eltwise_132_0_0_reduce_133_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 76824);
    eltwise_132_0_0_reduce_133_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 76824);
    reduce_133_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    reduce_133_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    reduce_133_Mul_output_array.data = AI_PTR(g_network_activations_map[0] + 400);
    reduce_133_Mul_output_array.data_start = AI_PTR(g_network_activations_map[0] + 400);
    reduce_133_Mul_0_1_eltwise_134_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    reduce_133_Mul_0_1_eltwise_134_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    reduce_133_Mul_0_1_eltwise_134_conversion_0_1_eltwise_134_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 100);
    reduce_133_Mul_0_1_eltwise_134_conversion_0_1_eltwise_134_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 100);
    eltwise_134_output_array.data = AI_PTR(g_network_activations_map[0] + 179224);
    eltwise_134_output_array.data_start = AI_PTR(g_network_activations_map[0] + 179224);
    reduce_135_output_array.data = AI_PTR(g_network_activations_map[0] + 100);
    reduce_135_output_array.data_start = AI_PTR(g_network_activations_map[0] + 100);
    reduce_135_Mul_output_array.data = AI_PTR(g_network_activations_map[0] + 500);
    reduce_135_Mul_output_array.data_start = AI_PTR(g_network_activations_map[0] + 500);
    reduce_135_Mul_0_0_eltwise_136_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 100);
    reduce_135_Mul_0_0_eltwise_136_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 100);
    eltwise_136_output_array.data = AI_PTR(g_network_activations_map[0] + 200);
    eltwise_136_output_array.data_start = AI_PTR(g_network_activations_map[0] + 200);
    nl_137_output_array.data = AI_PTR(g_network_activations_map[0] + 100);
    nl_137_output_array.data_start = AI_PTR(g_network_activations_map[0] + 100);
    eltwise_138_output_array.data = AI_PTR(g_network_activations_map[0] + 25616);
    eltwise_138_output_array.data_start = AI_PTR(g_network_activations_map[0] + 25616);
    eltwise_140_output_array.data = AI_PTR(g_network_activations_map[0] + 76824);
    eltwise_140_output_array.data_start = AI_PTR(g_network_activations_map[0] + 76824);
    eltwise_141_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    eltwise_141_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    eltwise_139_output_array.data = AI_PTR(g_network_activations_map[0] + 76824);
    eltwise_139_output_array.data_start = AI_PTR(g_network_activations_map[0] + 76824);
    eltwise_142_output_array.data = AI_PTR(g_network_activations_map[0] + 25616);
    eltwise_142_output_array.data_start = AI_PTR(g_network_activations_map[0] + 25616);
    gemm_151_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    gemm_151_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    gemm_151_output_array.data = AI_PTR(g_network_activations_map[0] + 51216);
    gemm_151_output_array.data_start = AI_PTR(g_network_activations_map[0] + 51216);
    eltwise_153_output_array.data = AI_PTR(g_network_activations_map[0] + 102416);
    eltwise_153_output_array.data_start = AI_PTR(g_network_activations_map[0] + 102416);
    gemm_162_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    gemm_162_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    gemm_162_output_array.data = AI_PTR(g_network_activations_map[0] + 51216);
    gemm_162_output_array.data_start = AI_PTR(g_network_activations_map[0] + 51216);
    eltwise_164_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    eltwise_164_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    eltwise_165_output_array.data = AI_PTR(g_network_activations_map[0] + 51216);
    eltwise_165_output_array.data_start = AI_PTR(g_network_activations_map[0] + 51216);
    eltwise_165_0_0_reduce_166_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 76816);
    eltwise_165_0_0_reduce_166_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 76816);
    reduce_166_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    reduce_166_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    reduce_166_Mul_output_array.data = AI_PTR(g_network_activations_map[0] + 400);
    reduce_166_Mul_output_array.data_start = AI_PTR(g_network_activations_map[0] + 400);
    reduce_166_Mul_0_1_eltwise_167_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    reduce_166_Mul_0_1_eltwise_167_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    reduce_166_Mul_0_1_eltwise_167_conversion_0_1_eltwise_167_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 100);
    reduce_166_Mul_0_1_eltwise_167_conversion_0_1_eltwise_167_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 100);
    eltwise_167_output_array.data = AI_PTR(g_network_activations_map[0] + 179216);
    eltwise_167_output_array.data_start = AI_PTR(g_network_activations_map[0] + 179216);
    reduce_168_output_array.data = AI_PTR(g_network_activations_map[0] + 100);
    reduce_168_output_array.data_start = AI_PTR(g_network_activations_map[0] + 100);
    reduce_168_Mul_output_array.data = AI_PTR(g_network_activations_map[0] + 500);
    reduce_168_Mul_output_array.data_start = AI_PTR(g_network_activations_map[0] + 500);
    reduce_168_Mul_0_0_eltwise_169_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 100);
    reduce_168_Mul_0_0_eltwise_169_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 100);
    eltwise_169_output_array.data = AI_PTR(g_network_activations_map[0] + 200);
    eltwise_169_output_array.data_start = AI_PTR(g_network_activations_map[0] + 200);
    nl_170_output_array.data = AI_PTR(g_network_activations_map[0] + 100);
    nl_170_output_array.data_start = AI_PTR(g_network_activations_map[0] + 100);
    eltwise_171_output_array.data = AI_PTR(g_network_activations_map[0] + 25616);
    eltwise_171_output_array.data_start = AI_PTR(g_network_activations_map[0] + 25616);
    eltwise_173_output_array.data = AI_PTR(g_network_activations_map[0] + 76816);
    eltwise_173_output_array.data_start = AI_PTR(g_network_activations_map[0] + 76816);
    eltwise_174_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    eltwise_174_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    eltwise_172_output_array.data = AI_PTR(g_network_activations_map[0] + 76816);
    eltwise_172_output_array.data_start = AI_PTR(g_network_activations_map[0] + 76816);
    eltwise_175_output_array.data = AI_PTR(g_network_activations_map[0] + 25616);
    eltwise_175_output_array.data_start = AI_PTR(g_network_activations_map[0] + 25616);
    gemm_200_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    gemm_200_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    gemm_200_output_array.data = AI_PTR(g_network_activations_map[0] + 51216);
    gemm_200_output_array.data_start = AI_PTR(g_network_activations_map[0] + 51216);
    eltwise_202_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    eltwise_202_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    transpose_204_output_array.data = AI_PTR(g_network_activations_map[0] + 51216);
    transpose_204_output_array.data_start = AI_PTR(g_network_activations_map[0] + 51216);
    transpose_204_0_1_gemm_205_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 76816);
    transpose_204_0_1_gemm_205_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 76816);
    gemm_192_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    gemm_192_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    gemm_192_output_array.data = AI_PTR(g_network_activations_map[0] + 51216);
    gemm_192_output_array.data_start = AI_PTR(g_network_activations_map[0] + 51216);
    eltwise_194_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    eltwise_194_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    transpose_196_output_array.data = AI_PTR(g_network_activations_map[0] + 51216);
    transpose_196_output_array.data_start = AI_PTR(g_network_activations_map[0] + 51216);
    transpose_196_0_0_gemm_197_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 179216);
    transpose_196_0_0_gemm_197_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 179216);
    gemm_187_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    gemm_187_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    gemm_187_output_array.data = AI_PTR(g_network_activations_map[0] + 51216);
    gemm_187_output_array.data_start = AI_PTR(g_network_activations_map[0] + 51216);
    eltwise_189_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    eltwise_189_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    transpose_bgemm_197_out_output_array.data = AI_PTR(g_network_activations_map[0] + 51216);
    transpose_bgemm_197_out_output_array.data_start = AI_PTR(g_network_activations_map[0] + 51216);
    transpose_bgemm_197_out_0_1_gemm_197_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 281616);
    transpose_bgemm_197_out_0_1_gemm_197_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 281616);
    gemm_197_output_array.data = AI_PTR(g_network_activations_map[0] + 384016);
    gemm_197_output_array.data_start = AI_PTR(g_network_activations_map[0] + 384016);
    gemm_197_0_0_eltwise_198_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 179216);
    gemm_197_0_0_eltwise_198_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 179216);
    eltwise_198_output_array.data = AI_PTR(g_network_activations_map[0] + 219216);
    eltwise_198_output_array.data_start = AI_PTR(g_network_activations_map[0] + 219216);
    nl_199_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    nl_199_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    nl_199_output_array.data = AI_PTR(g_network_activations_map[0] + 179216);
    nl_199_output_array.data_start = AI_PTR(g_network_activations_map[0] + 179216);
    nl_199_0_0_gemm_205_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 219216);
    nl_199_0_0_gemm_205_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 219216);
    gemm_205_output_array.data = AI_PTR(g_network_activations_map[0] + 379216);
    gemm_205_output_array.data_start = AI_PTR(g_network_activations_map[0] + 379216);
    gemm_205_0_0_transpose_206_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    gemm_205_0_0_transpose_206_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    transpose_206_output_array.data = AI_PTR(g_network_activations_map[0] + 51216);
    transpose_206_output_array.data_start = AI_PTR(g_network_activations_map[0] + 51216);
    gemm_216_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    gemm_216_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    gemm_216_output_array.data = AI_PTR(g_network_activations_map[0] + 76816);
    gemm_216_output_array.data_start = AI_PTR(g_network_activations_map[0] + 76816);
    eltwise_218_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    eltwise_218_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    eltwise_219_output_array.data = AI_PTR(g_network_activations_map[0] + 51216);
    eltwise_219_output_array.data_start = AI_PTR(g_network_activations_map[0] + 51216);
    eltwise_219_0_0_reduce_220_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 76816);
    eltwise_219_0_0_reduce_220_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 76816);
    reduce_220_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    reduce_220_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    reduce_220_Mul_output_array.data = AI_PTR(g_network_activations_map[0] + 400);
    reduce_220_Mul_output_array.data_start = AI_PTR(g_network_activations_map[0] + 400);
    reduce_220_Mul_0_1_eltwise_221_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    reduce_220_Mul_0_1_eltwise_221_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    reduce_220_Mul_0_1_eltwise_221_conversion_0_1_eltwise_221_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 100);
    reduce_220_Mul_0_1_eltwise_221_conversion_0_1_eltwise_221_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 100);
    eltwise_221_output_array.data = AI_PTR(g_network_activations_map[0] + 179216);
    eltwise_221_output_array.data_start = AI_PTR(g_network_activations_map[0] + 179216);
    reduce_222_output_array.data = AI_PTR(g_network_activations_map[0] + 100);
    reduce_222_output_array.data_start = AI_PTR(g_network_activations_map[0] + 100);
    reduce_222_Mul_output_array.data = AI_PTR(g_network_activations_map[0] + 500);
    reduce_222_Mul_output_array.data_start = AI_PTR(g_network_activations_map[0] + 500);
    reduce_222_Mul_0_0_eltwise_223_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 100);
    reduce_222_Mul_0_0_eltwise_223_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 100);
    eltwise_223_output_array.data = AI_PTR(g_network_activations_map[0] + 200);
    eltwise_223_output_array.data_start = AI_PTR(g_network_activations_map[0] + 200);
    nl_224_output_array.data = AI_PTR(g_network_activations_map[0] + 100);
    nl_224_output_array.data_start = AI_PTR(g_network_activations_map[0] + 100);
    eltwise_225_output_array.data = AI_PTR(g_network_activations_map[0] + 25608);
    eltwise_225_output_array.data_start = AI_PTR(g_network_activations_map[0] + 25608);
    eltwise_227_output_array.data = AI_PTR(g_network_activations_map[0] + 76816);
    eltwise_227_output_array.data_start = AI_PTR(g_network_activations_map[0] + 76816);
    eltwise_228_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    eltwise_228_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    eltwise_226_output_array.data = AI_PTR(g_network_activations_map[0] + 76816);
    eltwise_226_output_array.data_start = AI_PTR(g_network_activations_map[0] + 76816);
    eltwise_229_output_array.data = AI_PTR(g_network_activations_map[0] + 25608);
    eltwise_229_output_array.data_start = AI_PTR(g_network_activations_map[0] + 25608);
    gemm_238_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    gemm_238_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    gemm_238_output_array.data = AI_PTR(g_network_activations_map[0] + 51208);
    gemm_238_output_array.data_start = AI_PTR(g_network_activations_map[0] + 51208);
    eltwise_240_output_array.data = AI_PTR(g_network_activations_map[0] + 102408);
    eltwise_240_output_array.data_start = AI_PTR(g_network_activations_map[0] + 102408);
    gemm_249_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    gemm_249_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    gemm_249_output_array.data = AI_PTR(g_network_activations_map[0] + 51208);
    gemm_249_output_array.data_start = AI_PTR(g_network_activations_map[0] + 51208);
    eltwise_251_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    eltwise_251_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    eltwise_252_output_array.data = AI_PTR(g_network_activations_map[0] + 51208);
    eltwise_252_output_array.data_start = AI_PTR(g_network_activations_map[0] + 51208);
    eltwise_252_0_0_reduce_253_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 76808);
    eltwise_252_0_0_reduce_253_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 76808);
    reduce_253_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    reduce_253_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    reduce_253_Mul_output_array.data = AI_PTR(g_network_activations_map[0] + 400);
    reduce_253_Mul_output_array.data_start = AI_PTR(g_network_activations_map[0] + 400);
    reduce_253_Mul_0_1_eltwise_254_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    reduce_253_Mul_0_1_eltwise_254_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    reduce_253_Mul_0_1_eltwise_254_conversion_0_1_eltwise_254_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 100);
    reduce_253_Mul_0_1_eltwise_254_conversion_0_1_eltwise_254_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 100);
    eltwise_254_output_array.data = AI_PTR(g_network_activations_map[0] + 179208);
    eltwise_254_output_array.data_start = AI_PTR(g_network_activations_map[0] + 179208);
    reduce_255_output_array.data = AI_PTR(g_network_activations_map[0] + 100);
    reduce_255_output_array.data_start = AI_PTR(g_network_activations_map[0] + 100);
    reduce_255_Mul_output_array.data = AI_PTR(g_network_activations_map[0] + 500);
    reduce_255_Mul_output_array.data_start = AI_PTR(g_network_activations_map[0] + 500);
    reduce_255_Mul_0_0_eltwise_256_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 100);
    reduce_255_Mul_0_0_eltwise_256_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 100);
    eltwise_256_output_array.data = AI_PTR(g_network_activations_map[0] + 200);
    eltwise_256_output_array.data_start = AI_PTR(g_network_activations_map[0] + 200);
    nl_257_output_array.data = AI_PTR(g_network_activations_map[0] + 100);
    nl_257_output_array.data_start = AI_PTR(g_network_activations_map[0] + 100);
    eltwise_258_output_array.data = AI_PTR(g_network_activations_map[0] + 25608);
    eltwise_258_output_array.data_start = AI_PTR(g_network_activations_map[0] + 25608);
    eltwise_260_output_array.data = AI_PTR(g_network_activations_map[0] + 76808);
    eltwise_260_output_array.data_start = AI_PTR(g_network_activations_map[0] + 76808);
    eltwise_261_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    eltwise_261_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    eltwise_259_output_array.data = AI_PTR(g_network_activations_map[0] + 76808);
    eltwise_259_output_array.data_start = AI_PTR(g_network_activations_map[0] + 76808);
    eltwise_262_output_array.data = AI_PTR(g_network_activations_map[0] + 25608);
    eltwise_262_output_array.data_start = AI_PTR(g_network_activations_map[0] + 25608);
    gemm_287_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    gemm_287_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    gemm_287_output_array.data = AI_PTR(g_network_activations_map[0] + 51208);
    gemm_287_output_array.data_start = AI_PTR(g_network_activations_map[0] + 51208);
    eltwise_289_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    eltwise_289_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    transpose_291_output_array.data = AI_PTR(g_network_activations_map[0] + 51208);
    transpose_291_output_array.data_start = AI_PTR(g_network_activations_map[0] + 51208);
    transpose_291_0_1_gemm_292_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 76808);
    transpose_291_0_1_gemm_292_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 76808);
    gemm_279_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    gemm_279_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    gemm_279_output_array.data = AI_PTR(g_network_activations_map[0] + 51208);
    gemm_279_output_array.data_start = AI_PTR(g_network_activations_map[0] + 51208);
    eltwise_281_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    eltwise_281_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    transpose_283_output_array.data = AI_PTR(g_network_activations_map[0] + 51208);
    transpose_283_output_array.data_start = AI_PTR(g_network_activations_map[0] + 51208);
    transpose_283_0_0_gemm_284_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 179208);
    transpose_283_0_0_gemm_284_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 179208);
    gemm_274_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    gemm_274_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    gemm_274_output_array.data = AI_PTR(g_network_activations_map[0] + 51208);
    gemm_274_output_array.data_start = AI_PTR(g_network_activations_map[0] + 51208);
    eltwise_276_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    eltwise_276_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    transpose_bgemm_284_out_output_array.data = AI_PTR(g_network_activations_map[0] + 51208);
    transpose_bgemm_284_out_output_array.data_start = AI_PTR(g_network_activations_map[0] + 51208);
    transpose_bgemm_284_out_0_1_gemm_284_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 281608);
    transpose_bgemm_284_out_0_1_gemm_284_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 281608);
    gemm_284_output_array.data = AI_PTR(g_network_activations_map[0] + 384008);
    gemm_284_output_array.data_start = AI_PTR(g_network_activations_map[0] + 384008);
    gemm_284_0_0_eltwise_285_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 179208);
    gemm_284_0_0_eltwise_285_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 179208);
    eltwise_285_output_array.data = AI_PTR(g_network_activations_map[0] + 219208);
    eltwise_285_output_array.data_start = AI_PTR(g_network_activations_map[0] + 219208);
    nl_286_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    nl_286_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    nl_286_output_array.data = AI_PTR(g_network_activations_map[0] + 179208);
    nl_286_output_array.data_start = AI_PTR(g_network_activations_map[0] + 179208);
    nl_286_0_0_gemm_292_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 219208);
    nl_286_0_0_gemm_292_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 219208);
    gemm_292_output_array.data = AI_PTR(g_network_activations_map[0] + 379208);
    gemm_292_output_array.data_start = AI_PTR(g_network_activations_map[0] + 379208);
    gemm_292_0_0_transpose_293_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    gemm_292_0_0_transpose_293_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    transpose_293_output_array.data = AI_PTR(g_network_activations_map[0] + 51208);
    transpose_293_output_array.data_start = AI_PTR(g_network_activations_map[0] + 51208);
    gemm_303_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    gemm_303_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    gemm_303_output_array.data = AI_PTR(g_network_activations_map[0] + 76808);
    gemm_303_output_array.data_start = AI_PTR(g_network_activations_map[0] + 76808);
    eltwise_305_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    eltwise_305_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    eltwise_306_output_array.data = AI_PTR(g_network_activations_map[0] + 51208);
    eltwise_306_output_array.data_start = AI_PTR(g_network_activations_map[0] + 51208);
    eltwise_306_0_0_reduce_307_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 76808);
    eltwise_306_0_0_reduce_307_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 76808);
    reduce_307_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    reduce_307_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    reduce_307_Mul_output_array.data = AI_PTR(g_network_activations_map[0] + 400);
    reduce_307_Mul_output_array.data_start = AI_PTR(g_network_activations_map[0] + 400);
    reduce_307_Mul_0_1_eltwise_308_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    reduce_307_Mul_0_1_eltwise_308_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    reduce_307_Mul_0_1_eltwise_308_conversion_0_1_eltwise_308_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 100);
    reduce_307_Mul_0_1_eltwise_308_conversion_0_1_eltwise_308_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 100);
    eltwise_308_output_array.data = AI_PTR(g_network_activations_map[0] + 179208);
    eltwise_308_output_array.data_start = AI_PTR(g_network_activations_map[0] + 179208);
    reduce_309_output_array.data = AI_PTR(g_network_activations_map[0] + 100);
    reduce_309_output_array.data_start = AI_PTR(g_network_activations_map[0] + 100);
    reduce_309_Mul_output_array.data = AI_PTR(g_network_activations_map[0] + 500);
    reduce_309_Mul_output_array.data_start = AI_PTR(g_network_activations_map[0] + 500);
    reduce_309_Mul_0_0_eltwise_310_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 100);
    reduce_309_Mul_0_0_eltwise_310_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 100);
    eltwise_310_output_array.data = AI_PTR(g_network_activations_map[0] + 200);
    eltwise_310_output_array.data_start = AI_PTR(g_network_activations_map[0] + 200);
    nl_311_output_array.data = AI_PTR(g_network_activations_map[0] + 100);
    nl_311_output_array.data_start = AI_PTR(g_network_activations_map[0] + 100);
    eltwise_312_output_array.data = AI_PTR(g_network_activations_map[0] + 200);
    eltwise_312_output_array.data_start = AI_PTR(g_network_activations_map[0] + 200);
    eltwise_314_output_array.data = AI_PTR(g_network_activations_map[0] + 76808);
    eltwise_314_output_array.data_start = AI_PTR(g_network_activations_map[0] + 76808);
    eltwise_315_output_array.data = AI_PTR(g_network_activations_map[0] + 102408);
    eltwise_315_output_array.data_start = AI_PTR(g_network_activations_map[0] + 102408);
    eltwise_313_output_array.data = AI_PTR(g_network_activations_map[0] + 76808);
    eltwise_313_output_array.data_start = AI_PTR(g_network_activations_map[0] + 76808);
    eltwise_316_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    eltwise_316_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    gemm_325_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 25600);
    gemm_325_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 25600);
    gemm_325_output_array.data = AI_PTR(g_network_activations_map[0] + 26624);
    gemm_325_output_array.data_start = AI_PTR(g_network_activations_map[0] + 26624);
    eltwise_327_output_array.data = AI_PTR(g_network_activations_map[0] + 77824);
    eltwise_327_output_array.data_start = AI_PTR(g_network_activations_map[0] + 77824);
    gemm_336_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 25600);
    gemm_336_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 25600);
    gemm_336_output_array.data = AI_PTR(g_network_activations_map[0] + 27648);
    gemm_336_output_array.data_start = AI_PTR(g_network_activations_map[0] + 27648);
    eltwise_338_output_array.data = AI_PTR(g_network_activations_map[0] + 53248);
    eltwise_338_output_array.data_start = AI_PTR(g_network_activations_map[0] + 53248);
    eltwise_339_output_array.data = AI_PTR(g_network_activations_map[0] + 25600);
    eltwise_339_output_array.data_start = AI_PTR(g_network_activations_map[0] + 25600);
    eltwise_339_0_0_reduce_340_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 51200);
    eltwise_339_0_0_reduce_340_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 51200);
    reduce_340_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    reduce_340_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    reduce_340_Mul_output_array.data = AI_PTR(g_network_activations_map[0] + 400);
    reduce_340_Mul_output_array.data_start = AI_PTR(g_network_activations_map[0] + 400);
    reduce_340_Mul_0_1_eltwise_341_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    reduce_340_Mul_0_1_eltwise_341_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    reduce_340_Mul_0_1_eltwise_341_conversion_0_1_eltwise_341_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 100);
    reduce_340_Mul_0_1_eltwise_341_conversion_0_1_eltwise_341_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 100);
    eltwise_341_output_array.data = AI_PTR(g_network_activations_map[0] + 153600);
    eltwise_341_output_array.data_start = AI_PTR(g_network_activations_map[0] + 153600);
    reduce_342_output_array.data = AI_PTR(g_network_activations_map[0] + 100);
    reduce_342_output_array.data_start = AI_PTR(g_network_activations_map[0] + 100);
    reduce_342_Mul_output_array.data = AI_PTR(g_network_activations_map[0] + 500);
    reduce_342_Mul_output_array.data_start = AI_PTR(g_network_activations_map[0] + 500);
    reduce_342_Mul_0_0_eltwise_343_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 100);
    reduce_342_Mul_0_0_eltwise_343_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 100);
    eltwise_343_output_array.data = AI_PTR(g_network_activations_map[0] + 200);
    eltwise_343_output_array.data_start = AI_PTR(g_network_activations_map[0] + 200);
    nl_344_output_array.data = AI_PTR(g_network_activations_map[0] + 100);
    nl_344_output_array.data_start = AI_PTR(g_network_activations_map[0] + 100);
    eltwise_345_output_array.data = AI_PTR(g_network_activations_map[0] + 51200);
    eltwise_345_output_array.data_start = AI_PTR(g_network_activations_map[0] + 51200);
    eltwise_347_output_array.data = AI_PTR(g_network_activations_map[0] + 76800);
    eltwise_347_output_array.data_start = AI_PTR(g_network_activations_map[0] + 76800);
    eltwise_348_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    eltwise_348_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    eltwise_346_output_array.data = AI_PTR(g_network_activations_map[0] + 76800);
    eltwise_346_output_array.data_start = AI_PTR(g_network_activations_map[0] + 76800);
    eltwise_349_output_array.data = AI_PTR(g_network_activations_map[0] + 25600);
    eltwise_349_output_array.data_start = AI_PTR(g_network_activations_map[0] + 25600);
    gemm_358_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    gemm_358_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    gemm_358_output_array.data = AI_PTR(g_network_activations_map[0] + 1024);
    gemm_358_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1024);
    eltwise_360_output_array.data = AI_PTR(g_network_activations_map[0] + 7624);
    eltwise_360_output_array.data_start = AI_PTR(g_network_activations_map[0] + 7624);
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_ACTIVATIONS);
  return false;
}




/******************************************************************************/
AI_DECLARE_STATIC
ai_bool network_configure_weights(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_weights_map(g_network_weights_map, 1, params)) {
    /* Updating weights (byte) offsets */
    
    gemm_292_bias_array.format |= AI_FMT_FLAG_CONST;
    gemm_292_bias_array.data = AI_PTR(g_network_weights_map[0] + 0);
    gemm_292_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 0);
    gemm_284_bias_array.format |= AI_FMT_FLAG_CONST;
    gemm_284_bias_array.data = AI_PTR(g_network_weights_map[0] + 4);
    gemm_284_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 4);
    gemm_205_bias_array.format |= AI_FMT_FLAG_CONST;
    gemm_205_bias_array.data = AI_PTR(g_network_weights_map[0] + 8);
    gemm_205_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 8);
    gemm_197_bias_array.format |= AI_FMT_FLAG_CONST;
    gemm_197_bias_array.data = AI_PTR(g_network_weights_map[0] + 12);
    gemm_197_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 12);
    gemm_118_bias_array.format |= AI_FMT_FLAG_CONST;
    gemm_118_bias_array.data = AI_PTR(g_network_weights_map[0] + 16);
    gemm_118_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 16);
    gemm_110_bias_array.format |= AI_FMT_FLAG_CONST;
    gemm_110_bias_array.data = AI_PTR(g_network_weights_map[0] + 20);
    gemm_110_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 20);
    gemm_31_bias_array.format |= AI_FMT_FLAG_CONST;
    gemm_31_bias_array.data = AI_PTR(g_network_weights_map[0] + 24);
    gemm_31_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 24);
    gemm_23_bias_array.format |= AI_FMT_FLAG_CONST;
    gemm_23_bias_array.data = AI_PTR(g_network_weights_map[0] + 28);
    gemm_23_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 28);
    tiny_bert_generator_tf_math_multiply_1_Mul_y_3D_array.format |= AI_FMT_FLAG_CONST;
    tiny_bert_generator_tf_math_multiply_1_Mul_y_3D_array.data = AI_PTR(g_network_weights_map[0] + 32);
    tiny_bert_generator_tf_math_multiply_1_Mul_y_3D_array.data_start = AI_PTR(g_network_weights_map[0] + 32);
    tiny_bert_generator_enc_0_mha_wk_BiasAdd_ReadVariableOp_3D_array.format |= AI_FMT_FLAG_CONST;
    tiny_bert_generator_enc_0_mha_wk_BiasAdd_ReadVariableOp_3D_array.data = AI_PTR(g_network_weights_map[0] + 36);
    tiny_bert_generator_enc_0_mha_wk_BiasAdd_ReadVariableOp_3D_array.data_start = AI_PTR(g_network_weights_map[0] + 36);
    tiny_bert_generator_enc_0_mha_wq_BiasAdd_ReadVariableOp_3D_array.format |= AI_FMT_FLAG_CONST;
    tiny_bert_generator_enc_0_mha_wq_BiasAdd_ReadVariableOp_3D_array.data = AI_PTR(g_network_weights_map[0] + 292);
    tiny_bert_generator_enc_0_mha_wq_BiasAdd_ReadVariableOp_3D_array.data_start = AI_PTR(g_network_weights_map[0] + 292);
    tiny_bert_generator_enc_3_mha_truedivtiny_bert_generator_enc_3_mha_Sqrt_4D_array.format |= AI_FMT_FLAG_CONST;
    tiny_bert_generator_enc_3_mha_truedivtiny_bert_generator_enc_3_mha_Sqrt_4D_array.data = AI_PTR(g_network_weights_map[0] + 548);
    tiny_bert_generator_enc_3_mha_truedivtiny_bert_generator_enc_3_mha_Sqrt_4D_array.data_start = AI_PTR(g_network_weights_map[0] + 548);
    tiny_bert_generator_enc_0_mha_wv_BiasAdd_ReadVariableOp_3D_array.format |= AI_FMT_FLAG_CONST;
    tiny_bert_generator_enc_0_mha_wv_BiasAdd_ReadVariableOp_3D_array.data = AI_PTR(g_network_weights_map[0] + 552);
    tiny_bert_generator_enc_0_mha_wv_BiasAdd_ReadVariableOp_3D_array.data_start = AI_PTR(g_network_weights_map[0] + 552);
    tiny_bert_generator_enc_0_mha_output_dense_BiasAdd_ReadVariableOp_3D_array.format |= AI_FMT_FLAG_CONST;
    tiny_bert_generator_enc_0_mha_output_dense_BiasAdd_ReadVariableOp_3D_array.data = AI_PTR(g_network_weights_map[0] + 808);
    tiny_bert_generator_enc_0_mha_output_dense_BiasAdd_ReadVariableOp_3D_array.data_start = AI_PTR(g_network_weights_map[0] + 808);
    tiny_bert_generator_enc_0_ln1_batchnorm_add_y_3D_array.format |= AI_FMT_FLAG_CONST;
    tiny_bert_generator_enc_0_ln1_batchnorm_add_y_3D_array.data = AI_PTR(g_network_weights_map[0] + 1064);
    tiny_bert_generator_enc_0_ln1_batchnorm_add_y_3D_array.data_start = AI_PTR(g_network_weights_map[0] + 1064);
    tiny_bert_generator_enc_0_ln1_batchnorm_mul_ReadVariableOp_3D_array.format |= AI_FMT_FLAG_CONST;
    tiny_bert_generator_enc_0_ln1_batchnorm_mul_ReadVariableOp_3D_array.data = AI_PTR(g_network_weights_map[0] + 1068);
    tiny_bert_generator_enc_0_ln1_batchnorm_mul_ReadVariableOp_3D_array.data_start = AI_PTR(g_network_weights_map[0] + 1068);
    tiny_bert_generator_enc_0_ln1_batchnorm_ReadVariableOp_3D_array.format |= AI_FMT_FLAG_CONST;
    tiny_bert_generator_enc_0_ln1_batchnorm_ReadVariableOp_3D_array.data = AI_PTR(g_network_weights_map[0] + 1324);
    tiny_bert_generator_enc_0_ln1_batchnorm_ReadVariableOp_3D_array.data_start = AI_PTR(g_network_weights_map[0] + 1324);
    tiny_bert_generator_enc_0_ffn_dense_24_BiasAdd_ReadVariableOp_3D_array.format |= AI_FMT_FLAG_CONST;
    tiny_bert_generator_enc_0_ffn_dense_24_BiasAdd_ReadVariableOp_3D_array.data = AI_PTR(g_network_weights_map[0] + 1580);
    tiny_bert_generator_enc_0_ffn_dense_24_BiasAdd_ReadVariableOp_3D_array.data_start = AI_PTR(g_network_weights_map[0] + 1580);
    tiny_bert_generator_enc_0_ffn_dense_25_BiasAdd_ReadVariableOp_3D_array.format |= AI_FMT_FLAG_CONST;
    tiny_bert_generator_enc_0_ffn_dense_25_BiasAdd_ReadVariableOp_3D_array.data = AI_PTR(g_network_weights_map[0] + 2092);
    tiny_bert_generator_enc_0_ffn_dense_25_BiasAdd_ReadVariableOp_3D_array.data_start = AI_PTR(g_network_weights_map[0] + 2092);
    tiny_bert_generator_enc_0_ln2_batchnorm_mul_ReadVariableOp_3D_array.format |= AI_FMT_FLAG_CONST;
    tiny_bert_generator_enc_0_ln2_batchnorm_mul_ReadVariableOp_3D_array.data = AI_PTR(g_network_weights_map[0] + 2348);
    tiny_bert_generator_enc_0_ln2_batchnorm_mul_ReadVariableOp_3D_array.data_start = AI_PTR(g_network_weights_map[0] + 2348);
    tiny_bert_generator_enc_0_ln2_batchnorm_ReadVariableOp_3D_array.format |= AI_FMT_FLAG_CONST;
    tiny_bert_generator_enc_0_ln2_batchnorm_ReadVariableOp_3D_array.data = AI_PTR(g_network_weights_map[0] + 2604);
    tiny_bert_generator_enc_0_ln2_batchnorm_ReadVariableOp_3D_array.data_start = AI_PTR(g_network_weights_map[0] + 2604);
    tiny_bert_generator_enc_1_mha_wk_BiasAdd_ReadVariableOp_3D_array.format |= AI_FMT_FLAG_CONST;
    tiny_bert_generator_enc_1_mha_wk_BiasAdd_ReadVariableOp_3D_array.data = AI_PTR(g_network_weights_map[0] + 2860);
    tiny_bert_generator_enc_1_mha_wk_BiasAdd_ReadVariableOp_3D_array.data_start = AI_PTR(g_network_weights_map[0] + 2860);
    tiny_bert_generator_enc_1_mha_wq_BiasAdd_ReadVariableOp_3D_array.format |= AI_FMT_FLAG_CONST;
    tiny_bert_generator_enc_1_mha_wq_BiasAdd_ReadVariableOp_3D_array.data = AI_PTR(g_network_weights_map[0] + 3116);
    tiny_bert_generator_enc_1_mha_wq_BiasAdd_ReadVariableOp_3D_array.data_start = AI_PTR(g_network_weights_map[0] + 3116);
    tiny_bert_generator_enc_1_mha_wv_BiasAdd_ReadVariableOp_3D_array.format |= AI_FMT_FLAG_CONST;
    tiny_bert_generator_enc_1_mha_wv_BiasAdd_ReadVariableOp_3D_array.data = AI_PTR(g_network_weights_map[0] + 3372);
    tiny_bert_generator_enc_1_mha_wv_BiasAdd_ReadVariableOp_3D_array.data_start = AI_PTR(g_network_weights_map[0] + 3372);
    tiny_bert_generator_enc_1_mha_output_dense_BiasAdd_ReadVariableOp_3D_array.format |= AI_FMT_FLAG_CONST;
    tiny_bert_generator_enc_1_mha_output_dense_BiasAdd_ReadVariableOp_3D_array.data = AI_PTR(g_network_weights_map[0] + 3628);
    tiny_bert_generator_enc_1_mha_output_dense_BiasAdd_ReadVariableOp_3D_array.data_start = AI_PTR(g_network_weights_map[0] + 3628);
    tiny_bert_generator_enc_1_ln1_batchnorm_mul_ReadVariableOp_3D_array.format |= AI_FMT_FLAG_CONST;
    tiny_bert_generator_enc_1_ln1_batchnorm_mul_ReadVariableOp_3D_array.data = AI_PTR(g_network_weights_map[0] + 3884);
    tiny_bert_generator_enc_1_ln1_batchnorm_mul_ReadVariableOp_3D_array.data_start = AI_PTR(g_network_weights_map[0] + 3884);
    tiny_bert_generator_enc_1_ln1_batchnorm_ReadVariableOp_3D_array.format |= AI_FMT_FLAG_CONST;
    tiny_bert_generator_enc_1_ln1_batchnorm_ReadVariableOp_3D_array.data = AI_PTR(g_network_weights_map[0] + 4140);
    tiny_bert_generator_enc_1_ln1_batchnorm_ReadVariableOp_3D_array.data_start = AI_PTR(g_network_weights_map[0] + 4140);
    tiny_bert_generator_enc_1_ffn_dense_26_BiasAdd_ReadVariableOp_3D_array.format |= AI_FMT_FLAG_CONST;
    tiny_bert_generator_enc_1_ffn_dense_26_BiasAdd_ReadVariableOp_3D_array.data = AI_PTR(g_network_weights_map[0] + 4396);
    tiny_bert_generator_enc_1_ffn_dense_26_BiasAdd_ReadVariableOp_3D_array.data_start = AI_PTR(g_network_weights_map[0] + 4396);
    tiny_bert_generator_enc_1_ffn_dense_27_BiasAdd_ReadVariableOp_3D_array.format |= AI_FMT_FLAG_CONST;
    tiny_bert_generator_enc_1_ffn_dense_27_BiasAdd_ReadVariableOp_3D_array.data = AI_PTR(g_network_weights_map[0] + 4908);
    tiny_bert_generator_enc_1_ffn_dense_27_BiasAdd_ReadVariableOp_3D_array.data_start = AI_PTR(g_network_weights_map[0] + 4908);
    tiny_bert_generator_enc_1_ln2_batchnorm_mul_ReadVariableOp_3D_array.format |= AI_FMT_FLAG_CONST;
    tiny_bert_generator_enc_1_ln2_batchnorm_mul_ReadVariableOp_3D_array.data = AI_PTR(g_network_weights_map[0] + 5164);
    tiny_bert_generator_enc_1_ln2_batchnorm_mul_ReadVariableOp_3D_array.data_start = AI_PTR(g_network_weights_map[0] + 5164);
    tiny_bert_generator_enc_1_ln2_batchnorm_ReadVariableOp_3D_array.format |= AI_FMT_FLAG_CONST;
    tiny_bert_generator_enc_1_ln2_batchnorm_ReadVariableOp_3D_array.data = AI_PTR(g_network_weights_map[0] + 5420);
    tiny_bert_generator_enc_1_ln2_batchnorm_ReadVariableOp_3D_array.data_start = AI_PTR(g_network_weights_map[0] + 5420);
    tiny_bert_generator_enc_2_mha_wk_BiasAdd_ReadVariableOp_3D_array.format |= AI_FMT_FLAG_CONST;
    tiny_bert_generator_enc_2_mha_wk_BiasAdd_ReadVariableOp_3D_array.data = AI_PTR(g_network_weights_map[0] + 5676);
    tiny_bert_generator_enc_2_mha_wk_BiasAdd_ReadVariableOp_3D_array.data_start = AI_PTR(g_network_weights_map[0] + 5676);
    tiny_bert_generator_enc_2_mha_wq_BiasAdd_ReadVariableOp_3D_array.format |= AI_FMT_FLAG_CONST;
    tiny_bert_generator_enc_2_mha_wq_BiasAdd_ReadVariableOp_3D_array.data = AI_PTR(g_network_weights_map[0] + 5932);
    tiny_bert_generator_enc_2_mha_wq_BiasAdd_ReadVariableOp_3D_array.data_start = AI_PTR(g_network_weights_map[0] + 5932);
    tiny_bert_generator_enc_2_mha_wv_BiasAdd_ReadVariableOp_3D_array.format |= AI_FMT_FLAG_CONST;
    tiny_bert_generator_enc_2_mha_wv_BiasAdd_ReadVariableOp_3D_array.data = AI_PTR(g_network_weights_map[0] + 6188);
    tiny_bert_generator_enc_2_mha_wv_BiasAdd_ReadVariableOp_3D_array.data_start = AI_PTR(g_network_weights_map[0] + 6188);
    tiny_bert_generator_enc_2_mha_output_dense_BiasAdd_ReadVariableOp_3D_array.format |= AI_FMT_FLAG_CONST;
    tiny_bert_generator_enc_2_mha_output_dense_BiasAdd_ReadVariableOp_3D_array.data = AI_PTR(g_network_weights_map[0] + 6444);
    tiny_bert_generator_enc_2_mha_output_dense_BiasAdd_ReadVariableOp_3D_array.data_start = AI_PTR(g_network_weights_map[0] + 6444);
    tiny_bert_generator_enc_2_ln1_batchnorm_mul_ReadVariableOp_3D_array.format |= AI_FMT_FLAG_CONST;
    tiny_bert_generator_enc_2_ln1_batchnorm_mul_ReadVariableOp_3D_array.data = AI_PTR(g_network_weights_map[0] + 6700);
    tiny_bert_generator_enc_2_ln1_batchnorm_mul_ReadVariableOp_3D_array.data_start = AI_PTR(g_network_weights_map[0] + 6700);
    tiny_bert_generator_enc_2_ln1_batchnorm_ReadVariableOp_3D_array.format |= AI_FMT_FLAG_CONST;
    tiny_bert_generator_enc_2_ln1_batchnorm_ReadVariableOp_3D_array.data = AI_PTR(g_network_weights_map[0] + 6956);
    tiny_bert_generator_enc_2_ln1_batchnorm_ReadVariableOp_3D_array.data_start = AI_PTR(g_network_weights_map[0] + 6956);
    tiny_bert_generator_enc_2_ffn_dense_28_BiasAdd_ReadVariableOp_3D_array.format |= AI_FMT_FLAG_CONST;
    tiny_bert_generator_enc_2_ffn_dense_28_BiasAdd_ReadVariableOp_3D_array.data = AI_PTR(g_network_weights_map[0] + 7212);
    tiny_bert_generator_enc_2_ffn_dense_28_BiasAdd_ReadVariableOp_3D_array.data_start = AI_PTR(g_network_weights_map[0] + 7212);
    tiny_bert_generator_enc_2_ffn_dense_29_BiasAdd_ReadVariableOp_3D_array.format |= AI_FMT_FLAG_CONST;
    tiny_bert_generator_enc_2_ffn_dense_29_BiasAdd_ReadVariableOp_3D_array.data = AI_PTR(g_network_weights_map[0] + 7724);
    tiny_bert_generator_enc_2_ffn_dense_29_BiasAdd_ReadVariableOp_3D_array.data_start = AI_PTR(g_network_weights_map[0] + 7724);
    tiny_bert_generator_enc_2_ln2_batchnorm_mul_ReadVariableOp_3D_array.format |= AI_FMT_FLAG_CONST;
    tiny_bert_generator_enc_2_ln2_batchnorm_mul_ReadVariableOp_3D_array.data = AI_PTR(g_network_weights_map[0] + 7980);
    tiny_bert_generator_enc_2_ln2_batchnorm_mul_ReadVariableOp_3D_array.data_start = AI_PTR(g_network_weights_map[0] + 7980);
    tiny_bert_generator_enc_2_ln2_batchnorm_ReadVariableOp_3D_array.format |= AI_FMT_FLAG_CONST;
    tiny_bert_generator_enc_2_ln2_batchnorm_ReadVariableOp_3D_array.data = AI_PTR(g_network_weights_map[0] + 8236);
    tiny_bert_generator_enc_2_ln2_batchnorm_ReadVariableOp_3D_array.data_start = AI_PTR(g_network_weights_map[0] + 8236);
    tiny_bert_generator_enc_3_mha_wk_BiasAdd_ReadVariableOp_3D_array.format |= AI_FMT_FLAG_CONST;
    tiny_bert_generator_enc_3_mha_wk_BiasAdd_ReadVariableOp_3D_array.data = AI_PTR(g_network_weights_map[0] + 8492);
    tiny_bert_generator_enc_3_mha_wk_BiasAdd_ReadVariableOp_3D_array.data_start = AI_PTR(g_network_weights_map[0] + 8492);
    tiny_bert_generator_enc_3_mha_wq_BiasAdd_ReadVariableOp_3D_array.format |= AI_FMT_FLAG_CONST;
    tiny_bert_generator_enc_3_mha_wq_BiasAdd_ReadVariableOp_3D_array.data = AI_PTR(g_network_weights_map[0] + 8748);
    tiny_bert_generator_enc_3_mha_wq_BiasAdd_ReadVariableOp_3D_array.data_start = AI_PTR(g_network_weights_map[0] + 8748);
    tiny_bert_generator_enc_3_mha_wv_BiasAdd_ReadVariableOp_3D_array.format |= AI_FMT_FLAG_CONST;
    tiny_bert_generator_enc_3_mha_wv_BiasAdd_ReadVariableOp_3D_array.data = AI_PTR(g_network_weights_map[0] + 9004);
    tiny_bert_generator_enc_3_mha_wv_BiasAdd_ReadVariableOp_3D_array.data_start = AI_PTR(g_network_weights_map[0] + 9004);
    tiny_bert_generator_enc_3_mha_output_dense_BiasAdd_ReadVariableOp_3D_array.format |= AI_FMT_FLAG_CONST;
    tiny_bert_generator_enc_3_mha_output_dense_BiasAdd_ReadVariableOp_3D_array.data = AI_PTR(g_network_weights_map[0] + 9260);
    tiny_bert_generator_enc_3_mha_output_dense_BiasAdd_ReadVariableOp_3D_array.data_start = AI_PTR(g_network_weights_map[0] + 9260);
    tiny_bert_generator_enc_3_ln1_batchnorm_mul_ReadVariableOp_3D_array.format |= AI_FMT_FLAG_CONST;
    tiny_bert_generator_enc_3_ln1_batchnorm_mul_ReadVariableOp_3D_array.data = AI_PTR(g_network_weights_map[0] + 9516);
    tiny_bert_generator_enc_3_ln1_batchnorm_mul_ReadVariableOp_3D_array.data_start = AI_PTR(g_network_weights_map[0] + 9516);
    tiny_bert_generator_enc_3_ln1_batchnorm_ReadVariableOp_3D_array.format |= AI_FMT_FLAG_CONST;
    tiny_bert_generator_enc_3_ln1_batchnorm_ReadVariableOp_3D_array.data = AI_PTR(g_network_weights_map[0] + 9772);
    tiny_bert_generator_enc_3_ln1_batchnorm_ReadVariableOp_3D_array.data_start = AI_PTR(g_network_weights_map[0] + 9772);
    tiny_bert_generator_enc_3_ffn_dense_30_BiasAdd_ReadVariableOp_3D_array.format |= AI_FMT_FLAG_CONST;
    tiny_bert_generator_enc_3_ffn_dense_30_BiasAdd_ReadVariableOp_3D_array.data = AI_PTR(g_network_weights_map[0] + 10028);
    tiny_bert_generator_enc_3_ffn_dense_30_BiasAdd_ReadVariableOp_3D_array.data_start = AI_PTR(g_network_weights_map[0] + 10028);
    tiny_bert_generator_enc_3_ffn_dense_31_BiasAdd_ReadVariableOp_3D_array.format |= AI_FMT_FLAG_CONST;
    tiny_bert_generator_enc_3_ffn_dense_31_BiasAdd_ReadVariableOp_3D_array.data = AI_PTR(g_network_weights_map[0] + 10540);
    tiny_bert_generator_enc_3_ffn_dense_31_BiasAdd_ReadVariableOp_3D_array.data_start = AI_PTR(g_network_weights_map[0] + 10540);
    tiny_bert_generator_enc_3_ln2_batchnorm_mul_ReadVariableOp_3D_array.format |= AI_FMT_FLAG_CONST;
    tiny_bert_generator_enc_3_ln2_batchnorm_mul_ReadVariableOp_3D_array.data = AI_PTR(g_network_weights_map[0] + 10796);
    tiny_bert_generator_enc_3_ln2_batchnorm_mul_ReadVariableOp_3D_array.data_start = AI_PTR(g_network_weights_map[0] + 10796);
    tiny_bert_generator_enc_3_ln2_batchnorm_ReadVariableOp_3D_array.format |= AI_FMT_FLAG_CONST;
    tiny_bert_generator_enc_3_ln2_batchnorm_ReadVariableOp_3D_array.data = AI_PTR(g_network_weights_map[0] + 11052);
    tiny_bert_generator_enc_3_ln2_batchnorm_ReadVariableOp_3D_array.data_start = AI_PTR(g_network_weights_map[0] + 11052);
    tiny_bert_generator_output_dense_BiasAdd_ReadVariableOp_3D_array.format |= AI_FMT_FLAG_CONST;
    tiny_bert_generator_output_dense_BiasAdd_ReadVariableOp_3D_array.data = AI_PTR(g_network_weights_map[0] + 11308);
    tiny_bert_generator_output_dense_BiasAdd_ReadVariableOp_3D_array.data_start = AI_PTR(g_network_weights_map[0] + 11308);
    tiny_bert_generator_tf___operators___add_1_y_array.format |= AI_FMT_FLAG_CONST;
    tiny_bert_generator_tf___operators___add_1_y_array.data = AI_PTR(g_network_weights_map[0] + 11376);
    tiny_bert_generator_tf___operators___add_1_y_array.data_start = AI_PTR(g_network_weights_map[0] + 11376);
    gemm_26_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_26_weights_array.data = AI_PTR(g_network_weights_map[0] + 36976);
    gemm_26_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 36976);
    gemm_26_bias_array.format |= AI_FMT_FLAG_CONST;
    gemm_26_bias_array.data = AI_PTR(g_network_weights_map[0] + 102512);
    gemm_26_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 102512);
    gemm_18_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_18_weights_array.data = AI_PTR(g_network_weights_map[0] + 103536);
    gemm_18_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 103536);
    gemm_13_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_13_weights_array.data = AI_PTR(g_network_weights_map[0] + 169072);
    gemm_13_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 169072);
    gemm_42_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_42_weights_array.data = AI_PTR(g_network_weights_map[0] + 234608);
    gemm_42_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 234608);
    reduce_46_Mul_scale_array.format |= AI_FMT_FLAG_CONST;
    reduce_46_Mul_scale_array.data = AI_PTR(g_network_weights_map[0] + 300144);
    reduce_46_Mul_scale_array.data_start = AI_PTR(g_network_weights_map[0] + 300144);
    reduce_46_Mul_bias_array.format |= AI_FMT_FLAG_CONST;
    reduce_46_Mul_bias_array.data = AI_PTR(g_network_weights_map[0] + 300148);
    reduce_46_Mul_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 300148);
    gemm_64_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_64_weights_array.data = AI_PTR(g_network_weights_map[0] + 300152);
    gemm_64_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 300152);
    gemm_64_bias_array.format |= AI_FMT_FLAG_CONST;
    gemm_64_bias_array.data = AI_PTR(g_network_weights_map[0] + 431224);
    gemm_64_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 431224);
    gemm_75_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_75_weights_array.data = AI_PTR(g_network_weights_map[0] + 433272);
    gemm_75_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 433272);
    gemm_113_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_113_weights_array.data = AI_PTR(g_network_weights_map[0] + 564344);
    gemm_113_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 564344);
    gemm_105_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_105_weights_array.data = AI_PTR(g_network_weights_map[0] + 629880);
    gemm_105_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 629880);
    gemm_100_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_100_weights_array.data = AI_PTR(g_network_weights_map[0] + 695416);
    gemm_100_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 695416);
    gemm_129_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_129_weights_array.data = AI_PTR(g_network_weights_map[0] + 760952);
    gemm_129_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 760952);
    gemm_151_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_151_weights_array.data = AI_PTR(g_network_weights_map[0] + 826488);
    gemm_151_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 826488);
    gemm_162_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_162_weights_array.data = AI_PTR(g_network_weights_map[0] + 957560);
    gemm_162_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 957560);
    gemm_200_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_200_weights_array.data = AI_PTR(g_network_weights_map[0] + 1088632);
    gemm_200_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 1088632);
    gemm_192_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_192_weights_array.data = AI_PTR(g_network_weights_map[0] + 1154168);
    gemm_192_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 1154168);
    gemm_187_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_187_weights_array.data = AI_PTR(g_network_weights_map[0] + 1219704);
    gemm_187_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 1219704);
    gemm_216_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_216_weights_array.data = AI_PTR(g_network_weights_map[0] + 1285240);
    gemm_216_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 1285240);
    gemm_238_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_238_weights_array.data = AI_PTR(g_network_weights_map[0] + 1350776);
    gemm_238_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 1350776);
    gemm_249_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_249_weights_array.data = AI_PTR(g_network_weights_map[0] + 1481848);
    gemm_249_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 1481848);
    gemm_287_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_287_weights_array.data = AI_PTR(g_network_weights_map[0] + 1612920);
    gemm_287_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 1612920);
    gemm_279_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_279_weights_array.data = AI_PTR(g_network_weights_map[0] + 1678456);
    gemm_279_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 1678456);
    gemm_274_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_274_weights_array.data = AI_PTR(g_network_weights_map[0] + 1743992);
    gemm_274_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 1743992);
    gemm_303_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_303_weights_array.data = AI_PTR(g_network_weights_map[0] + 1809528);
    gemm_303_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 1809528);
    gemm_325_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_325_weights_array.data = AI_PTR(g_network_weights_map[0] + 1875064);
    gemm_325_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 1875064);
    gemm_336_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_336_weights_array.data = AI_PTR(g_network_weights_map[0] + 2006136);
    gemm_336_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 2006136);
    gemm_358_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_358_weights_array.data = AI_PTR(g_network_weights_map[0] + 2137208);
    gemm_358_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 2137208);
    gemm_358_bias_array.format |= AI_FMT_FLAG_CONST;
    gemm_358_bias_array.data = AI_PTR(g_network_weights_map[0] + 2154104);
    gemm_358_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 2154104);
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_WEIGHTS);
  return false;
}


/**  PUBLIC APIs SECTION  *****************************************************/



AI_DEPRECATED
AI_API_ENTRY
ai_bool ai_network_get_info(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_NETWORK_MODEL_NAME,
      .model_signature   = AI_NETWORK_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = AI_STRUCT_INIT,

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 8411143898,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .params            = AI_STRUCT_INIT,
      .activations       = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0x17a8d7a1,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}



AI_API_ENTRY
ai_bool ai_network_get_report(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_NETWORK_MODEL_NAME,
      .model_signature   = AI_NETWORK_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = AI_STRUCT_INIT,

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 8411143898,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .map_signature     = AI_MAGIC_SIGNATURE,
      .map_weights       = AI_STRUCT_INIT,
      .map_activations   = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0x17a8d7a1,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}


AI_API_ENTRY
ai_error ai_network_get_error(ai_handle network)
{
  return ai_platform_network_get_error(network);
}


AI_API_ENTRY
ai_error ai_network_create(
  ai_handle* network, const ai_buffer* network_config)
{
  return ai_platform_network_create(
    network, network_config, 
    AI_CONTEXT_OBJ(&AI_NET_OBJ_INSTANCE),
    AI_TOOLS_API_VERSION_MAJOR, AI_TOOLS_API_VERSION_MINOR, AI_TOOLS_API_VERSION_MICRO);
}


AI_API_ENTRY
ai_error ai_network_create_and_init(
  ai_handle* network, const ai_handle activations[], const ai_handle weights[])
{
  ai_error err;
  ai_network_params params;

  err = ai_network_create(network, AI_NETWORK_DATA_CONFIG);
  if (err.type != AI_ERROR_NONE) {
    return err;
  }
  
  if (ai_network_data_params_get(&params) != true) {
    err = ai_network_get_error(*network);
    return err;
  }
#if defined(AI_NETWORK_DATA_ACTIVATIONS_COUNT)
  /* set the addresses of the activations buffers */
  for (ai_u16 idx=0; activations && idx<params.map_activations.size; idx++) {
    AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_activations, idx, activations[idx]);
  }
#endif
#if defined(AI_NETWORK_DATA_WEIGHTS_COUNT)
  /* set the addresses of the weight buffers */
  for (ai_u16 idx=0; weights && idx<params.map_weights.size; idx++) {
    AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_weights, idx, weights[idx]);
  }
#endif
  if (ai_network_init(*network, &params) != true) {
    err = ai_network_get_error(*network);
  }
  return err;
}


AI_API_ENTRY
ai_buffer* ai_network_inputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    AI_NETWORK_OBJ(network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_inputs_get(network, n_buffer);
}


AI_API_ENTRY
ai_buffer* ai_network_outputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    AI_NETWORK_OBJ(network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_outputs_get(network, n_buffer);
}


AI_API_ENTRY
ai_handle ai_network_destroy(ai_handle network)
{
  return ai_platform_network_destroy(network);
}


AI_API_ENTRY
ai_bool ai_network_init(
  ai_handle network, const ai_network_params* params)
{
  ai_network* net_ctx = AI_NETWORK_OBJ(ai_platform_network_init(network, params));
  ai_bool ok = true;

  if (!net_ctx) return false;
  ok &= network_configure_weights(net_ctx, params);
  ok &= network_configure_activations(net_ctx, params);

  ok &= ai_platform_network_post_init(network);

  return ok;
}


AI_API_ENTRY
ai_i32 ai_network_run(
  ai_handle network, const ai_buffer* input, ai_buffer* output)
{
  return ai_platform_network_process(network, input, output);
}


AI_API_ENTRY
ai_i32 ai_network_forward(ai_handle network, const ai_buffer* input)
{
  return ai_platform_network_process(network, input, NULL);
}



#undef AI_NETWORK_MODEL_SIGNATURE
#undef AI_NET_OBJ_INSTANCE
#undef AI_TOOLS_DATE_TIME
#undef AI_TOOLS_COMPILE_TIME

