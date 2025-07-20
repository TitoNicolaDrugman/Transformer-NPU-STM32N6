/**
  ******************************************************************************
  * @file    network.c
  * @author  AST Embedded Analytics Research Platform
  * @date    2025-06-24T12:00:42+0200
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
#define AI_NETWORK_MODEL_SIGNATURE     "0xb547d7a78f99968df5d62d3a153468d0"

#ifndef AI_TOOLS_REVISION_ID
#define AI_TOOLS_REVISION_ID     ""
#endif

#undef AI_TOOLS_DATE_TIME
#define AI_TOOLS_DATE_TIME   "2025-06-24T12:00:42+0200"

#undef AI_TOOLS_COMPILE_TIME
#define AI_TOOLS_COMPILE_TIME    __DATE__ " " __TIME__

#undef AI_NETWORK_N_BATCHES
#define AI_NETWORK_N_BATCHES         (1)

static ai_ptr g_network_activations_map[1] = AI_C_ARRAY_INIT;
static ai_ptr g_network_weights_map[1] = AI_C_ARRAY_INIT;



/**  Array declarations section  **********************************************/
/* Array#0 */
AI_ARRAY_OBJ_DECLARE(
  gemm_293_bias_0_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)

/* Array#1 */
AI_ARRAY_OBJ_DECLARE(
  gemm_285_bias_0_2_gemm_285_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)

/* Array#2 */
AI_ARRAY_OBJ_DECLARE(
  gemm_206_bias_0_2_gemm_206_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)

/* Array#3 */
AI_ARRAY_OBJ_DECLARE(
  gemm_198_bias_0_2_gemm_198_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)

/* Array#4 */
AI_ARRAY_OBJ_DECLARE(
  gemm_119_bias_0_2_gemm_119_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)

/* Array#5 */
AI_ARRAY_OBJ_DECLARE(
  gemm_111_bias_0_2_gemm_111_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)

/* Array#6 */
AI_ARRAY_OBJ_DECLARE(
  gemm_32_bias_0_2_gemm_32_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)

/* Array#7 */
AI_ARRAY_OBJ_DECLARE(
  gemm_24_bias_0_2_gemm_24_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)

/* Array#8 */
AI_ARRAY_OBJ_DECLARE(
  serving_default_input_10_output_array, AI_ARRAY_FORMAT_S32|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 100, AI_STATIC)

/* Array#9 */
AI_ARRAY_OBJ_DECLARE(
  gather_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#10 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_1_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#11 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_2_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#12 */
AI_ARRAY_OBJ_DECLARE(
  gemm_27_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#13 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_29_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#14 */
AI_ARRAY_OBJ_DECLARE(
  transpose_31_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#15 */
AI_ARRAY_OBJ_DECLARE(
  transpose_31_0_1_gemm_32_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#16 */
AI_ARRAY_OBJ_DECLARE(
  gemm_19_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#17 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_21_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#18 */
AI_ARRAY_OBJ_DECLARE(
  transpose_bgemm_24_out_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#19 */
AI_ARRAY_OBJ_DECLARE(
  transpose_bgemm_24_out_0_1_gemm_24_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#20 */
AI_ARRAY_OBJ_DECLARE(
  gemm_14_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#21 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_16_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#22 */
AI_ARRAY_OBJ_DECLARE(
  transpose_18_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#23 */
AI_ARRAY_OBJ_DECLARE(
  transpose_18_0_0_gemm_24_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#24 */
AI_ARRAY_OBJ_DECLARE(
  gemm_24_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 40000, AI_STATIC)

/* Array#25 */
AI_ARRAY_OBJ_DECLARE(
  gemm_24_0_0_eltwise_25_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 40000, AI_STATIC)

/* Array#26 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_25_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 40000, AI_STATIC)

/* Array#27 */
AI_ARRAY_OBJ_DECLARE(
  nl_26_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 40000, AI_STATIC)

/* Array#28 */
AI_ARRAY_OBJ_DECLARE(
  nl_26_0_0_gemm_32_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 40000, AI_STATIC)

/* Array#29 */
AI_ARRAY_OBJ_DECLARE(
  gemm_32_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#30 */
AI_ARRAY_OBJ_DECLARE(
  gemm_32_0_0_transpose_33_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#31 */
AI_ARRAY_OBJ_DECLARE(
  transpose_33_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#32 */
AI_ARRAY_OBJ_DECLARE(
  gemm_43_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#33 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_45_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#34 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_46_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#35 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_46_0_0_reduce_47_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#36 */
AI_ARRAY_OBJ_DECLARE(
  reduce_47_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#37 */
AI_ARRAY_OBJ_DECLARE(
  reduce_47_Mul_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#38 */
AI_ARRAY_OBJ_DECLARE(
  reduce_47_Mul_0_1_eltwise_48_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 100, AI_STATIC)

/* Array#39 */
AI_ARRAY_OBJ_DECLARE(
  reduce_47_Mul_0_1_eltwise_48_conversion_0_1_eltwise_48_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#40 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_48_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#41 */
AI_ARRAY_OBJ_DECLARE(
  reduce_49_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#42 */
AI_ARRAY_OBJ_DECLARE(
  reduce_49_Mul_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#43 */
AI_ARRAY_OBJ_DECLARE(
  reduce_49_Mul_0_0_eltwise_50_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 100, AI_STATIC)

/* Array#44 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_50_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 100, AI_STATIC)

/* Array#45 */
AI_ARRAY_OBJ_DECLARE(
  nl_51_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 100, AI_STATIC)

/* Array#46 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_52_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#47 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_54_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#48 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_55_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#49 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_53_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#50 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_56_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#51 */
AI_ARRAY_OBJ_DECLARE(
  gemm_65_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 51200, AI_STATIC)

/* Array#52 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_67_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 51200, AI_STATIC)

/* Array#53 */
AI_ARRAY_OBJ_DECLARE(
  gemm_76_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#54 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_78_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#55 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_79_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#56 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_79_0_0_reduce_80_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#57 */
AI_ARRAY_OBJ_DECLARE(
  reduce_80_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#58 */
AI_ARRAY_OBJ_DECLARE(
  reduce_80_Mul_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#59 */
AI_ARRAY_OBJ_DECLARE(
  reduce_80_Mul_0_1_eltwise_81_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 100, AI_STATIC)

/* Array#60 */
AI_ARRAY_OBJ_DECLARE(
  reduce_80_Mul_0_1_eltwise_81_conversion_0_1_eltwise_81_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#61 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_81_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#62 */
AI_ARRAY_OBJ_DECLARE(
  reduce_82_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#63 */
AI_ARRAY_OBJ_DECLARE(
  reduce_82_Mul_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#64 */
AI_ARRAY_OBJ_DECLARE(
  reduce_82_Mul_0_0_eltwise_83_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 100, AI_STATIC)

/* Array#65 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_83_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 100, AI_STATIC)

/* Array#66 */
AI_ARRAY_OBJ_DECLARE(
  nl_84_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 100, AI_STATIC)

/* Array#67 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_85_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#68 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_87_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#69 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_88_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#70 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_86_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#71 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_89_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#72 */
AI_ARRAY_OBJ_DECLARE(
  gemm_114_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#73 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_116_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#74 */
AI_ARRAY_OBJ_DECLARE(
  transpose_118_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#75 */
AI_ARRAY_OBJ_DECLARE(
  transpose_118_0_1_gemm_119_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#76 */
AI_ARRAY_OBJ_DECLARE(
  gemm_106_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#77 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_108_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#78 */
AI_ARRAY_OBJ_DECLARE(
  transpose_bgemm_111_out_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#79 */
AI_ARRAY_OBJ_DECLARE(
  transpose_bgemm_111_out_0_1_gemm_111_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#80 */
AI_ARRAY_OBJ_DECLARE(
  gemm_101_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#81 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_103_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#82 */
AI_ARRAY_OBJ_DECLARE(
  transpose_105_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#83 */
AI_ARRAY_OBJ_DECLARE(
  transpose_105_0_0_gemm_111_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#84 */
AI_ARRAY_OBJ_DECLARE(
  gemm_111_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 40000, AI_STATIC)

/* Array#85 */
AI_ARRAY_OBJ_DECLARE(
  gemm_111_0_0_eltwise_112_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 40000, AI_STATIC)

/* Array#86 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_112_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 40000, AI_STATIC)

/* Array#87 */
AI_ARRAY_OBJ_DECLARE(
  nl_113_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 40000, AI_STATIC)

/* Array#88 */
AI_ARRAY_OBJ_DECLARE(
  nl_113_0_0_gemm_119_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 40000, AI_STATIC)

/* Array#89 */
AI_ARRAY_OBJ_DECLARE(
  gemm_119_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#90 */
AI_ARRAY_OBJ_DECLARE(
  gemm_119_0_0_transpose_120_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#91 */
AI_ARRAY_OBJ_DECLARE(
  transpose_120_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#92 */
AI_ARRAY_OBJ_DECLARE(
  gemm_130_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#93 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_132_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#94 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_133_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#95 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_133_0_0_reduce_134_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#96 */
AI_ARRAY_OBJ_DECLARE(
  reduce_134_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#97 */
AI_ARRAY_OBJ_DECLARE(
  reduce_134_Mul_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#98 */
AI_ARRAY_OBJ_DECLARE(
  reduce_134_Mul_0_1_eltwise_135_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 100, AI_STATIC)

/* Array#99 */
AI_ARRAY_OBJ_DECLARE(
  reduce_134_Mul_0_1_eltwise_135_conversion_0_1_eltwise_135_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#100 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_135_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#101 */
AI_ARRAY_OBJ_DECLARE(
  reduce_136_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#102 */
AI_ARRAY_OBJ_DECLARE(
  reduce_136_Mul_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#103 */
AI_ARRAY_OBJ_DECLARE(
  reduce_136_Mul_0_0_eltwise_137_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 100, AI_STATIC)

/* Array#104 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_137_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 100, AI_STATIC)

/* Array#105 */
AI_ARRAY_OBJ_DECLARE(
  nl_138_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 100, AI_STATIC)

/* Array#106 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_139_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#107 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_141_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#108 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_142_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#109 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_140_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#110 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_143_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#111 */
AI_ARRAY_OBJ_DECLARE(
  gemm_152_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 51200, AI_STATIC)

/* Array#112 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_154_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 51200, AI_STATIC)

/* Array#113 */
AI_ARRAY_OBJ_DECLARE(
  gemm_163_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#114 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_165_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#115 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_166_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#116 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_166_0_0_reduce_167_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#117 */
AI_ARRAY_OBJ_DECLARE(
  reduce_167_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#118 */
AI_ARRAY_OBJ_DECLARE(
  reduce_167_Mul_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#119 */
AI_ARRAY_OBJ_DECLARE(
  reduce_167_Mul_0_1_eltwise_168_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 100, AI_STATIC)

/* Array#120 */
AI_ARRAY_OBJ_DECLARE(
  reduce_167_Mul_0_1_eltwise_168_conversion_0_1_eltwise_168_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#121 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_168_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#122 */
AI_ARRAY_OBJ_DECLARE(
  reduce_169_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#123 */
AI_ARRAY_OBJ_DECLARE(
  reduce_169_Mul_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#124 */
AI_ARRAY_OBJ_DECLARE(
  reduce_169_Mul_0_0_eltwise_170_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 100, AI_STATIC)

/* Array#125 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_170_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 100, AI_STATIC)

/* Array#126 */
AI_ARRAY_OBJ_DECLARE(
  nl_171_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 100, AI_STATIC)

/* Array#127 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_172_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#128 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_174_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#129 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_175_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#130 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_173_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#131 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_176_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#132 */
AI_ARRAY_OBJ_DECLARE(
  gemm_201_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#133 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_203_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#134 */
AI_ARRAY_OBJ_DECLARE(
  transpose_205_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#135 */
AI_ARRAY_OBJ_DECLARE(
  transpose_205_0_1_gemm_206_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#136 */
AI_ARRAY_OBJ_DECLARE(
  gemm_193_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#137 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_195_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#138 */
AI_ARRAY_OBJ_DECLARE(
  transpose_bgemm_198_out_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#139 */
AI_ARRAY_OBJ_DECLARE(
  transpose_bgemm_198_out_0_1_gemm_198_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#140 */
AI_ARRAY_OBJ_DECLARE(
  gemm_188_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#141 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_190_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#142 */
AI_ARRAY_OBJ_DECLARE(
  transpose_192_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#143 */
AI_ARRAY_OBJ_DECLARE(
  transpose_192_0_0_gemm_198_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#144 */
AI_ARRAY_OBJ_DECLARE(
  gemm_198_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 40000, AI_STATIC)

/* Array#145 */
AI_ARRAY_OBJ_DECLARE(
  gemm_198_0_0_eltwise_199_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 40000, AI_STATIC)

/* Array#146 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_199_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 40000, AI_STATIC)

/* Array#147 */
AI_ARRAY_OBJ_DECLARE(
  nl_200_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 40000, AI_STATIC)

/* Array#148 */
AI_ARRAY_OBJ_DECLARE(
  nl_200_0_0_gemm_206_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 40000, AI_STATIC)

/* Array#149 */
AI_ARRAY_OBJ_DECLARE(
  gemm_206_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#150 */
AI_ARRAY_OBJ_DECLARE(
  gemm_206_0_0_transpose_207_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#151 */
AI_ARRAY_OBJ_DECLARE(
  transpose_207_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#152 */
AI_ARRAY_OBJ_DECLARE(
  gemm_217_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#153 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_219_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#154 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_220_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#155 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_220_0_0_reduce_221_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#156 */
AI_ARRAY_OBJ_DECLARE(
  reduce_221_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#157 */
AI_ARRAY_OBJ_DECLARE(
  reduce_221_Mul_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#158 */
AI_ARRAY_OBJ_DECLARE(
  reduce_221_Mul_0_1_eltwise_222_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 100, AI_STATIC)

/* Array#159 */
AI_ARRAY_OBJ_DECLARE(
  reduce_221_Mul_0_1_eltwise_222_conversion_0_1_eltwise_222_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#160 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_222_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#161 */
AI_ARRAY_OBJ_DECLARE(
  reduce_223_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#162 */
AI_ARRAY_OBJ_DECLARE(
  reduce_223_Mul_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#163 */
AI_ARRAY_OBJ_DECLARE(
  reduce_223_Mul_0_0_eltwise_224_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 100, AI_STATIC)

/* Array#164 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_224_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 100, AI_STATIC)

/* Array#165 */
AI_ARRAY_OBJ_DECLARE(
  nl_225_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 100, AI_STATIC)

/* Array#166 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_226_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#167 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_228_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#168 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_229_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#169 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_227_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#170 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_230_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#171 */
AI_ARRAY_OBJ_DECLARE(
  gemm_239_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 51200, AI_STATIC)

/* Array#172 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_241_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 51200, AI_STATIC)

/* Array#173 */
AI_ARRAY_OBJ_DECLARE(
  gemm_250_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#174 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_252_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#175 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_253_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#176 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_253_0_0_reduce_254_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#177 */
AI_ARRAY_OBJ_DECLARE(
  reduce_254_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#178 */
AI_ARRAY_OBJ_DECLARE(
  reduce_254_Mul_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#179 */
AI_ARRAY_OBJ_DECLARE(
  reduce_254_Mul_0_1_eltwise_255_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 100, AI_STATIC)

/* Array#180 */
AI_ARRAY_OBJ_DECLARE(
  reduce_254_Mul_0_1_eltwise_255_conversion_0_1_eltwise_255_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#181 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_255_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#182 */
AI_ARRAY_OBJ_DECLARE(
  reduce_256_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#183 */
AI_ARRAY_OBJ_DECLARE(
  reduce_256_Mul_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#184 */
AI_ARRAY_OBJ_DECLARE(
  reduce_256_Mul_0_0_eltwise_257_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 100, AI_STATIC)

/* Array#185 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_257_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 100, AI_STATIC)

/* Array#186 */
AI_ARRAY_OBJ_DECLARE(
  nl_258_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 100, AI_STATIC)

/* Array#187 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_259_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#188 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_261_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#189 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_262_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#190 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_260_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#191 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_263_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#192 */
AI_ARRAY_OBJ_DECLARE(
  gemm_288_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#193 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_290_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#194 */
AI_ARRAY_OBJ_DECLARE(
  transpose_292_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#195 */
AI_ARRAY_OBJ_DECLARE(
  transpose_292_0_1_gemm_293_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#196 */
AI_ARRAY_OBJ_DECLARE(
  gemm_280_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#197 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_282_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#198 */
AI_ARRAY_OBJ_DECLARE(
  transpose_bgemm_285_out_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#199 */
AI_ARRAY_OBJ_DECLARE(
  transpose_bgemm_285_out_0_1_gemm_285_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#200 */
AI_ARRAY_OBJ_DECLARE(
  gemm_275_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#201 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_277_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#202 */
AI_ARRAY_OBJ_DECLARE(
  transpose_279_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#203 */
AI_ARRAY_OBJ_DECLARE(
  transpose_279_0_0_gemm_285_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#204 */
AI_ARRAY_OBJ_DECLARE(
  gemm_285_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 40000, AI_STATIC)

/* Array#205 */
AI_ARRAY_OBJ_DECLARE(
  gemm_285_0_0_eltwise_286_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 40000, AI_STATIC)

/* Array#206 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_286_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 40000, AI_STATIC)

/* Array#207 */
AI_ARRAY_OBJ_DECLARE(
  nl_287_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 40000, AI_STATIC)

/* Array#208 */
AI_ARRAY_OBJ_DECLARE(
  nl_287_0_0_gemm_293_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 40000, AI_STATIC)

/* Array#209 */
AI_ARRAY_OBJ_DECLARE(
  gemm_293_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#210 */
AI_ARRAY_OBJ_DECLARE(
  gemm_293_0_0_transpose_294_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#211 */
AI_ARRAY_OBJ_DECLARE(
  transpose_294_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#212 */
AI_ARRAY_OBJ_DECLARE(
  gemm_304_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#213 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_306_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#214 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_307_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#215 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_307_0_0_reduce_308_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#216 */
AI_ARRAY_OBJ_DECLARE(
  reduce_308_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#217 */
AI_ARRAY_OBJ_DECLARE(
  reduce_308_Mul_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#218 */
AI_ARRAY_OBJ_DECLARE(
  reduce_308_Mul_0_1_eltwise_309_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 100, AI_STATIC)

/* Array#219 */
AI_ARRAY_OBJ_DECLARE(
  reduce_308_Mul_0_1_eltwise_309_conversion_0_1_eltwise_309_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#220 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_309_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#221 */
AI_ARRAY_OBJ_DECLARE(
  reduce_310_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#222 */
AI_ARRAY_OBJ_DECLARE(
  reduce_310_Mul_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#223 */
AI_ARRAY_OBJ_DECLARE(
  reduce_310_Mul_0_0_eltwise_311_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 100, AI_STATIC)

/* Array#224 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_311_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 100, AI_STATIC)

/* Array#225 */
AI_ARRAY_OBJ_DECLARE(
  nl_312_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 100, AI_STATIC)

/* Array#226 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_313_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#227 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_315_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#228 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_316_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#229 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_314_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#230 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_317_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#231 */
AI_ARRAY_OBJ_DECLARE(
  gemm_326_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 51200, AI_STATIC)

/* Array#232 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_328_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 51200, AI_STATIC)

/* Array#233 */
AI_ARRAY_OBJ_DECLARE(
  gemm_337_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#234 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_339_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#235 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_340_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#236 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_340_0_0_reduce_341_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#237 */
AI_ARRAY_OBJ_DECLARE(
  reduce_341_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#238 */
AI_ARRAY_OBJ_DECLARE(
  reduce_341_Mul_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#239 */
AI_ARRAY_OBJ_DECLARE(
  reduce_341_Mul_0_1_eltwise_342_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 100, AI_STATIC)

/* Array#240 */
AI_ARRAY_OBJ_DECLARE(
  reduce_341_Mul_0_1_eltwise_342_conversion_0_1_eltwise_342_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#241 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_342_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#242 */
AI_ARRAY_OBJ_DECLARE(
  reduce_343_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#243 */
AI_ARRAY_OBJ_DECLARE(
  reduce_343_Mul_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#244 */
AI_ARRAY_OBJ_DECLARE(
  reduce_343_Mul_0_0_eltwise_344_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 100, AI_STATIC)

/* Array#245 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_344_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 100, AI_STATIC)

/* Array#246 */
AI_ARRAY_OBJ_DECLARE(
  nl_345_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 100, AI_STATIC)

/* Array#247 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_346_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#248 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_348_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#249 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_349_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#250 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_347_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#251 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_350_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#252 */
AI_ARRAY_OBJ_DECLARE(
  gemm_359_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 6600, AI_STATIC)

/* Array#253 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_361_output_array, AI_ARRAY_FORMAT_S8|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 6600, AI_STATIC)

/* Array#254 */
AI_ARRAY_OBJ_DECLARE(
  gemm_293_bias_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1, AI_STATIC)

/* Array#255 */
AI_ARRAY_OBJ_DECLARE(
  gemm_285_bias_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1, AI_STATIC)

/* Array#256 */
AI_ARRAY_OBJ_DECLARE(
  gemm_206_bias_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1, AI_STATIC)

/* Array#257 */
AI_ARRAY_OBJ_DECLARE(
  gemm_198_bias_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1, AI_STATIC)

/* Array#258 */
AI_ARRAY_OBJ_DECLARE(
  gemm_119_bias_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1, AI_STATIC)

/* Array#259 */
AI_ARRAY_OBJ_DECLARE(
  gemm_111_bias_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1, AI_STATIC)

/* Array#260 */
AI_ARRAY_OBJ_DECLARE(
  gemm_32_bias_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1, AI_STATIC)

/* Array#261 */
AI_ARRAY_OBJ_DECLARE(
  gemm_24_bias_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1, AI_STATIC)

/* Array#262 */
AI_ARRAY_OBJ_DECLARE(
  tiny_bert_generator_tf_math_multiply_Mul_y_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1, AI_STATIC)

/* Array#263 */
AI_ARRAY_OBJ_DECLARE(
  encoder_layer_multi_head_attention_8_dense_49_bias_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#264 */
AI_ARRAY_OBJ_DECLARE(
  encoder_layer_multi_head_attention_8_dense_50_bias_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#265 */
AI_ARRAY_OBJ_DECLARE(
  tiny_bert_generator_encoder_layer_3_multi_head_attention_11_truedivtiny_bert_generator_encoder_layer_3_multi_head_attention_11_Sqrt_4D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1, AI_STATIC)

/* Array#266 */
AI_ARRAY_OBJ_DECLARE(
  encoder_layer_multi_head_attention_8_dense_51_bias_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#267 */
AI_ARRAY_OBJ_DECLARE(
  encoder_layer_multi_head_attention_8_dense_52_bias_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#268 */
AI_ARRAY_OBJ_DECLARE(
  tiny_bert_generator_encoder_layer_layer_normalization_16_batchnorm_add_y_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1, AI_STATIC)

/* Array#269 */
AI_ARRAY_OBJ_DECLARE(
  encoder_layer_layer_normalization_16_gamma_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#270 */
AI_ARRAY_OBJ_DECLARE(
  encoder_layer_layer_normalization_16_beta_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#271 */
AI_ARRAY_OBJ_DECLARE(
  dense_53_bias_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 512, AI_STATIC)

/* Array#272 */
AI_ARRAY_OBJ_DECLARE(
  dense_54_bias_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#273 */
AI_ARRAY_OBJ_DECLARE(
  encoder_layer_layer_normalization_17_gamma_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#274 */
AI_ARRAY_OBJ_DECLARE(
  encoder_layer_layer_normalization_17_beta_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#275 */
AI_ARRAY_OBJ_DECLARE(
  encoder_layer_1_multi_head_attention_9_dense_55_bias_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#276 */
AI_ARRAY_OBJ_DECLARE(
  encoder_layer_1_multi_head_attention_9_dense_56_bias_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#277 */
AI_ARRAY_OBJ_DECLARE(
  encoder_layer_1_multi_head_attention_9_dense_57_bias_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#278 */
AI_ARRAY_OBJ_DECLARE(
  encoder_layer_1_multi_head_attention_9_dense_58_bias_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#279 */
AI_ARRAY_OBJ_DECLARE(
  encoder_layer_1_layer_normalization_18_gamma_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#280 */
AI_ARRAY_OBJ_DECLARE(
  encoder_layer_1_layer_normalization_18_beta_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#281 */
AI_ARRAY_OBJ_DECLARE(
  dense_59_bias_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 512, AI_STATIC)

/* Array#282 */
AI_ARRAY_OBJ_DECLARE(
  dense_60_bias_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#283 */
AI_ARRAY_OBJ_DECLARE(
  encoder_layer_1_layer_normalization_19_gamma_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#284 */
AI_ARRAY_OBJ_DECLARE(
  encoder_layer_1_layer_normalization_19_beta_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#285 */
AI_ARRAY_OBJ_DECLARE(
  encoder_layer_2_multi_head_attention_10_dense_61_bias_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#286 */
AI_ARRAY_OBJ_DECLARE(
  encoder_layer_2_multi_head_attention_10_dense_62_bias_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#287 */
AI_ARRAY_OBJ_DECLARE(
  encoder_layer_2_multi_head_attention_10_dense_63_bias_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#288 */
AI_ARRAY_OBJ_DECLARE(
  encoder_layer_2_multi_head_attention_10_dense_64_bias_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#289 */
AI_ARRAY_OBJ_DECLARE(
  encoder_layer_2_layer_normalization_20_gamma_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#290 */
AI_ARRAY_OBJ_DECLARE(
  encoder_layer_2_layer_normalization_20_beta_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#291 */
AI_ARRAY_OBJ_DECLARE(
  dense_65_bias_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 512, AI_STATIC)

/* Array#292 */
AI_ARRAY_OBJ_DECLARE(
  dense_66_bias_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#293 */
AI_ARRAY_OBJ_DECLARE(
  encoder_layer_2_layer_normalization_21_gamma_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#294 */
AI_ARRAY_OBJ_DECLARE(
  encoder_layer_2_layer_normalization_21_beta_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#295 */
AI_ARRAY_OBJ_DECLARE(
  encoder_layer_3_multi_head_attention_11_dense_67_bias_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#296 */
AI_ARRAY_OBJ_DECLARE(
  encoder_layer_3_multi_head_attention_11_dense_68_bias_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#297 */
AI_ARRAY_OBJ_DECLARE(
  encoder_layer_3_multi_head_attention_11_dense_69_bias_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#298 */
AI_ARRAY_OBJ_DECLARE(
  encoder_layer_3_multi_head_attention_11_dense_70_bias_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#299 */
AI_ARRAY_OBJ_DECLARE(
  encoder_layer_3_layer_normalization_22_gamma_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#300 */
AI_ARRAY_OBJ_DECLARE(
  encoder_layer_3_layer_normalization_22_beta_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#301 */
AI_ARRAY_OBJ_DECLARE(
  dense_71_bias_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 512, AI_STATIC)

/* Array#302 */
AI_ARRAY_OBJ_DECLARE(
  dense_72_bias_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#303 */
AI_ARRAY_OBJ_DECLARE(
  encoder_layer_3_layer_normalization_23_gamma_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#304 */
AI_ARRAY_OBJ_DECLARE(
  encoder_layer_3_layer_normalization_23_beta_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#305 */
AI_ARRAY_OBJ_DECLARE(
  dense_24_bias_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 66, AI_STATIC)

/* Array#306 */
AI_ARRAY_OBJ_DECLARE(
  tiny_bert_generator_tf___operators___add_y_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#307 */
AI_ARRAY_OBJ_DECLARE(
  embedding_embeddings_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16896, AI_STATIC)

/* Array#308 */
AI_ARRAY_OBJ_DECLARE(
  gemm_27_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 65536, AI_STATIC)

/* Array#309 */
AI_ARRAY_OBJ_DECLARE(
  gemm_27_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 256, AI_STATIC)

/* Array#310 */
AI_ARRAY_OBJ_DECLARE(
  gemm_19_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 65536, AI_STATIC)

/* Array#311 */
AI_ARRAY_OBJ_DECLARE(
  gemm_14_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 65536, AI_STATIC)

/* Array#312 */
AI_ARRAY_OBJ_DECLARE(
  gemm_43_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 65536, AI_STATIC)

/* Array#313 */
AI_ARRAY_OBJ_DECLARE(
  reduce_47_Mul_scale_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)

/* Array#314 */
AI_ARRAY_OBJ_DECLARE(
  reduce_47_Mul_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)

/* Array#315 */
AI_ARRAY_OBJ_DECLARE(
  gemm_65_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 131072, AI_STATIC)

/* Array#316 */
AI_ARRAY_OBJ_DECLARE(
  gemm_65_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 512, AI_STATIC)

/* Array#317 */
AI_ARRAY_OBJ_DECLARE(
  gemm_76_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 131072, AI_STATIC)

/* Array#318 */
AI_ARRAY_OBJ_DECLARE(
  gemm_114_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 65536, AI_STATIC)

/* Array#319 */
AI_ARRAY_OBJ_DECLARE(
  gemm_106_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 65536, AI_STATIC)

/* Array#320 */
AI_ARRAY_OBJ_DECLARE(
  gemm_101_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 65536, AI_STATIC)

/* Array#321 */
AI_ARRAY_OBJ_DECLARE(
  gemm_130_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 65536, AI_STATIC)

/* Array#322 */
AI_ARRAY_OBJ_DECLARE(
  gemm_152_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 131072, AI_STATIC)

/* Array#323 */
AI_ARRAY_OBJ_DECLARE(
  gemm_163_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 131072, AI_STATIC)

/* Array#324 */
AI_ARRAY_OBJ_DECLARE(
  gemm_201_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 65536, AI_STATIC)

/* Array#325 */
AI_ARRAY_OBJ_DECLARE(
  gemm_193_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 65536, AI_STATIC)

/* Array#326 */
AI_ARRAY_OBJ_DECLARE(
  gemm_188_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 65536, AI_STATIC)

/* Array#327 */
AI_ARRAY_OBJ_DECLARE(
  gemm_217_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 65536, AI_STATIC)

/* Array#328 */
AI_ARRAY_OBJ_DECLARE(
  gemm_239_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 131072, AI_STATIC)

/* Array#329 */
AI_ARRAY_OBJ_DECLARE(
  gemm_250_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 131072, AI_STATIC)

/* Array#330 */
AI_ARRAY_OBJ_DECLARE(
  gemm_288_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 65536, AI_STATIC)

/* Array#331 */
AI_ARRAY_OBJ_DECLARE(
  gemm_280_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 65536, AI_STATIC)

/* Array#332 */
AI_ARRAY_OBJ_DECLARE(
  gemm_275_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 65536, AI_STATIC)

/* Array#333 */
AI_ARRAY_OBJ_DECLARE(
  gemm_304_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 65536, AI_STATIC)

/* Array#334 */
AI_ARRAY_OBJ_DECLARE(
  gemm_326_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 131072, AI_STATIC)

/* Array#335 */
AI_ARRAY_OBJ_DECLARE(
  gemm_337_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 131072, AI_STATIC)

/* Array#336 */
AI_ARRAY_OBJ_DECLARE(
  gemm_359_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16896, AI_STATIC)

/* Array#337 */
AI_ARRAY_OBJ_DECLARE(
  gemm_359_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 66, AI_STATIC)

/* Array#338 */
AI_ARRAY_OBJ_DECLARE(
  gemm_27_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1024, AI_STATIC)

/* Array#339 */
AI_ARRAY_OBJ_DECLARE(
  gemm_19_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1024, AI_STATIC)

/* Array#340 */
AI_ARRAY_OBJ_DECLARE(
  gemm_14_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1024, AI_STATIC)

/* Array#341 */
AI_ARRAY_OBJ_DECLARE(
  nl_26_scratch0_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 256, AI_STATIC)

/* Array#342 */
AI_ARRAY_OBJ_DECLARE(
  gemm_43_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1024, AI_STATIC)

/* Array#343 */
AI_ARRAY_OBJ_DECLARE(
  gemm_65_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1024, AI_STATIC)

/* Array#344 */
AI_ARRAY_OBJ_DECLARE(
  gemm_76_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2048, AI_STATIC)

/* Array#345 */
AI_ARRAY_OBJ_DECLARE(
  gemm_114_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1024, AI_STATIC)

/* Array#346 */
AI_ARRAY_OBJ_DECLARE(
  gemm_106_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1024, AI_STATIC)

/* Array#347 */
AI_ARRAY_OBJ_DECLARE(
  gemm_101_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1024, AI_STATIC)

/* Array#348 */
AI_ARRAY_OBJ_DECLARE(
  nl_113_scratch0_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 256, AI_STATIC)

/* Array#349 */
AI_ARRAY_OBJ_DECLARE(
  gemm_130_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1024, AI_STATIC)

/* Array#350 */
AI_ARRAY_OBJ_DECLARE(
  gemm_152_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1024, AI_STATIC)

/* Array#351 */
AI_ARRAY_OBJ_DECLARE(
  gemm_163_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2048, AI_STATIC)

/* Array#352 */
AI_ARRAY_OBJ_DECLARE(
  gemm_201_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1024, AI_STATIC)

/* Array#353 */
AI_ARRAY_OBJ_DECLARE(
  gemm_193_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1024, AI_STATIC)

/* Array#354 */
AI_ARRAY_OBJ_DECLARE(
  gemm_188_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1024, AI_STATIC)

/* Array#355 */
AI_ARRAY_OBJ_DECLARE(
  nl_200_scratch0_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 256, AI_STATIC)

/* Array#356 */
AI_ARRAY_OBJ_DECLARE(
  gemm_217_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1024, AI_STATIC)

/* Array#357 */
AI_ARRAY_OBJ_DECLARE(
  gemm_239_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1024, AI_STATIC)

/* Array#358 */
AI_ARRAY_OBJ_DECLARE(
  gemm_250_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2048, AI_STATIC)

/* Array#359 */
AI_ARRAY_OBJ_DECLARE(
  gemm_288_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1024, AI_STATIC)

/* Array#360 */
AI_ARRAY_OBJ_DECLARE(
  gemm_280_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1024, AI_STATIC)

/* Array#361 */
AI_ARRAY_OBJ_DECLARE(
  gemm_275_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1024, AI_STATIC)

/* Array#362 */
AI_ARRAY_OBJ_DECLARE(
  nl_287_scratch0_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 256, AI_STATIC)

/* Array#363 */
AI_ARRAY_OBJ_DECLARE(
  gemm_304_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1024, AI_STATIC)

/* Array#364 */
AI_ARRAY_OBJ_DECLARE(
  gemm_326_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1024, AI_STATIC)

/* Array#365 */
AI_ARRAY_OBJ_DECLARE(
  gemm_337_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2048, AI_STATIC)

/* Array#366 */
AI_ARRAY_OBJ_DECLARE(
  gemm_359_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1024, AI_STATIC)

/**  Array metadata declarations section  *************************************/
/* Int quant #0 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(dense_24_bias_3D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(4.466792961466126e-05f),
    AI_PACK_INTQ_ZP(-3)))

/* Int quant #1 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(dense_53_bias_3D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(4.1335162677569315e-05f),
    AI_PACK_INTQ_ZP(-3)))

/* Int quant #2 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(dense_54_bias_3D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(4.111100497539155e-05f),
    AI_PACK_INTQ_ZP(-2)))

/* Int quant #3 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(dense_59_bias_3D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(4.2116713302675635e-05f),
    AI_PACK_INTQ_ZP(-2)))

/* Int quant #4 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(dense_60_bias_3D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(4.269059354555793e-05f),
    AI_PACK_INTQ_ZP(-4)))

/* Int quant #5 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(dense_65_bias_3D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(3.9821523387217894e-05f),
    AI_PACK_INTQ_ZP(-5)))

/* Int quant #6 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(dense_66_bias_3D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(4.224567601340823e-05f),
    AI_PACK_INTQ_ZP(-5)))

/* Int quant #7 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(dense_71_bias_3D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(4.116924537811428e-05f),
    AI_PACK_INTQ_ZP(-10)))

/* Int quant #8 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(dense_72_bias_3D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(4.1336214053444564e-05f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #9 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_103_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.027497505769133568f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #10 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_108_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02919275313615799f),
    AI_PACK_INTQ_ZP(-3)))

/* Int quant #11 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_112_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02483091689646244f),
    AI_PACK_INTQ_ZP(-7)))

/* Int quant #12 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_116_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.035226888954639435f),
    AI_PACK_INTQ_ZP(-6)))

/* Int quant #13 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_132_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03893619775772095f),
    AI_PACK_INTQ_ZP(1)))

/* Int quant #14 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_133_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.05572595074772835f),
    AI_PACK_INTQ_ZP(5)))

/* Int quant #15 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_137_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.023076320067048073f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #16 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_139_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0016889177495613694f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #17 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_140_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.023454545065760612f),
    AI_PACK_INTQ_ZP(6)))

/* Int quant #18 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_141_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00036302744410932064f),
    AI_PACK_INTQ_ZP(127)))

/* Int quant #19 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_142_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00038352468982338905f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #20 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_143_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02347411960363388f),
    AI_PACK_INTQ_ZP(2)))

/* Int quant #21 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_154_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.014156094752252102f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #22 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_165_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.024212762713432312f),
    AI_PACK_INTQ_ZP(1)))

/* Int quant #23 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_166_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04126435145735741f),
    AI_PACK_INTQ_ZP(9)))

/* Int quant #24 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_16_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.029773792251944542f),
    AI_PACK_INTQ_ZP(6)))

/* Int quant #25 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_170_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.01395377516746521f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #26 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_172_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.002164098434150219f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #27 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_173_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.022344281896948814f),
    AI_PACK_INTQ_ZP(9)))

/* Int quant #28 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_174_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00017833220772445202f),
    AI_PACK_INTQ_ZP(127)))

/* Int quant #29 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_175_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00020069409220013767f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #30 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_176_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.022359047085046768f),
    AI_PACK_INTQ_ZP(7)))

/* Int quant #31 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_190_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.026544129475951195f),
    AI_PACK_INTQ_ZP(-3)))

/* Int quant #32 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_195_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.025259431451559067f),
    AI_PACK_INTQ_ZP(-18)))

/* Int quant #33 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_199_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.010870122350752354f),
    AI_PACK_INTQ_ZP(-47)))

/* Int quant #34 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_1_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.006879668682813644f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #35 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_203_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.027622293680906296f),
    AI_PACK_INTQ_ZP(-9)))

/* Int quant #36 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_219_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.039027974009513855f),
    AI_PACK_INTQ_ZP(14)))

/* Int quant #37 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_21_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03217128664255142f),
    AI_PACK_INTQ_ZP(-7)))

/* Int quant #38 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_220_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.051184456795454025f),
    AI_PACK_INTQ_ZP(8)))

/* Int quant #39 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_224_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02278527244925499f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #40 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_226_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0016507772961631417f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #41 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_227_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.021417858079075813f),
    AI_PACK_INTQ_ZP(8)))

/* Int quant #42 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_228_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(9.941380267264321e-05f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #43 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_229_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00012099038576707244f),
    AI_PACK_INTQ_ZP(127)))

/* Int quant #44 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_230_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.021451756358146667f),
    AI_PACK_INTQ_ZP(9)))

/* Int quant #45 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_241_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.012663066387176514f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #46 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_252_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02421327866613865f),
    AI_PACK_INTQ_ZP(-9)))

/* Int quant #47 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_253_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03836870938539505f),
    AI_PACK_INTQ_ZP(-10)))

/* Int quant #48 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_257_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.01250156294554472f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #49 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_259_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.002229431876912713f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #50 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_25_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.035693053156137466f),
    AI_PACK_INTQ_ZP(-3)))

/* Int quant #51 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_260_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02167745679616928f),
    AI_PACK_INTQ_ZP(-11)))

/* Int quant #52 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_261_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00010846184159163386f),
    AI_PACK_INTQ_ZP(127)))

/* Int quant #53 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_262_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0001305954356212169f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #54 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_263_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.021717794239521027f),
    AI_PACK_INTQ_ZP(-12)))

/* Int quant #55 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_277_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.022357836365699768f),
    AI_PACK_INTQ_ZP(-12)))

/* Int quant #56 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_282_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02295767143368721f),
    AI_PACK_INTQ_ZP(-2)))

/* Int quant #57 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_286_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.008026223629713058f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #58 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_290_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.029577750712633133f),
    AI_PACK_INTQ_ZP(5)))

/* Int quant #59 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_29_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03430883586406708f),
    AI_PACK_INTQ_ZP(-12)))

/* Int quant #60 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_2_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.014721683226525784f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #61 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_306_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.044305793941020966f),
    AI_PACK_INTQ_ZP(1)))

/* Int quant #62 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_307_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.05743250623345375f),
    AI_PACK_INTQ_ZP(-10)))

/* Int quant #63 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_311_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.025198256596922874f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #64 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_313_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0015597976744174957f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #65 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_314_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.022806961089372635f),
    AI_PACK_INTQ_ZP(-10)))

/* Int quant #66 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_315_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00017056061187759042f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #67 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_316_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00019198123482055962f),
    AI_PACK_INTQ_ZP(127)))

/* Int quant #68 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_317_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02284243516623974f),
    AI_PACK_INTQ_ZP(-8)))

/* Int quant #69 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_328_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.011848527938127518f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #70 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_339_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.025709152221679688f),
    AI_PACK_INTQ_ZP(7)))

/* Int quant #71 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_340_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.040775690227746964f),
    AI_PACK_INTQ_ZP(-3)))

/* Int quant #72 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_344_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.014830291271209717f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #73 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_346_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00202898308634758f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #74 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_347_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.020977532491087914f),
    AI_PACK_INTQ_ZP(-3)))

/* Int quant #75 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_348_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(1.361213162454078e-05f),
    AI_PACK_INTQ_ZP(127)))

/* Int quant #76 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_349_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(4.2990002839360386e-05f),
    AI_PACK_INTQ_ZP(-67)))

/* Int quant #77 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_350_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02098384127020836f),
    AI_PACK_INTQ_ZP(-3)))

/* Int quant #78 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_361_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03096124902367592f),
    AI_PACK_INTQ_ZP(24)))

/* Int quant #79 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_45_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.024682801216840744f),
    AI_PACK_INTQ_ZP(-5)))

/* Int quant #80 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_46_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03573121502995491f),
    AI_PACK_INTQ_ZP(-12)))

/* Int quant #81 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_50_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.008662102743983269f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #82 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_52_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0032246485352516174f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #83 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_53_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02757403254508972f),
    AI_PACK_INTQ_ZP(-10)))

/* Int quant #84 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_54_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0017649562796577811f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #85 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_55_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0017831810982897878f),
    AI_PACK_INTQ_ZP(127)))

/* Int quant #86 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_56_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.027484649792313576f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #87 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_67_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.014237361960113049f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #88 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_78_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03377589210867882f),
    AI_PACK_INTQ_ZP(-15)))

/* Int quant #89 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_79_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.049465011805295944f),
    AI_PACK_INTQ_ZP(-12)))

/* Int quant #90 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_83_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.016249820590019226f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #91 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_85_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0022003024350851774f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #92 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_86_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02544489875435829f),
    AI_PACK_INTQ_ZP(-14)))

/* Int quant #93 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_87_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0002586648042779416f),
    AI_PACK_INTQ_ZP(-24)))

/* Int quant #94 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_88_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0002992201189044863f),
    AI_PACK_INTQ_ZP(19)))

/* Int quant #95 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_89_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.025484368205070496f),
    AI_PACK_INTQ_ZP(-14)))

/* Int quant #96 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(embedding_embeddings_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0004299792926758528f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #97 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(encoder_layer_1_layer_normalization_18_beta_3D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(4.3411993829067796e-05f),
    AI_PACK_INTQ_ZP(1)))

/* Int quant #98 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(encoder_layer_1_layer_normalization_18_gamma_3D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00394402164965868f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #99 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(encoder_layer_1_layer_normalization_19_beta_3D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(4.3014108086936176e-05f),
    AI_PACK_INTQ_ZP(-6)))

/* Int quant #100 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(encoder_layer_1_layer_normalization_19_gamma_3D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.003943675197660923f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #101 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(encoder_layer_1_multi_head_attention_9_dense_55_bias_3D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(3.901650052284822e-05f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #102 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(encoder_layer_1_multi_head_attention_9_dense_56_bias_3D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(3.921568403342235e-09f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #103 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(encoder_layer_1_multi_head_attention_9_dense_57_bias_3D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(4.1431743738939986e-05f),
    AI_PACK_INTQ_ZP(-2)))

/* Int quant #104 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(encoder_layer_1_multi_head_attention_9_dense_58_bias_3D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(4.3160125642316416e-05f),
    AI_PACK_INTQ_ZP(-4)))

/* Int quant #105 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(encoder_layer_2_layer_normalization_20_beta_3D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(4.323077519075014e-05f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #106 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(encoder_layer_2_layer_normalization_20_gamma_3D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.003944822587072849f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #107 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(encoder_layer_2_layer_normalization_21_beta_3D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(4.269382043275982e-05f),
    AI_PACK_INTQ_ZP(-5)))

/* Int quant #108 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(encoder_layer_2_layer_normalization_21_gamma_3D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.003943596966564655f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #109 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(encoder_layer_2_multi_head_attention_10_dense_61_bias_3D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(3.8719404983567074e-05f),
    AI_PACK_INTQ_ZP(-7)))

/* Int quant #110 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(encoder_layer_2_multi_head_attention_10_dense_62_bias_3D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(3.921568403342235e-09f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #111 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(encoder_layer_2_multi_head_attention_10_dense_63_bias_3D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(4.231610000715591e-05f),
    AI_PACK_INTQ_ZP(-3)))

/* Int quant #112 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(encoder_layer_2_multi_head_attention_10_dense_64_bias_3D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(4.194758730591275e-05f),
    AI_PACK_INTQ_ZP(-5)))

/* Int quant #113 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(encoder_layer_3_layer_normalization_22_beta_3D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(4.2831314203795046e-05f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #114 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(encoder_layer_3_layer_normalization_22_gamma_3D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.003942808602005243f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #115 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(encoder_layer_3_layer_normalization_23_beta_3D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(3.7187124689808115e-05f),
    AI_PACK_INTQ_ZP(-4)))

/* Int quant #116 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(encoder_layer_3_layer_normalization_23_gamma_3D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0039366912096738815f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #117 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(encoder_layer_3_multi_head_attention_11_dense_67_bias_3D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(3.6314420867711306e-05f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #118 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(encoder_layer_3_multi_head_attention_11_dense_68_bias_3D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(3.921568403342235e-09f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #119 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(encoder_layer_3_multi_head_attention_11_dense_69_bias_3D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(4.402088961796835e-05f),
    AI_PACK_INTQ_ZP(-2)))

/* Int quant #120 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(encoder_layer_3_multi_head_attention_11_dense_70_bias_3D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(4.176698348601349e-05f),
    AI_PACK_INTQ_ZP(-2)))

/* Int quant #121 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(encoder_layer_layer_normalization_16_beta_3D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(4.3017822463298216e-05f),
    AI_PACK_INTQ_ZP(4)))

/* Int quant #122 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(encoder_layer_layer_normalization_16_gamma_3D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.003942896146327257f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #123 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(encoder_layer_layer_normalization_17_beta_3D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(4.065815301146358e-05f),
    AI_PACK_INTQ_ZP(-2)))

/* Int quant #124 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(encoder_layer_layer_normalization_17_gamma_3D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.003943859599530697f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #125 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(encoder_layer_multi_head_attention_8_dense_49_bias_3D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(3.929867307306267e-05f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #126 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(encoder_layer_multi_head_attention_8_dense_50_bias_3D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(3.921568403342235e-09f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #127 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(encoder_layer_multi_head_attention_8_dense_51_bias_3D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(4.2190804379060864e-05f),
    AI_PACK_INTQ_ZP(-4)))

/* Int quant #128 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(encoder_layer_multi_head_attention_8_dense_52_bias_3D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(4.1933566535590217e-05f),
    AI_PACK_INTQ_ZP(5)))

/* Int quant #129 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gather_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0004299792926758528f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #130 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_101_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.027489541098475456f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #131 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_101_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.000890115275979042f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #132 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_106_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02919275313615799f),
    AI_PACK_INTQ_ZP(-3)))

/* Int quant #133 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_106_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0008928222814574838f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #134 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_111_0_0_eltwise_112_conversion_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.19864733517169952f),
    AI_PACK_INTQ_ZP(-7)))

/* Int quant #135 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_111_bias_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.19864733517169952f),
    AI_PACK_INTQ_ZP(-7)))

/* Int quant #136 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_114_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.035202253609895706f),
    AI_PACK_INTQ_ZP(-6)))

/* Int quant #137 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_114_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0008928148890845478f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #138 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_119_0_0_transpose_120_conversion_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.029204588383436203f),
    AI_PACK_INTQ_ZP(-8)))

/* Int quant #139 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_119_bias_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.029204588383436203f),
    AI_PACK_INTQ_ZP(-8)))

/* Int quant #140 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_130_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03890177607536316f),
    AI_PACK_INTQ_ZP(1)))

/* Int quant #141 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_130_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0008948758477345109f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #142 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_14_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02974001131951809f),
    AI_PACK_INTQ_ZP(6)))

/* Int quant #143 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_14_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0008933691424317658f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #144 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_152_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.026434050872921944f),
    AI_PACK_INTQ_ZP(-9)))

/* Int quant #145 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_152_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0007365959463641047f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #146 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_163_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.024183716624975204f),
    AI_PACK_INTQ_ZP(1)))

/* Int quant #147 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_163_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0007384680793620646f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #148 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_188_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.026516810059547424f),
    AI_PACK_INTQ_ZP(-3)))

/* Int quant #149 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_188_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0008897301158867776f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #150 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_193_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.025259431451559067f),
    AI_PACK_INTQ_ZP(-18)))

/* Int quant #151 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_193_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0008911079494282603f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #152 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_198_0_0_eltwise_199_conversion_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.08696097880601883f),
    AI_PACK_INTQ_ZP(-47)))

/* Int quant #153 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_198_bias_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.08696097880601883f),
    AI_PACK_INTQ_ZP(-47)))

/* Int quant #154 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_19_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03217128664255142f),
    AI_PACK_INTQ_ZP(-7)))

/* Int quant #155 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_19_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0008908897871151567f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #156 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_201_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.027585014700889587f),
    AI_PACK_INTQ_ZP(-9)))

/* Int quant #157 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_201_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00089347327593714f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #158 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_206_0_0_transpose_207_conversion_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.025932038202881813f),
    AI_PACK_INTQ_ZP(-10)))

/* Int quant #159 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_206_bias_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.025932038202881813f),
    AI_PACK_INTQ_ZP(-10)))

/* Int quant #160 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_217_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03899608552455902f),
    AI_PACK_INTQ_ZP(14)))

/* Int quant #161 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_217_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0008915417129173875f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #162 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_239_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.021786293014883995f),
    AI_PACK_INTQ_ZP(-21)))

/* Int quant #163 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_239_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0007375386194325984f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #164 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_24_0_0_eltwise_25_conversion_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.28554442524909973f),
    AI_PACK_INTQ_ZP(-3)))

/* Int quant #165 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_24_bias_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.28554442524909973f),
    AI_PACK_INTQ_ZP(-3)))

/* Int quant #166 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_250_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.024171186611056328f),
    AI_PACK_INTQ_ZP(-9)))

/* Int quant #167 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_250_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0007373551488853991f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #168 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_275_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02235664613544941f),
    AI_PACK_INTQ_ZP(-12)))

/* Int quant #169 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_275_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0008876381325535476f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #170 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_27_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03429887071251869f),
    AI_PACK_INTQ_ZP(-12)))

/* Int quant #171 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_27_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0008936116937547922f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #172 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_280_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02295767143368721f),
    AI_PACK_INTQ_ZP(-2)))

/* Int quant #173 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_280_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0008867759024724364f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #174 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_285_0_0_eltwise_286_conversion_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.06420978903770447f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #175 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_285_bias_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.06420978903770447f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #176 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_288_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02954975515604019f),
    AI_PACK_INTQ_ZP(5)))

/* Int quant #177 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_288_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0008923957939259708f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #178 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_293_0_0_transpose_294_conversion_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02905571088194847f),
    AI_PACK_INTQ_ZP(5)))

/* Int quant #179 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_293_bias_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02905571088194847f),
    AI_PACK_INTQ_ZP(5)))

/* Int quant #180 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_304_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04426678270101547f),
    AI_PACK_INTQ_ZP(1)))

/* Int quant #181 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_304_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0008902641711756587f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #182 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_326_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02120722085237503f),
    AI_PACK_INTQ_ZP(-15)))

/* Int quant #183 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_326_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0007374093402177095f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #184 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_32_0_0_transpose_33_conversion_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.019155630841851234f),
    AI_PACK_INTQ_ZP(-10)))

/* Int quant #185 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_32_bias_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.019155630841851234f),
    AI_PACK_INTQ_ZP(-10)))

/* Int quant #186 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_337_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02567179873585701f),
    AI_PACK_INTQ_ZP(7)))

/* Int quant #187 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_337_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0007366304635070264f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #188 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_359_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03094756416976452f),
    AI_PACK_INTQ_ZP(24)))

/* Int quant #189 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_359_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0011163227027282119f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #190 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_43_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.024654537439346313f),
    AI_PACK_INTQ_ZP(-5)))

/* Int quant #191 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_43_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0008964139851741493f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #192 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_65_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.027729583904147148f),
    AI_PACK_INTQ_ZP(-4)))

/* Int quant #193 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_65_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0007367486250586808f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #194 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_76_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03373799845576286f),
    AI_PACK_INTQ_ZP(-15)))

/* Int quant #195 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_76_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0007399347377941012f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #196 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_113_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00390625f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #197 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_138_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.001679302891716361f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #198 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_171_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0021519672591239214f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #199 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_200_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00390625f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #200 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_225_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00164104625582695f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #201 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_258_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0022169784642755985f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #202 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_26_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00390625f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #203 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_287_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00390625f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #204 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_312_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0015513950493186712f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #205 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_345_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0020211886148899794f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #206 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_51_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.003207205794751644f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #207 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_84_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0021878660190850496f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #208 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(reduce_134_Mul_0_1_eltwise_135_conversion_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0008494089706800878f),
    AI_PACK_INTQ_ZP(127)))

/* Int quant #209 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(reduce_136_Mul_0_0_eltwise_137_conversion_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.023076316341757774f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #210 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(reduce_167_Mul_0_1_eltwise_168_conversion_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0003288733714725822f),
    AI_PACK_INTQ_ZP(127)))

/* Int quant #211 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(reduce_169_Mul_0_0_eltwise_170_conversion_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.013953771442174911f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #212 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(reduce_221_Mul_0_1_eltwise_222_conversion_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0002364956890232861f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #213 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(reduce_223_Mul_0_0_eltwise_224_conversion_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02278526872396469f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #214 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(reduce_254_Mul_0_1_eltwise_255_conversion_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00019189475278835744f),
    AI_PACK_INTQ_ZP(127)))

/* Int quant #215 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(reduce_256_Mul_0_0_eltwise_257_conversion_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.012501559220254421f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #216 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(reduce_308_Mul_0_1_eltwise_309_conversion_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00042917064274661243f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #217 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(reduce_310_Mul_0_0_eltwise_311_conversion_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.025198252871632576f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #218 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(reduce_341_Mul_0_1_eltwise_342_conversion_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(2.632921314216219e-05f),
    AI_PACK_INTQ_ZP(127)))

/* Int quant #219 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(reduce_343_Mul_0_0_eltwise_344_conversion_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.014830287545919418f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #220 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(reduce_47_Mul_0_1_eltwise_48_conversion_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.002362731145694852f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #221 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(reduce_49_Mul_0_0_eltwise_50_conversion_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00866209901869297f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #222 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(reduce_80_Mul_0_1_eltwise_81_conversion_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0004984618863090873f),
    AI_PACK_INTQ_ZP(-27)))

/* Int quant #223 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(reduce_82_Mul_0_0_eltwise_83_conversion_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.016249816864728928f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #224 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(tiny_bert_generator_encoder_layer_3_multi_head_attention_11_truedivtiny_bert_generator_encoder_layer_3_multi_head_attention_11_Sqrt_4D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0004901961074210703f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #225 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(tiny_bert_generator_encoder_layer_layer_normalization_16_batchnorm_add_y_3D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(5.882353271147167e-09f),
    AI_PACK_INTQ_ZP(-43)))

/* Int quant #226 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(tiny_bert_generator_tf___operators___add_y_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #227 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(tiny_bert_generator_tf_math_multiply_Mul_y_3D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.062745101749897f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #228 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(transpose_105_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.027497505769133568f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #229 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(transpose_118_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.035226888954639435f),
    AI_PACK_INTQ_ZP(-6)))

/* Int quant #230 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(transpose_120_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.029204588383436203f),
    AI_PACK_INTQ_ZP(-8)))

/* Int quant #231 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(transpose_18_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.029773792251944542f),
    AI_PACK_INTQ_ZP(6)))

/* Int quant #232 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(transpose_192_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.026544129475951195f),
    AI_PACK_INTQ_ZP(-3)))

/* Int quant #233 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(transpose_205_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.027622293680906296f),
    AI_PACK_INTQ_ZP(-9)))

/* Int quant #234 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(transpose_207_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.025932038202881813f),
    AI_PACK_INTQ_ZP(-10)))

/* Int quant #235 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(transpose_279_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.022357836365699768f),
    AI_PACK_INTQ_ZP(-12)))

/* Int quant #236 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(transpose_292_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.029577750712633133f),
    AI_PACK_INTQ_ZP(5)))

/* Int quant #237 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(transpose_294_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02905571088194847f),
    AI_PACK_INTQ_ZP(5)))

/* Int quant #238 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(transpose_31_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03430883586406708f),
    AI_PACK_INTQ_ZP(-12)))

/* Int quant #239 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(transpose_33_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.019155630841851234f),
    AI_PACK_INTQ_ZP(-10)))

/* Int quant #240 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(transpose_bgemm_111_out_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02919275313615799f),
    AI_PACK_INTQ_ZP(-3)))

/* Int quant #241 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(transpose_bgemm_198_out_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.025259431451559067f),
    AI_PACK_INTQ_ZP(-18)))

/* Int quant #242 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(transpose_bgemm_24_out_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03217128664255142f),
    AI_PACK_INTQ_ZP(-7)))

/* Int quant #243 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(transpose_bgemm_285_out_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02295767143368721f),
    AI_PACK_INTQ_ZP(-2)))

/**  Tensor declarations section  *********************************************/
/* Tensor #0 */
AI_TENSOR_OBJ_DECLARE(
  dense_24_bias_3D, AI_STATIC,
  0, 0x1,
  AI_SHAPE_INIT(4, 1, 66, 1, 1), AI_STRIDE_INIT(4, 1, 1, 66, 66),
  1, &dense_24_bias_3D_array, &dense_24_bias_3D_array_intq)

/* Tensor #1 */
AI_TENSOR_OBJ_DECLARE(
  dense_53_bias_3D, AI_STATIC,
  1, 0x1,
  AI_SHAPE_INIT(4, 1, 512, 1, 1), AI_STRIDE_INIT(4, 1, 1, 512, 512),
  1, &dense_53_bias_3D_array, &dense_53_bias_3D_array_intq)

/* Tensor #2 */
AI_TENSOR_OBJ_DECLARE(
  dense_54_bias_3D, AI_STATIC,
  2, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &dense_54_bias_3D_array, &dense_54_bias_3D_array_intq)

/* Tensor #3 */
AI_TENSOR_OBJ_DECLARE(
  dense_59_bias_3D, AI_STATIC,
  3, 0x1,
  AI_SHAPE_INIT(4, 1, 512, 1, 1), AI_STRIDE_INIT(4, 1, 1, 512, 512),
  1, &dense_59_bias_3D_array, &dense_59_bias_3D_array_intq)

/* Tensor #4 */
AI_TENSOR_OBJ_DECLARE(
  dense_60_bias_3D, AI_STATIC,
  4, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &dense_60_bias_3D_array, &dense_60_bias_3D_array_intq)

/* Tensor #5 */
AI_TENSOR_OBJ_DECLARE(
  dense_65_bias_3D, AI_STATIC,
  5, 0x1,
  AI_SHAPE_INIT(4, 1, 512, 1, 1), AI_STRIDE_INIT(4, 1, 1, 512, 512),
  1, &dense_65_bias_3D_array, &dense_65_bias_3D_array_intq)

/* Tensor #6 */
AI_TENSOR_OBJ_DECLARE(
  dense_66_bias_3D, AI_STATIC,
  6, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &dense_66_bias_3D_array, &dense_66_bias_3D_array_intq)

/* Tensor #7 */
AI_TENSOR_OBJ_DECLARE(
  dense_71_bias_3D, AI_STATIC,
  7, 0x1,
  AI_SHAPE_INIT(4, 1, 512, 1, 1), AI_STRIDE_INIT(4, 1, 1, 512, 512),
  1, &dense_71_bias_3D_array, &dense_71_bias_3D_array_intq)

/* Tensor #8 */
AI_TENSOR_OBJ_DECLARE(
  dense_72_bias_3D, AI_STATIC,
  8, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &dense_72_bias_3D_array, &dense_72_bias_3D_array_intq)

/* Tensor #9 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_103_output, AI_STATIC,
  9, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_103_output_array, &eltwise_103_output_array_intq)

/* Tensor #10 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_103_output0, AI_STATIC,
  10, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 4, 100), AI_STRIDE_INIT(4, 1, 1, 64, 256),
  1, &eltwise_103_output_array, &eltwise_103_output_array_intq)

/* Tensor #11 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_108_output, AI_STATIC,
  11, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_108_output_array, &eltwise_108_output_array_intq)

/* Tensor #12 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_108_output0, AI_STATIC,
  12, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 4, 100), AI_STRIDE_INIT(4, 1, 1, 64, 256),
  1, &eltwise_108_output_array, &eltwise_108_output_array_intq)

/* Tensor #13 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_112_output, AI_STATIC,
  13, 0x1,
  AI_SHAPE_INIT(4, 1, 100, 100, 4), AI_STRIDE_INIT(4, 1, 1, 100, 10000),
  1, &eltwise_112_output_array, &eltwise_112_output_array_intq)

/* Tensor #14 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_116_output, AI_STATIC,
  14, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_116_output_array, &eltwise_116_output_array_intq)

/* Tensor #15 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_116_output0, AI_STATIC,
  15, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 4, 100), AI_STRIDE_INIT(4, 1, 1, 64, 256),
  1, &eltwise_116_output_array, &eltwise_116_output_array_intq)

/* Tensor #16 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_132_output, AI_STATIC,
  16, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_132_output_array, &eltwise_132_output_array_intq)

/* Tensor #17 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_133_0_0_reduce_134_conversion_output, AI_STATIC,
  17, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 4, 4, 1024, 1024),
  1, &eltwise_133_0_0_reduce_134_conversion_output_array, NULL)

/* Tensor #18 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_133_output, AI_STATIC,
  18, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_133_output_array, &eltwise_133_output_array_intq)

/* Tensor #19 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_135_output, AI_STATIC,
  19, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 4, 4, 1024, 1024),
  1, &eltwise_135_output_array, NULL)

/* Tensor #20 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_137_output, AI_STATIC,
  20, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &eltwise_137_output_array, &eltwise_137_output_array_intq)

/* Tensor #21 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_139_output, AI_STATIC,
  21, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_139_output_array, &eltwise_139_output_array_intq)

/* Tensor #22 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_140_output, AI_STATIC,
  22, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_140_output_array, &eltwise_140_output_array_intq)

/* Tensor #23 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_141_output, AI_STATIC,
  23, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_141_output_array, &eltwise_141_output_array_intq)

/* Tensor #24 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_142_output, AI_STATIC,
  24, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_142_output_array, &eltwise_142_output_array_intq)

/* Tensor #25 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_143_output, AI_STATIC,
  25, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_143_output_array, &eltwise_143_output_array_intq)

/* Tensor #26 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_154_output, AI_STATIC,
  26, 0x1,
  AI_SHAPE_INIT(4, 1, 512, 1, 100), AI_STRIDE_INIT(4, 1, 1, 512, 512),
  1, &eltwise_154_output_array, &eltwise_154_output_array_intq)

/* Tensor #27 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_165_output, AI_STATIC,
  27, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_165_output_array, &eltwise_165_output_array_intq)

/* Tensor #28 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_166_0_0_reduce_167_conversion_output, AI_STATIC,
  28, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 4, 4, 1024, 1024),
  1, &eltwise_166_0_0_reduce_167_conversion_output_array, NULL)

/* Tensor #29 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_166_output, AI_STATIC,
  29, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_166_output_array, &eltwise_166_output_array_intq)

/* Tensor #30 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_168_output, AI_STATIC,
  30, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 4, 4, 1024, 1024),
  1, &eltwise_168_output_array, NULL)

/* Tensor #31 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_16_output, AI_STATIC,
  31, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_16_output_array, &eltwise_16_output_array_intq)

/* Tensor #32 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_16_output0, AI_STATIC,
  32, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 4, 100), AI_STRIDE_INIT(4, 1, 1, 64, 256),
  1, &eltwise_16_output_array, &eltwise_16_output_array_intq)

/* Tensor #33 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_170_output, AI_STATIC,
  33, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &eltwise_170_output_array, &eltwise_170_output_array_intq)

/* Tensor #34 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_172_output, AI_STATIC,
  34, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_172_output_array, &eltwise_172_output_array_intq)

/* Tensor #35 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_173_output, AI_STATIC,
  35, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_173_output_array, &eltwise_173_output_array_intq)

/* Tensor #36 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_174_output, AI_STATIC,
  36, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_174_output_array, &eltwise_174_output_array_intq)

/* Tensor #37 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_175_output, AI_STATIC,
  37, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_175_output_array, &eltwise_175_output_array_intq)

/* Tensor #38 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_176_output, AI_STATIC,
  38, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_176_output_array, &eltwise_176_output_array_intq)

/* Tensor #39 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_190_output, AI_STATIC,
  39, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_190_output_array, &eltwise_190_output_array_intq)

/* Tensor #40 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_190_output0, AI_STATIC,
  40, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 4, 100), AI_STRIDE_INIT(4, 1, 1, 64, 256),
  1, &eltwise_190_output_array, &eltwise_190_output_array_intq)

/* Tensor #41 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_195_output, AI_STATIC,
  41, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_195_output_array, &eltwise_195_output_array_intq)

/* Tensor #42 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_195_output0, AI_STATIC,
  42, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 4, 100), AI_STRIDE_INIT(4, 1, 1, 64, 256),
  1, &eltwise_195_output_array, &eltwise_195_output_array_intq)

/* Tensor #43 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_199_output, AI_STATIC,
  43, 0x1,
  AI_SHAPE_INIT(4, 1, 100, 100, 4), AI_STRIDE_INIT(4, 1, 1, 100, 10000),
  1, &eltwise_199_output_array, &eltwise_199_output_array_intq)

/* Tensor #44 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_1_output, AI_STATIC,
  44, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_1_output_array, &eltwise_1_output_array_intq)

/* Tensor #45 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_203_output, AI_STATIC,
  45, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_203_output_array, &eltwise_203_output_array_intq)

/* Tensor #46 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_203_output0, AI_STATIC,
  46, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 4, 100), AI_STRIDE_INIT(4, 1, 1, 64, 256),
  1, &eltwise_203_output_array, &eltwise_203_output_array_intq)

/* Tensor #47 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_219_output, AI_STATIC,
  47, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_219_output_array, &eltwise_219_output_array_intq)

/* Tensor #48 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_21_output, AI_STATIC,
  48, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_21_output_array, &eltwise_21_output_array_intq)

/* Tensor #49 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_21_output0, AI_STATIC,
  49, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 4, 100), AI_STRIDE_INIT(4, 1, 1, 64, 256),
  1, &eltwise_21_output_array, &eltwise_21_output_array_intq)

/* Tensor #50 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_220_0_0_reduce_221_conversion_output, AI_STATIC,
  50, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 4, 4, 1024, 1024),
  1, &eltwise_220_0_0_reduce_221_conversion_output_array, NULL)

/* Tensor #51 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_220_output, AI_STATIC,
  51, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_220_output_array, &eltwise_220_output_array_intq)

/* Tensor #52 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_222_output, AI_STATIC,
  52, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 4, 4, 1024, 1024),
  1, &eltwise_222_output_array, NULL)

/* Tensor #53 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_224_output, AI_STATIC,
  53, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &eltwise_224_output_array, &eltwise_224_output_array_intq)

/* Tensor #54 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_226_output, AI_STATIC,
  54, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_226_output_array, &eltwise_226_output_array_intq)

/* Tensor #55 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_227_output, AI_STATIC,
  55, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_227_output_array, &eltwise_227_output_array_intq)

/* Tensor #56 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_228_output, AI_STATIC,
  56, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_228_output_array, &eltwise_228_output_array_intq)

/* Tensor #57 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_229_output, AI_STATIC,
  57, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_229_output_array, &eltwise_229_output_array_intq)

/* Tensor #58 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_230_output, AI_STATIC,
  58, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_230_output_array, &eltwise_230_output_array_intq)

/* Tensor #59 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_241_output, AI_STATIC,
  59, 0x1,
  AI_SHAPE_INIT(4, 1, 512, 1, 100), AI_STRIDE_INIT(4, 1, 1, 512, 512),
  1, &eltwise_241_output_array, &eltwise_241_output_array_intq)

/* Tensor #60 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_252_output, AI_STATIC,
  60, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_252_output_array, &eltwise_252_output_array_intq)

/* Tensor #61 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_253_0_0_reduce_254_conversion_output, AI_STATIC,
  61, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 4, 4, 1024, 1024),
  1, &eltwise_253_0_0_reduce_254_conversion_output_array, NULL)

/* Tensor #62 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_253_output, AI_STATIC,
  62, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_253_output_array, &eltwise_253_output_array_intq)

/* Tensor #63 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_255_output, AI_STATIC,
  63, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 4, 4, 1024, 1024),
  1, &eltwise_255_output_array, NULL)

/* Tensor #64 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_257_output, AI_STATIC,
  64, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &eltwise_257_output_array, &eltwise_257_output_array_intq)

/* Tensor #65 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_259_output, AI_STATIC,
  65, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_259_output_array, &eltwise_259_output_array_intq)

/* Tensor #66 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_25_output, AI_STATIC,
  66, 0x1,
  AI_SHAPE_INIT(4, 1, 100, 100, 4), AI_STRIDE_INIT(4, 1, 1, 100, 10000),
  1, &eltwise_25_output_array, &eltwise_25_output_array_intq)

/* Tensor #67 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_260_output, AI_STATIC,
  67, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_260_output_array, &eltwise_260_output_array_intq)

/* Tensor #68 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_261_output, AI_STATIC,
  68, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_261_output_array, &eltwise_261_output_array_intq)

/* Tensor #69 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_262_output, AI_STATIC,
  69, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_262_output_array, &eltwise_262_output_array_intq)

/* Tensor #70 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_263_output, AI_STATIC,
  70, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_263_output_array, &eltwise_263_output_array_intq)

/* Tensor #71 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_277_output, AI_STATIC,
  71, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_277_output_array, &eltwise_277_output_array_intq)

/* Tensor #72 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_277_output0, AI_STATIC,
  72, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 4, 100), AI_STRIDE_INIT(4, 1, 1, 64, 256),
  1, &eltwise_277_output_array, &eltwise_277_output_array_intq)

/* Tensor #73 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_282_output, AI_STATIC,
  73, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_282_output_array, &eltwise_282_output_array_intq)

/* Tensor #74 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_282_output0, AI_STATIC,
  74, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 4, 100), AI_STRIDE_INIT(4, 1, 1, 64, 256),
  1, &eltwise_282_output_array, &eltwise_282_output_array_intq)

/* Tensor #75 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_286_output, AI_STATIC,
  75, 0x1,
  AI_SHAPE_INIT(4, 1, 100, 100, 4), AI_STRIDE_INIT(4, 1, 1, 100, 10000),
  1, &eltwise_286_output_array, &eltwise_286_output_array_intq)

/* Tensor #76 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_290_output, AI_STATIC,
  76, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_290_output_array, &eltwise_290_output_array_intq)

/* Tensor #77 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_290_output0, AI_STATIC,
  77, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 4, 100), AI_STRIDE_INIT(4, 1, 1, 64, 256),
  1, &eltwise_290_output_array, &eltwise_290_output_array_intq)

/* Tensor #78 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_29_output, AI_STATIC,
  78, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_29_output_array, &eltwise_29_output_array_intq)

/* Tensor #79 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_29_output0, AI_STATIC,
  79, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 4, 100), AI_STRIDE_INIT(4, 1, 1, 64, 256),
  1, &eltwise_29_output_array, &eltwise_29_output_array_intq)

/* Tensor #80 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_2_output, AI_STATIC,
  80, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_2_output_array, &eltwise_2_output_array_intq)

/* Tensor #81 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_306_output, AI_STATIC,
  81, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_306_output_array, &eltwise_306_output_array_intq)

/* Tensor #82 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_307_0_0_reduce_308_conversion_output, AI_STATIC,
  82, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 4, 4, 1024, 1024),
  1, &eltwise_307_0_0_reduce_308_conversion_output_array, NULL)

/* Tensor #83 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_307_output, AI_STATIC,
  83, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_307_output_array, &eltwise_307_output_array_intq)

/* Tensor #84 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_309_output, AI_STATIC,
  84, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 4, 4, 1024, 1024),
  1, &eltwise_309_output_array, NULL)

/* Tensor #85 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_311_output, AI_STATIC,
  85, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &eltwise_311_output_array, &eltwise_311_output_array_intq)

/* Tensor #86 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_313_output, AI_STATIC,
  86, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_313_output_array, &eltwise_313_output_array_intq)

/* Tensor #87 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_314_output, AI_STATIC,
  87, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_314_output_array, &eltwise_314_output_array_intq)

/* Tensor #88 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_315_output, AI_STATIC,
  88, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_315_output_array, &eltwise_315_output_array_intq)

/* Tensor #89 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_316_output, AI_STATIC,
  89, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_316_output_array, &eltwise_316_output_array_intq)

/* Tensor #90 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_317_output, AI_STATIC,
  90, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_317_output_array, &eltwise_317_output_array_intq)

/* Tensor #91 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_328_output, AI_STATIC,
  91, 0x1,
  AI_SHAPE_INIT(4, 1, 512, 1, 100), AI_STRIDE_INIT(4, 1, 1, 512, 512),
  1, &eltwise_328_output_array, &eltwise_328_output_array_intq)

/* Tensor #92 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_339_output, AI_STATIC,
  92, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_339_output_array, &eltwise_339_output_array_intq)

/* Tensor #93 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_340_0_0_reduce_341_conversion_output, AI_STATIC,
  93, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 4, 4, 1024, 1024),
  1, &eltwise_340_0_0_reduce_341_conversion_output_array, NULL)

/* Tensor #94 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_340_output, AI_STATIC,
  94, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_340_output_array, &eltwise_340_output_array_intq)

/* Tensor #95 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_342_output, AI_STATIC,
  95, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 4, 4, 1024, 1024),
  1, &eltwise_342_output_array, NULL)

/* Tensor #96 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_344_output, AI_STATIC,
  96, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &eltwise_344_output_array, &eltwise_344_output_array_intq)

/* Tensor #97 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_346_output, AI_STATIC,
  97, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_346_output_array, &eltwise_346_output_array_intq)

/* Tensor #98 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_347_output, AI_STATIC,
  98, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_347_output_array, &eltwise_347_output_array_intq)

/* Tensor #99 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_348_output, AI_STATIC,
  99, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_348_output_array, &eltwise_348_output_array_intq)

/* Tensor #100 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_349_output, AI_STATIC,
  100, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_349_output_array, &eltwise_349_output_array_intq)

/* Tensor #101 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_350_output, AI_STATIC,
  101, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_350_output_array, &eltwise_350_output_array_intq)

/* Tensor #102 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_361_output, AI_STATIC,
  102, 0x1,
  AI_SHAPE_INIT(4, 1, 66, 1, 100), AI_STRIDE_INIT(4, 1, 1, 66, 66),
  1, &eltwise_361_output_array, &eltwise_361_output_array_intq)

/* Tensor #103 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_45_output, AI_STATIC,
  103, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_45_output_array, &eltwise_45_output_array_intq)

/* Tensor #104 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_46_0_0_reduce_47_conversion_output, AI_STATIC,
  104, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 4, 4, 1024, 1024),
  1, &eltwise_46_0_0_reduce_47_conversion_output_array, NULL)

/* Tensor #105 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_46_output, AI_STATIC,
  105, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_46_output_array, &eltwise_46_output_array_intq)

/* Tensor #106 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_48_output, AI_STATIC,
  106, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 4, 4, 1024, 1024),
  1, &eltwise_48_output_array, NULL)

/* Tensor #107 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_50_output, AI_STATIC,
  107, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &eltwise_50_output_array, &eltwise_50_output_array_intq)

/* Tensor #108 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_52_output, AI_STATIC,
  108, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_52_output_array, &eltwise_52_output_array_intq)

/* Tensor #109 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_53_output, AI_STATIC,
  109, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_53_output_array, &eltwise_53_output_array_intq)

/* Tensor #110 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_54_output, AI_STATIC,
  110, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_54_output_array, &eltwise_54_output_array_intq)

/* Tensor #111 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_55_output, AI_STATIC,
  111, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_55_output_array, &eltwise_55_output_array_intq)

/* Tensor #112 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_56_output, AI_STATIC,
  112, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_56_output_array, &eltwise_56_output_array_intq)

/* Tensor #113 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_67_output, AI_STATIC,
  113, 0x1,
  AI_SHAPE_INIT(4, 1, 512, 1, 100), AI_STRIDE_INIT(4, 1, 1, 512, 512),
  1, &eltwise_67_output_array, &eltwise_67_output_array_intq)

/* Tensor #114 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_78_output, AI_STATIC,
  114, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_78_output_array, &eltwise_78_output_array_intq)

/* Tensor #115 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_79_0_0_reduce_80_conversion_output, AI_STATIC,
  115, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 4, 4, 1024, 1024),
  1, &eltwise_79_0_0_reduce_80_conversion_output_array, NULL)

/* Tensor #116 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_79_output, AI_STATIC,
  116, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_79_output_array, &eltwise_79_output_array_intq)

/* Tensor #117 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_81_output, AI_STATIC,
  117, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 4, 4, 1024, 1024),
  1, &eltwise_81_output_array, NULL)

/* Tensor #118 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_83_output, AI_STATIC,
  118, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &eltwise_83_output_array, &eltwise_83_output_array_intq)

/* Tensor #119 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_85_output, AI_STATIC,
  119, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_85_output_array, &eltwise_85_output_array_intq)

/* Tensor #120 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_86_output, AI_STATIC,
  120, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_86_output_array, &eltwise_86_output_array_intq)

/* Tensor #121 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_87_output, AI_STATIC,
  121, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_87_output_array, &eltwise_87_output_array_intq)

/* Tensor #122 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_88_output, AI_STATIC,
  122, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_88_output_array, &eltwise_88_output_array_intq)

/* Tensor #123 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_89_output, AI_STATIC,
  123, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &eltwise_89_output_array, &eltwise_89_output_array_intq)

/* Tensor #124 */
AI_TENSOR_OBJ_DECLARE(
  embedding_embeddings, AI_STATIC,
  124, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 66), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &embedding_embeddings_array, &embedding_embeddings_array_intq)

/* Tensor #125 */
AI_TENSOR_OBJ_DECLARE(
  encoder_layer_1_layer_normalization_18_beta_3D, AI_STATIC,
  125, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &encoder_layer_1_layer_normalization_18_beta_3D_array, &encoder_layer_1_layer_normalization_18_beta_3D_array_intq)

/* Tensor #126 */
AI_TENSOR_OBJ_DECLARE(
  encoder_layer_1_layer_normalization_18_gamma_3D, AI_STATIC,
  126, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &encoder_layer_1_layer_normalization_18_gamma_3D_array, &encoder_layer_1_layer_normalization_18_gamma_3D_array_intq)

/* Tensor #127 */
AI_TENSOR_OBJ_DECLARE(
  encoder_layer_1_layer_normalization_19_beta_3D, AI_STATIC,
  127, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &encoder_layer_1_layer_normalization_19_beta_3D_array, &encoder_layer_1_layer_normalization_19_beta_3D_array_intq)

/* Tensor #128 */
AI_TENSOR_OBJ_DECLARE(
  encoder_layer_1_layer_normalization_19_gamma_3D, AI_STATIC,
  128, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &encoder_layer_1_layer_normalization_19_gamma_3D_array, &encoder_layer_1_layer_normalization_19_gamma_3D_array_intq)

/* Tensor #129 */
AI_TENSOR_OBJ_DECLARE(
  encoder_layer_1_multi_head_attention_9_dense_55_bias_3D, AI_STATIC,
  129, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &encoder_layer_1_multi_head_attention_9_dense_55_bias_3D_array, &encoder_layer_1_multi_head_attention_9_dense_55_bias_3D_array_intq)

/* Tensor #130 */
AI_TENSOR_OBJ_DECLARE(
  encoder_layer_1_multi_head_attention_9_dense_56_bias_3D, AI_STATIC,
  130, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &encoder_layer_1_multi_head_attention_9_dense_56_bias_3D_array, &encoder_layer_1_multi_head_attention_9_dense_56_bias_3D_array_intq)

/* Tensor #131 */
AI_TENSOR_OBJ_DECLARE(
  encoder_layer_1_multi_head_attention_9_dense_57_bias_3D, AI_STATIC,
  131, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &encoder_layer_1_multi_head_attention_9_dense_57_bias_3D_array, &encoder_layer_1_multi_head_attention_9_dense_57_bias_3D_array_intq)

/* Tensor #132 */
AI_TENSOR_OBJ_DECLARE(
  encoder_layer_1_multi_head_attention_9_dense_58_bias_3D, AI_STATIC,
  132, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &encoder_layer_1_multi_head_attention_9_dense_58_bias_3D_array, &encoder_layer_1_multi_head_attention_9_dense_58_bias_3D_array_intq)

/* Tensor #133 */
AI_TENSOR_OBJ_DECLARE(
  encoder_layer_2_layer_normalization_20_beta_3D, AI_STATIC,
  133, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &encoder_layer_2_layer_normalization_20_beta_3D_array, &encoder_layer_2_layer_normalization_20_beta_3D_array_intq)

/* Tensor #134 */
AI_TENSOR_OBJ_DECLARE(
  encoder_layer_2_layer_normalization_20_gamma_3D, AI_STATIC,
  134, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &encoder_layer_2_layer_normalization_20_gamma_3D_array, &encoder_layer_2_layer_normalization_20_gamma_3D_array_intq)

/* Tensor #135 */
AI_TENSOR_OBJ_DECLARE(
  encoder_layer_2_layer_normalization_21_beta_3D, AI_STATIC,
  135, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &encoder_layer_2_layer_normalization_21_beta_3D_array, &encoder_layer_2_layer_normalization_21_beta_3D_array_intq)

/* Tensor #136 */
AI_TENSOR_OBJ_DECLARE(
  encoder_layer_2_layer_normalization_21_gamma_3D, AI_STATIC,
  136, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &encoder_layer_2_layer_normalization_21_gamma_3D_array, &encoder_layer_2_layer_normalization_21_gamma_3D_array_intq)

/* Tensor #137 */
AI_TENSOR_OBJ_DECLARE(
  encoder_layer_2_multi_head_attention_10_dense_61_bias_3D, AI_STATIC,
  137, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &encoder_layer_2_multi_head_attention_10_dense_61_bias_3D_array, &encoder_layer_2_multi_head_attention_10_dense_61_bias_3D_array_intq)

/* Tensor #138 */
AI_TENSOR_OBJ_DECLARE(
  encoder_layer_2_multi_head_attention_10_dense_62_bias_3D, AI_STATIC,
  138, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &encoder_layer_2_multi_head_attention_10_dense_62_bias_3D_array, &encoder_layer_2_multi_head_attention_10_dense_62_bias_3D_array_intq)

/* Tensor #139 */
AI_TENSOR_OBJ_DECLARE(
  encoder_layer_2_multi_head_attention_10_dense_63_bias_3D, AI_STATIC,
  139, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &encoder_layer_2_multi_head_attention_10_dense_63_bias_3D_array, &encoder_layer_2_multi_head_attention_10_dense_63_bias_3D_array_intq)

/* Tensor #140 */
AI_TENSOR_OBJ_DECLARE(
  encoder_layer_2_multi_head_attention_10_dense_64_bias_3D, AI_STATIC,
  140, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &encoder_layer_2_multi_head_attention_10_dense_64_bias_3D_array, &encoder_layer_2_multi_head_attention_10_dense_64_bias_3D_array_intq)

/* Tensor #141 */
AI_TENSOR_OBJ_DECLARE(
  encoder_layer_3_layer_normalization_22_beta_3D, AI_STATIC,
  141, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &encoder_layer_3_layer_normalization_22_beta_3D_array, &encoder_layer_3_layer_normalization_22_beta_3D_array_intq)

/* Tensor #142 */
AI_TENSOR_OBJ_DECLARE(
  encoder_layer_3_layer_normalization_22_gamma_3D, AI_STATIC,
  142, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &encoder_layer_3_layer_normalization_22_gamma_3D_array, &encoder_layer_3_layer_normalization_22_gamma_3D_array_intq)

/* Tensor #143 */
AI_TENSOR_OBJ_DECLARE(
  encoder_layer_3_layer_normalization_23_beta_3D, AI_STATIC,
  143, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &encoder_layer_3_layer_normalization_23_beta_3D_array, &encoder_layer_3_layer_normalization_23_beta_3D_array_intq)

/* Tensor #144 */
AI_TENSOR_OBJ_DECLARE(
  encoder_layer_3_layer_normalization_23_gamma_3D, AI_STATIC,
  144, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &encoder_layer_3_layer_normalization_23_gamma_3D_array, &encoder_layer_3_layer_normalization_23_gamma_3D_array_intq)

/* Tensor #145 */
AI_TENSOR_OBJ_DECLARE(
  encoder_layer_3_multi_head_attention_11_dense_67_bias_3D, AI_STATIC,
  145, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &encoder_layer_3_multi_head_attention_11_dense_67_bias_3D_array, &encoder_layer_3_multi_head_attention_11_dense_67_bias_3D_array_intq)

/* Tensor #146 */
AI_TENSOR_OBJ_DECLARE(
  encoder_layer_3_multi_head_attention_11_dense_68_bias_3D, AI_STATIC,
  146, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &encoder_layer_3_multi_head_attention_11_dense_68_bias_3D_array, &encoder_layer_3_multi_head_attention_11_dense_68_bias_3D_array_intq)

/* Tensor #147 */
AI_TENSOR_OBJ_DECLARE(
  encoder_layer_3_multi_head_attention_11_dense_69_bias_3D, AI_STATIC,
  147, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &encoder_layer_3_multi_head_attention_11_dense_69_bias_3D_array, &encoder_layer_3_multi_head_attention_11_dense_69_bias_3D_array_intq)

/* Tensor #148 */
AI_TENSOR_OBJ_DECLARE(
  encoder_layer_3_multi_head_attention_11_dense_70_bias_3D, AI_STATIC,
  148, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &encoder_layer_3_multi_head_attention_11_dense_70_bias_3D_array, &encoder_layer_3_multi_head_attention_11_dense_70_bias_3D_array_intq)

/* Tensor #149 */
AI_TENSOR_OBJ_DECLARE(
  encoder_layer_layer_normalization_16_beta_3D, AI_STATIC,
  149, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &encoder_layer_layer_normalization_16_beta_3D_array, &encoder_layer_layer_normalization_16_beta_3D_array_intq)

/* Tensor #150 */
AI_TENSOR_OBJ_DECLARE(
  encoder_layer_layer_normalization_16_gamma_3D, AI_STATIC,
  150, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &encoder_layer_layer_normalization_16_gamma_3D_array, &encoder_layer_layer_normalization_16_gamma_3D_array_intq)

/* Tensor #151 */
AI_TENSOR_OBJ_DECLARE(
  encoder_layer_layer_normalization_17_beta_3D, AI_STATIC,
  151, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &encoder_layer_layer_normalization_17_beta_3D_array, &encoder_layer_layer_normalization_17_beta_3D_array_intq)

/* Tensor #152 */
AI_TENSOR_OBJ_DECLARE(
  encoder_layer_layer_normalization_17_gamma_3D, AI_STATIC,
  152, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &encoder_layer_layer_normalization_17_gamma_3D_array, &encoder_layer_layer_normalization_17_gamma_3D_array_intq)

/* Tensor #153 */
AI_TENSOR_OBJ_DECLARE(
  encoder_layer_multi_head_attention_8_dense_49_bias_3D, AI_STATIC,
  153, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &encoder_layer_multi_head_attention_8_dense_49_bias_3D_array, &encoder_layer_multi_head_attention_8_dense_49_bias_3D_array_intq)

/* Tensor #154 */
AI_TENSOR_OBJ_DECLARE(
  encoder_layer_multi_head_attention_8_dense_50_bias_3D, AI_STATIC,
  154, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &encoder_layer_multi_head_attention_8_dense_50_bias_3D_array, &encoder_layer_multi_head_attention_8_dense_50_bias_3D_array_intq)

/* Tensor #155 */
AI_TENSOR_OBJ_DECLARE(
  encoder_layer_multi_head_attention_8_dense_51_bias_3D, AI_STATIC,
  155, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &encoder_layer_multi_head_attention_8_dense_51_bias_3D_array, &encoder_layer_multi_head_attention_8_dense_51_bias_3D_array_intq)

/* Tensor #156 */
AI_TENSOR_OBJ_DECLARE(
  encoder_layer_multi_head_attention_8_dense_52_bias_3D, AI_STATIC,
  156, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &encoder_layer_multi_head_attention_8_dense_52_bias_3D_array, &encoder_layer_multi_head_attention_8_dense_52_bias_3D_array_intq)

/* Tensor #157 */
AI_TENSOR_OBJ_DECLARE(
  gather_0_output, AI_STATIC,
  157, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 100, 1), AI_STRIDE_INIT(4, 1, 1, 256, 25600),
  1, &gather_0_output_array, &gather_0_output_array_intq)

/* Tensor #158 */
AI_TENSOR_OBJ_DECLARE(
  gather_0_output0, AI_STATIC,
  158, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &gather_0_output_array, &gather_0_output_array_intq)

/* Tensor #159 */
AI_TENSOR_OBJ_DECLARE(
  gemm_101_output, AI_STATIC,
  159, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &gemm_101_output_array, &gemm_101_output_array_intq)

/* Tensor #160 */
AI_TENSOR_OBJ_DECLARE(
  gemm_101_scratch0, AI_STATIC,
  160, 0x0,
  AI_SHAPE_INIT(4, 1, 1024, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1024, 1024),
  1, &gemm_101_scratch0_array, NULL)

/* Tensor #161 */
AI_TENSOR_OBJ_DECLARE(
  gemm_101_weights, AI_STATIC,
  161, 0x1,
  AI_SHAPE_INIT(4, 256, 1, 1, 256), AI_STRIDE_INIT(4, 1, 256, 65536, 65536),
  1, &gemm_101_weights_array, &gemm_101_weights_array_intq)

/* Tensor #162 */
AI_TENSOR_OBJ_DECLARE(
  gemm_106_output, AI_STATIC,
  162, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &gemm_106_output_array, &gemm_106_output_array_intq)

/* Tensor #163 */
AI_TENSOR_OBJ_DECLARE(
  gemm_106_scratch0, AI_STATIC,
  163, 0x0,
  AI_SHAPE_INIT(4, 1, 1024, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1024, 1024),
  1, &gemm_106_scratch0_array, NULL)

/* Tensor #164 */
AI_TENSOR_OBJ_DECLARE(
  gemm_106_weights, AI_STATIC,
  164, 0x1,
  AI_SHAPE_INIT(4, 256, 1, 1, 256), AI_STRIDE_INIT(4, 1, 256, 65536, 65536),
  1, &gemm_106_weights_array, &gemm_106_weights_array_intq)

/* Tensor #165 */
AI_TENSOR_OBJ_DECLARE(
  gemm_111_0_0_eltwise_112_conversion_output, AI_STATIC,
  165, 0x1,
  AI_SHAPE_INIT(4, 1, 100, 100, 4), AI_STRIDE_INIT(4, 1, 1, 100, 10000),
  1, &gemm_111_0_0_eltwise_112_conversion_output_array, &gemm_111_0_0_eltwise_112_conversion_output_array_intq)

/* Tensor #166 */
AI_TENSOR_OBJ_DECLARE(
  gemm_111_bias, AI_STATIC,
  166, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &gemm_111_bias_array, &gemm_111_bias_array_intq)

/* Tensor #167 */
AI_TENSOR_OBJ_DECLARE(
  gemm_111_bias_0_2_gemm_111_conversion_output, AI_STATIC,
  167, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &gemm_111_bias_0_2_gemm_111_conversion_output_array, NULL)

/* Tensor #168 */
AI_TENSOR_OBJ_DECLARE(
  gemm_111_output, AI_STATIC,
  168, 0x0,
  AI_SHAPE_INIT(4, 1, 100, 100, 4), AI_STRIDE_INIT(4, 4, 4, 400, 40000),
  1, &gemm_111_output_array, NULL)

/* Tensor #169 */
AI_TENSOR_OBJ_DECLARE(
  gemm_114_output, AI_STATIC,
  169, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &gemm_114_output_array, &gemm_114_output_array_intq)

/* Tensor #170 */
AI_TENSOR_OBJ_DECLARE(
  gemm_114_scratch0, AI_STATIC,
  170, 0x0,
  AI_SHAPE_INIT(4, 1, 1024, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1024, 1024),
  1, &gemm_114_scratch0_array, NULL)

/* Tensor #171 */
AI_TENSOR_OBJ_DECLARE(
  gemm_114_weights, AI_STATIC,
  171, 0x1,
  AI_SHAPE_INIT(4, 256, 1, 1, 256), AI_STRIDE_INIT(4, 1, 256, 65536, 65536),
  1, &gemm_114_weights_array, &gemm_114_weights_array_intq)

/* Tensor #172 */
AI_TENSOR_OBJ_DECLARE(
  gemm_119_0_0_transpose_120_conversion_output, AI_STATIC,
  172, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 100, 4), AI_STRIDE_INIT(4, 1, 1, 64, 6400),
  1, &gemm_119_0_0_transpose_120_conversion_output_array, &gemm_119_0_0_transpose_120_conversion_output_array_intq)

/* Tensor #173 */
AI_TENSOR_OBJ_DECLARE(
  gemm_119_bias, AI_STATIC,
  173, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &gemm_119_bias_array, &gemm_119_bias_array_intq)

/* Tensor #174 */
AI_TENSOR_OBJ_DECLARE(
  gemm_119_bias_0_2_gemm_119_conversion_output, AI_STATIC,
  174, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &gemm_119_bias_0_2_gemm_119_conversion_output_array, NULL)

/* Tensor #175 */
AI_TENSOR_OBJ_DECLARE(
  gemm_119_output, AI_STATIC,
  175, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 100, 4), AI_STRIDE_INIT(4, 4, 4, 256, 25600),
  1, &gemm_119_output_array, NULL)

/* Tensor #176 */
AI_TENSOR_OBJ_DECLARE(
  gemm_130_output, AI_STATIC,
  176, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &gemm_130_output_array, &gemm_130_output_array_intq)

/* Tensor #177 */
AI_TENSOR_OBJ_DECLARE(
  gemm_130_scratch0, AI_STATIC,
  177, 0x0,
  AI_SHAPE_INIT(4, 1, 1024, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1024, 1024),
  1, &gemm_130_scratch0_array, NULL)

/* Tensor #178 */
AI_TENSOR_OBJ_DECLARE(
  gemm_130_weights, AI_STATIC,
  178, 0x1,
  AI_SHAPE_INIT(4, 256, 1, 1, 256), AI_STRIDE_INIT(4, 1, 256, 65536, 65536),
  1, &gemm_130_weights_array, &gemm_130_weights_array_intq)

/* Tensor #179 */
AI_TENSOR_OBJ_DECLARE(
  gemm_14_output, AI_STATIC,
  179, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &gemm_14_output_array, &gemm_14_output_array_intq)

/* Tensor #180 */
AI_TENSOR_OBJ_DECLARE(
  gemm_14_scratch0, AI_STATIC,
  180, 0x0,
  AI_SHAPE_INIT(4, 1, 1024, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1024, 1024),
  1, &gemm_14_scratch0_array, NULL)

/* Tensor #181 */
AI_TENSOR_OBJ_DECLARE(
  gemm_14_weights, AI_STATIC,
  181, 0x1,
  AI_SHAPE_INIT(4, 256, 1, 1, 256), AI_STRIDE_INIT(4, 1, 256, 65536, 65536),
  1, &gemm_14_weights_array, &gemm_14_weights_array_intq)

/* Tensor #182 */
AI_TENSOR_OBJ_DECLARE(
  gemm_152_output, AI_STATIC,
  182, 0x1,
  AI_SHAPE_INIT(4, 1, 512, 1, 100), AI_STRIDE_INIT(4, 1, 1, 512, 512),
  1, &gemm_152_output_array, &gemm_152_output_array_intq)

/* Tensor #183 */
AI_TENSOR_OBJ_DECLARE(
  gemm_152_scratch0, AI_STATIC,
  183, 0x0,
  AI_SHAPE_INIT(4, 1, 1024, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1024, 1024),
  1, &gemm_152_scratch0_array, NULL)

/* Tensor #184 */
AI_TENSOR_OBJ_DECLARE(
  gemm_152_weights, AI_STATIC,
  184, 0x1,
  AI_SHAPE_INIT(4, 256, 1, 1, 512), AI_STRIDE_INIT(4, 1, 256, 131072, 131072),
  1, &gemm_152_weights_array, &gemm_152_weights_array_intq)

/* Tensor #185 */
AI_TENSOR_OBJ_DECLARE(
  gemm_163_output, AI_STATIC,
  185, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &gemm_163_output_array, &gemm_163_output_array_intq)

/* Tensor #186 */
AI_TENSOR_OBJ_DECLARE(
  gemm_163_scratch0, AI_STATIC,
  186, 0x0,
  AI_SHAPE_INIT(4, 1, 2048, 1, 1), AI_STRIDE_INIT(4, 1, 1, 2048, 2048),
  1, &gemm_163_scratch0_array, NULL)

/* Tensor #187 */
AI_TENSOR_OBJ_DECLARE(
  gemm_163_weights, AI_STATIC,
  187, 0x1,
  AI_SHAPE_INIT(4, 512, 1, 1, 256), AI_STRIDE_INIT(4, 1, 512, 131072, 131072),
  1, &gemm_163_weights_array, &gemm_163_weights_array_intq)

/* Tensor #188 */
AI_TENSOR_OBJ_DECLARE(
  gemm_188_output, AI_STATIC,
  188, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &gemm_188_output_array, &gemm_188_output_array_intq)

/* Tensor #189 */
AI_TENSOR_OBJ_DECLARE(
  gemm_188_scratch0, AI_STATIC,
  189, 0x0,
  AI_SHAPE_INIT(4, 1, 1024, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1024, 1024),
  1, &gemm_188_scratch0_array, NULL)

/* Tensor #190 */
AI_TENSOR_OBJ_DECLARE(
  gemm_188_weights, AI_STATIC,
  190, 0x1,
  AI_SHAPE_INIT(4, 256, 1, 1, 256), AI_STRIDE_INIT(4, 1, 256, 65536, 65536),
  1, &gemm_188_weights_array, &gemm_188_weights_array_intq)

/* Tensor #191 */
AI_TENSOR_OBJ_DECLARE(
  gemm_193_output, AI_STATIC,
  191, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &gemm_193_output_array, &gemm_193_output_array_intq)

/* Tensor #192 */
AI_TENSOR_OBJ_DECLARE(
  gemm_193_scratch0, AI_STATIC,
  192, 0x0,
  AI_SHAPE_INIT(4, 1, 1024, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1024, 1024),
  1, &gemm_193_scratch0_array, NULL)

/* Tensor #193 */
AI_TENSOR_OBJ_DECLARE(
  gemm_193_weights, AI_STATIC,
  193, 0x1,
  AI_SHAPE_INIT(4, 256, 1, 1, 256), AI_STRIDE_INIT(4, 1, 256, 65536, 65536),
  1, &gemm_193_weights_array, &gemm_193_weights_array_intq)

/* Tensor #194 */
AI_TENSOR_OBJ_DECLARE(
  gemm_198_0_0_eltwise_199_conversion_output, AI_STATIC,
  194, 0x1,
  AI_SHAPE_INIT(4, 1, 100, 100, 4), AI_STRIDE_INIT(4, 1, 1, 100, 10000),
  1, &gemm_198_0_0_eltwise_199_conversion_output_array, &gemm_198_0_0_eltwise_199_conversion_output_array_intq)

/* Tensor #195 */
AI_TENSOR_OBJ_DECLARE(
  gemm_198_bias, AI_STATIC,
  195, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &gemm_198_bias_array, &gemm_198_bias_array_intq)

/* Tensor #196 */
AI_TENSOR_OBJ_DECLARE(
  gemm_198_bias_0_2_gemm_198_conversion_output, AI_STATIC,
  196, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &gemm_198_bias_0_2_gemm_198_conversion_output_array, NULL)

/* Tensor #197 */
AI_TENSOR_OBJ_DECLARE(
  gemm_198_output, AI_STATIC,
  197, 0x0,
  AI_SHAPE_INIT(4, 1, 100, 100, 4), AI_STRIDE_INIT(4, 4, 4, 400, 40000),
  1, &gemm_198_output_array, NULL)

/* Tensor #198 */
AI_TENSOR_OBJ_DECLARE(
  gemm_19_output, AI_STATIC,
  198, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &gemm_19_output_array, &gemm_19_output_array_intq)

/* Tensor #199 */
AI_TENSOR_OBJ_DECLARE(
  gemm_19_scratch0, AI_STATIC,
  199, 0x0,
  AI_SHAPE_INIT(4, 1, 1024, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1024, 1024),
  1, &gemm_19_scratch0_array, NULL)

/* Tensor #200 */
AI_TENSOR_OBJ_DECLARE(
  gemm_19_weights, AI_STATIC,
  200, 0x1,
  AI_SHAPE_INIT(4, 256, 1, 1, 256), AI_STRIDE_INIT(4, 1, 256, 65536, 65536),
  1, &gemm_19_weights_array, &gemm_19_weights_array_intq)

/* Tensor #201 */
AI_TENSOR_OBJ_DECLARE(
  gemm_201_output, AI_STATIC,
  201, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &gemm_201_output_array, &gemm_201_output_array_intq)

/* Tensor #202 */
AI_TENSOR_OBJ_DECLARE(
  gemm_201_scratch0, AI_STATIC,
  202, 0x0,
  AI_SHAPE_INIT(4, 1, 1024, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1024, 1024),
  1, &gemm_201_scratch0_array, NULL)

/* Tensor #203 */
AI_TENSOR_OBJ_DECLARE(
  gemm_201_weights, AI_STATIC,
  203, 0x1,
  AI_SHAPE_INIT(4, 256, 1, 1, 256), AI_STRIDE_INIT(4, 1, 256, 65536, 65536),
  1, &gemm_201_weights_array, &gemm_201_weights_array_intq)

/* Tensor #204 */
AI_TENSOR_OBJ_DECLARE(
  gemm_206_0_0_transpose_207_conversion_output, AI_STATIC,
  204, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 100, 4), AI_STRIDE_INIT(4, 1, 1, 64, 6400),
  1, &gemm_206_0_0_transpose_207_conversion_output_array, &gemm_206_0_0_transpose_207_conversion_output_array_intq)

/* Tensor #205 */
AI_TENSOR_OBJ_DECLARE(
  gemm_206_bias, AI_STATIC,
  205, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &gemm_206_bias_array, &gemm_206_bias_array_intq)

/* Tensor #206 */
AI_TENSOR_OBJ_DECLARE(
  gemm_206_bias_0_2_gemm_206_conversion_output, AI_STATIC,
  206, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &gemm_206_bias_0_2_gemm_206_conversion_output_array, NULL)

/* Tensor #207 */
AI_TENSOR_OBJ_DECLARE(
  gemm_206_output, AI_STATIC,
  207, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 100, 4), AI_STRIDE_INIT(4, 4, 4, 256, 25600),
  1, &gemm_206_output_array, NULL)

/* Tensor #208 */
AI_TENSOR_OBJ_DECLARE(
  gemm_217_output, AI_STATIC,
  208, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &gemm_217_output_array, &gemm_217_output_array_intq)

/* Tensor #209 */
AI_TENSOR_OBJ_DECLARE(
  gemm_217_scratch0, AI_STATIC,
  209, 0x0,
  AI_SHAPE_INIT(4, 1, 1024, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1024, 1024),
  1, &gemm_217_scratch0_array, NULL)

/* Tensor #210 */
AI_TENSOR_OBJ_DECLARE(
  gemm_217_weights, AI_STATIC,
  210, 0x1,
  AI_SHAPE_INIT(4, 256, 1, 1, 256), AI_STRIDE_INIT(4, 1, 256, 65536, 65536),
  1, &gemm_217_weights_array, &gemm_217_weights_array_intq)

/* Tensor #211 */
AI_TENSOR_OBJ_DECLARE(
  gemm_239_output, AI_STATIC,
  211, 0x1,
  AI_SHAPE_INIT(4, 1, 512, 1, 100), AI_STRIDE_INIT(4, 1, 1, 512, 512),
  1, &gemm_239_output_array, &gemm_239_output_array_intq)

/* Tensor #212 */
AI_TENSOR_OBJ_DECLARE(
  gemm_239_scratch0, AI_STATIC,
  212, 0x0,
  AI_SHAPE_INIT(4, 1, 1024, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1024, 1024),
  1, &gemm_239_scratch0_array, NULL)

/* Tensor #213 */
AI_TENSOR_OBJ_DECLARE(
  gemm_239_weights, AI_STATIC,
  213, 0x1,
  AI_SHAPE_INIT(4, 256, 1, 1, 512), AI_STRIDE_INIT(4, 1, 256, 131072, 131072),
  1, &gemm_239_weights_array, &gemm_239_weights_array_intq)

/* Tensor #214 */
AI_TENSOR_OBJ_DECLARE(
  gemm_24_0_0_eltwise_25_conversion_output, AI_STATIC,
  214, 0x1,
  AI_SHAPE_INIT(4, 1, 100, 100, 4), AI_STRIDE_INIT(4, 1, 1, 100, 10000),
  1, &gemm_24_0_0_eltwise_25_conversion_output_array, &gemm_24_0_0_eltwise_25_conversion_output_array_intq)

/* Tensor #215 */
AI_TENSOR_OBJ_DECLARE(
  gemm_24_bias, AI_STATIC,
  215, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &gemm_24_bias_array, &gemm_24_bias_array_intq)

/* Tensor #216 */
AI_TENSOR_OBJ_DECLARE(
  gemm_24_bias_0_2_gemm_24_conversion_output, AI_STATIC,
  216, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &gemm_24_bias_0_2_gemm_24_conversion_output_array, NULL)

/* Tensor #217 */
AI_TENSOR_OBJ_DECLARE(
  gemm_24_output, AI_STATIC,
  217, 0x0,
  AI_SHAPE_INIT(4, 1, 100, 100, 4), AI_STRIDE_INIT(4, 4, 4, 400, 40000),
  1, &gemm_24_output_array, NULL)

/* Tensor #218 */
AI_TENSOR_OBJ_DECLARE(
  gemm_250_output, AI_STATIC,
  218, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &gemm_250_output_array, &gemm_250_output_array_intq)

/* Tensor #219 */
AI_TENSOR_OBJ_DECLARE(
  gemm_250_scratch0, AI_STATIC,
  219, 0x0,
  AI_SHAPE_INIT(4, 1, 2048, 1, 1), AI_STRIDE_INIT(4, 1, 1, 2048, 2048),
  1, &gemm_250_scratch0_array, NULL)

/* Tensor #220 */
AI_TENSOR_OBJ_DECLARE(
  gemm_250_weights, AI_STATIC,
  220, 0x1,
  AI_SHAPE_INIT(4, 512, 1, 1, 256), AI_STRIDE_INIT(4, 1, 512, 131072, 131072),
  1, &gemm_250_weights_array, &gemm_250_weights_array_intq)

/* Tensor #221 */
AI_TENSOR_OBJ_DECLARE(
  gemm_275_output, AI_STATIC,
  221, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &gemm_275_output_array, &gemm_275_output_array_intq)

/* Tensor #222 */
AI_TENSOR_OBJ_DECLARE(
  gemm_275_scratch0, AI_STATIC,
  222, 0x0,
  AI_SHAPE_INIT(4, 1, 1024, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1024, 1024),
  1, &gemm_275_scratch0_array, NULL)

/* Tensor #223 */
AI_TENSOR_OBJ_DECLARE(
  gemm_275_weights, AI_STATIC,
  223, 0x1,
  AI_SHAPE_INIT(4, 256, 1, 1, 256), AI_STRIDE_INIT(4, 1, 256, 65536, 65536),
  1, &gemm_275_weights_array, &gemm_275_weights_array_intq)

/* Tensor #224 */
AI_TENSOR_OBJ_DECLARE(
  gemm_27_bias, AI_STATIC,
  224, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 4, 4, 1024, 1024),
  1, &gemm_27_bias_array, NULL)

/* Tensor #225 */
AI_TENSOR_OBJ_DECLARE(
  gemm_27_output, AI_STATIC,
  225, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &gemm_27_output_array, &gemm_27_output_array_intq)

/* Tensor #226 */
AI_TENSOR_OBJ_DECLARE(
  gemm_27_scratch0, AI_STATIC,
  226, 0x0,
  AI_SHAPE_INIT(4, 1, 1024, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1024, 1024),
  1, &gemm_27_scratch0_array, NULL)

/* Tensor #227 */
AI_TENSOR_OBJ_DECLARE(
  gemm_27_weights, AI_STATIC,
  227, 0x1,
  AI_SHAPE_INIT(4, 256, 1, 1, 256), AI_STRIDE_INIT(4, 1, 256, 65536, 65536),
  1, &gemm_27_weights_array, &gemm_27_weights_array_intq)

/* Tensor #228 */
AI_TENSOR_OBJ_DECLARE(
  gemm_280_output, AI_STATIC,
  228, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &gemm_280_output_array, &gemm_280_output_array_intq)

/* Tensor #229 */
AI_TENSOR_OBJ_DECLARE(
  gemm_280_scratch0, AI_STATIC,
  229, 0x0,
  AI_SHAPE_INIT(4, 1, 1024, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1024, 1024),
  1, &gemm_280_scratch0_array, NULL)

/* Tensor #230 */
AI_TENSOR_OBJ_DECLARE(
  gemm_280_weights, AI_STATIC,
  230, 0x1,
  AI_SHAPE_INIT(4, 256, 1, 1, 256), AI_STRIDE_INIT(4, 1, 256, 65536, 65536),
  1, &gemm_280_weights_array, &gemm_280_weights_array_intq)

/* Tensor #231 */
AI_TENSOR_OBJ_DECLARE(
  gemm_285_0_0_eltwise_286_conversion_output, AI_STATIC,
  231, 0x1,
  AI_SHAPE_INIT(4, 1, 100, 100, 4), AI_STRIDE_INIT(4, 1, 1, 100, 10000),
  1, &gemm_285_0_0_eltwise_286_conversion_output_array, &gemm_285_0_0_eltwise_286_conversion_output_array_intq)

/* Tensor #232 */
AI_TENSOR_OBJ_DECLARE(
  gemm_285_bias, AI_STATIC,
  232, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &gemm_285_bias_array, &gemm_285_bias_array_intq)

/* Tensor #233 */
AI_TENSOR_OBJ_DECLARE(
  gemm_285_bias_0_2_gemm_285_conversion_output, AI_STATIC,
  233, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &gemm_285_bias_0_2_gemm_285_conversion_output_array, NULL)

/* Tensor #234 */
AI_TENSOR_OBJ_DECLARE(
  gemm_285_output, AI_STATIC,
  234, 0x0,
  AI_SHAPE_INIT(4, 1, 100, 100, 4), AI_STRIDE_INIT(4, 4, 4, 400, 40000),
  1, &gemm_285_output_array, NULL)

/* Tensor #235 */
AI_TENSOR_OBJ_DECLARE(
  gemm_288_output, AI_STATIC,
  235, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &gemm_288_output_array, &gemm_288_output_array_intq)

/* Tensor #236 */
AI_TENSOR_OBJ_DECLARE(
  gemm_288_scratch0, AI_STATIC,
  236, 0x0,
  AI_SHAPE_INIT(4, 1, 1024, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1024, 1024),
  1, &gemm_288_scratch0_array, NULL)

/* Tensor #237 */
AI_TENSOR_OBJ_DECLARE(
  gemm_288_weights, AI_STATIC,
  237, 0x1,
  AI_SHAPE_INIT(4, 256, 1, 1, 256), AI_STRIDE_INIT(4, 1, 256, 65536, 65536),
  1, &gemm_288_weights_array, &gemm_288_weights_array_intq)

/* Tensor #238 */
AI_TENSOR_OBJ_DECLARE(
  gemm_293_0_0_transpose_294_conversion_output, AI_STATIC,
  238, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 100, 4), AI_STRIDE_INIT(4, 1, 1, 64, 6400),
  1, &gemm_293_0_0_transpose_294_conversion_output_array, &gemm_293_0_0_transpose_294_conversion_output_array_intq)

/* Tensor #239 */
AI_TENSOR_OBJ_DECLARE(
  gemm_293_bias, AI_STATIC,
  239, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &gemm_293_bias_array, &gemm_293_bias_array_intq)

/* Tensor #240 */
AI_TENSOR_OBJ_DECLARE(
  gemm_293_bias_0_conversion_output, AI_STATIC,
  240, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &gemm_293_bias_0_conversion_output_array, NULL)

/* Tensor #241 */
AI_TENSOR_OBJ_DECLARE(
  gemm_293_output, AI_STATIC,
  241, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 100, 4), AI_STRIDE_INIT(4, 4, 4, 256, 25600),
  1, &gemm_293_output_array, NULL)

/* Tensor #242 */
AI_TENSOR_OBJ_DECLARE(
  gemm_304_output, AI_STATIC,
  242, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &gemm_304_output_array, &gemm_304_output_array_intq)

/* Tensor #243 */
AI_TENSOR_OBJ_DECLARE(
  gemm_304_scratch0, AI_STATIC,
  243, 0x0,
  AI_SHAPE_INIT(4, 1, 1024, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1024, 1024),
  1, &gemm_304_scratch0_array, NULL)

/* Tensor #244 */
AI_TENSOR_OBJ_DECLARE(
  gemm_304_weights, AI_STATIC,
  244, 0x1,
  AI_SHAPE_INIT(4, 256, 1, 1, 256), AI_STRIDE_INIT(4, 1, 256, 65536, 65536),
  1, &gemm_304_weights_array, &gemm_304_weights_array_intq)

/* Tensor #245 */
AI_TENSOR_OBJ_DECLARE(
  gemm_326_output, AI_STATIC,
  245, 0x1,
  AI_SHAPE_INIT(4, 1, 512, 1, 100), AI_STRIDE_INIT(4, 1, 1, 512, 512),
  1, &gemm_326_output_array, &gemm_326_output_array_intq)

/* Tensor #246 */
AI_TENSOR_OBJ_DECLARE(
  gemm_326_scratch0, AI_STATIC,
  246, 0x0,
  AI_SHAPE_INIT(4, 1, 1024, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1024, 1024),
  1, &gemm_326_scratch0_array, NULL)

/* Tensor #247 */
AI_TENSOR_OBJ_DECLARE(
  gemm_326_weights, AI_STATIC,
  247, 0x1,
  AI_SHAPE_INIT(4, 256, 1, 1, 512), AI_STRIDE_INIT(4, 1, 256, 131072, 131072),
  1, &gemm_326_weights_array, &gemm_326_weights_array_intq)

/* Tensor #248 */
AI_TENSOR_OBJ_DECLARE(
  gemm_32_0_0_transpose_33_conversion_output, AI_STATIC,
  248, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 100, 4), AI_STRIDE_INIT(4, 1, 1, 64, 6400),
  1, &gemm_32_0_0_transpose_33_conversion_output_array, &gemm_32_0_0_transpose_33_conversion_output_array_intq)

/* Tensor #249 */
AI_TENSOR_OBJ_DECLARE(
  gemm_32_bias, AI_STATIC,
  249, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &gemm_32_bias_array, &gemm_32_bias_array_intq)

/* Tensor #250 */
AI_TENSOR_OBJ_DECLARE(
  gemm_32_bias_0_2_gemm_32_conversion_output, AI_STATIC,
  250, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &gemm_32_bias_0_2_gemm_32_conversion_output_array, NULL)

/* Tensor #251 */
AI_TENSOR_OBJ_DECLARE(
  gemm_32_output, AI_STATIC,
  251, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 100, 4), AI_STRIDE_INIT(4, 4, 4, 256, 25600),
  1, &gemm_32_output_array, NULL)

/* Tensor #252 */
AI_TENSOR_OBJ_DECLARE(
  gemm_337_output, AI_STATIC,
  252, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &gemm_337_output_array, &gemm_337_output_array_intq)

/* Tensor #253 */
AI_TENSOR_OBJ_DECLARE(
  gemm_337_scratch0, AI_STATIC,
  253, 0x0,
  AI_SHAPE_INIT(4, 1, 2048, 1, 1), AI_STRIDE_INIT(4, 1, 1, 2048, 2048),
  1, &gemm_337_scratch0_array, NULL)

/* Tensor #254 */
AI_TENSOR_OBJ_DECLARE(
  gemm_337_weights, AI_STATIC,
  254, 0x1,
  AI_SHAPE_INIT(4, 512, 1, 1, 256), AI_STRIDE_INIT(4, 1, 512, 131072, 131072),
  1, &gemm_337_weights_array, &gemm_337_weights_array_intq)

/* Tensor #255 */
AI_TENSOR_OBJ_DECLARE(
  gemm_359_bias, AI_STATIC,
  255, 0x0,
  AI_SHAPE_INIT(4, 1, 66, 1, 1), AI_STRIDE_INIT(4, 4, 4, 264, 264),
  1, &gemm_359_bias_array, NULL)

/* Tensor #256 */
AI_TENSOR_OBJ_DECLARE(
  gemm_359_output, AI_STATIC,
  256, 0x1,
  AI_SHAPE_INIT(4, 1, 66, 1, 100), AI_STRIDE_INIT(4, 1, 1, 66, 66),
  1, &gemm_359_output_array, &gemm_359_output_array_intq)

/* Tensor #257 */
AI_TENSOR_OBJ_DECLARE(
  gemm_359_scratch0, AI_STATIC,
  257, 0x0,
  AI_SHAPE_INIT(4, 1, 1024, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1024, 1024),
  1, &gemm_359_scratch0_array, NULL)

/* Tensor #258 */
AI_TENSOR_OBJ_DECLARE(
  gemm_359_weights, AI_STATIC,
  258, 0x1,
  AI_SHAPE_INIT(4, 256, 1, 1, 66), AI_STRIDE_INIT(4, 1, 256, 16896, 16896),
  1, &gemm_359_weights_array, &gemm_359_weights_array_intq)

/* Tensor #259 */
AI_TENSOR_OBJ_DECLARE(
  gemm_43_output, AI_STATIC,
  259, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &gemm_43_output_array, &gemm_43_output_array_intq)

/* Tensor #260 */
AI_TENSOR_OBJ_DECLARE(
  gemm_43_scratch0, AI_STATIC,
  260, 0x0,
  AI_SHAPE_INIT(4, 1, 1024, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1024, 1024),
  1, &gemm_43_scratch0_array, NULL)

/* Tensor #261 */
AI_TENSOR_OBJ_DECLARE(
  gemm_43_weights, AI_STATIC,
  261, 0x1,
  AI_SHAPE_INIT(4, 256, 1, 1, 256), AI_STRIDE_INIT(4, 1, 256, 65536, 65536),
  1, &gemm_43_weights_array, &gemm_43_weights_array_intq)

/* Tensor #262 */
AI_TENSOR_OBJ_DECLARE(
  gemm_65_bias, AI_STATIC,
  262, 0x0,
  AI_SHAPE_INIT(4, 1, 512, 1, 1), AI_STRIDE_INIT(4, 4, 4, 2048, 2048),
  1, &gemm_65_bias_array, NULL)

/* Tensor #263 */
AI_TENSOR_OBJ_DECLARE(
  gemm_65_output, AI_STATIC,
  263, 0x1,
  AI_SHAPE_INIT(4, 1, 512, 1, 100), AI_STRIDE_INIT(4, 1, 1, 512, 512),
  1, &gemm_65_output_array, &gemm_65_output_array_intq)

/* Tensor #264 */
AI_TENSOR_OBJ_DECLARE(
  gemm_65_scratch0, AI_STATIC,
  264, 0x0,
  AI_SHAPE_INIT(4, 1, 1024, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1024, 1024),
  1, &gemm_65_scratch0_array, NULL)

/* Tensor #265 */
AI_TENSOR_OBJ_DECLARE(
  gemm_65_weights, AI_STATIC,
  265, 0x1,
  AI_SHAPE_INIT(4, 256, 1, 1, 512), AI_STRIDE_INIT(4, 1, 256, 131072, 131072),
  1, &gemm_65_weights_array, &gemm_65_weights_array_intq)

/* Tensor #266 */
AI_TENSOR_OBJ_DECLARE(
  gemm_76_output, AI_STATIC,
  266, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &gemm_76_output_array, &gemm_76_output_array_intq)

/* Tensor #267 */
AI_TENSOR_OBJ_DECLARE(
  gemm_76_scratch0, AI_STATIC,
  267, 0x0,
  AI_SHAPE_INIT(4, 1, 2048, 1, 1), AI_STRIDE_INIT(4, 1, 1, 2048, 2048),
  1, &gemm_76_scratch0_array, NULL)

/* Tensor #268 */
AI_TENSOR_OBJ_DECLARE(
  gemm_76_weights, AI_STATIC,
  268, 0x1,
  AI_SHAPE_INIT(4, 512, 1, 1, 256), AI_STRIDE_INIT(4, 1, 512, 131072, 131072),
  1, &gemm_76_weights_array, &gemm_76_weights_array_intq)

/* Tensor #269 */
AI_TENSOR_OBJ_DECLARE(
  nl_113_0_0_gemm_119_conversion_output, AI_STATIC,
  269, 0x0,
  AI_SHAPE_INIT(4, 1, 100, 100, 4), AI_STRIDE_INIT(4, 4, 4, 400, 40000),
  1, &nl_113_0_0_gemm_119_conversion_output_array, NULL)

/* Tensor #270 */
AI_TENSOR_OBJ_DECLARE(
  nl_113_output, AI_STATIC,
  270, 0x1,
  AI_SHAPE_INIT(4, 1, 100, 100, 4), AI_STRIDE_INIT(4, 1, 1, 100, 10000),
  1, &nl_113_output_array, &nl_113_output_array_intq)

/* Tensor #271 */
AI_TENSOR_OBJ_DECLARE(
  nl_113_scratch0, AI_STATIC,
  271, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 4, 4, 1024, 1024),
  1, &nl_113_scratch0_array, NULL)

/* Tensor #272 */
AI_TENSOR_OBJ_DECLARE(
  nl_138_output, AI_STATIC,
  272, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &nl_138_output_array, &nl_138_output_array_intq)

/* Tensor #273 */
AI_TENSOR_OBJ_DECLARE(
  nl_171_output, AI_STATIC,
  273, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &nl_171_output_array, &nl_171_output_array_intq)

/* Tensor #274 */
AI_TENSOR_OBJ_DECLARE(
  nl_200_0_0_gemm_206_conversion_output, AI_STATIC,
  274, 0x0,
  AI_SHAPE_INIT(4, 1, 100, 100, 4), AI_STRIDE_INIT(4, 4, 4, 400, 40000),
  1, &nl_200_0_0_gemm_206_conversion_output_array, NULL)

/* Tensor #275 */
AI_TENSOR_OBJ_DECLARE(
  nl_200_output, AI_STATIC,
  275, 0x1,
  AI_SHAPE_INIT(4, 1, 100, 100, 4), AI_STRIDE_INIT(4, 1, 1, 100, 10000),
  1, &nl_200_output_array, &nl_200_output_array_intq)

/* Tensor #276 */
AI_TENSOR_OBJ_DECLARE(
  nl_200_scratch0, AI_STATIC,
  276, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 4, 4, 1024, 1024),
  1, &nl_200_scratch0_array, NULL)

/* Tensor #277 */
AI_TENSOR_OBJ_DECLARE(
  nl_225_output, AI_STATIC,
  277, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &nl_225_output_array, &nl_225_output_array_intq)

/* Tensor #278 */
AI_TENSOR_OBJ_DECLARE(
  nl_258_output, AI_STATIC,
  278, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &nl_258_output_array, &nl_258_output_array_intq)

/* Tensor #279 */
AI_TENSOR_OBJ_DECLARE(
  nl_26_0_0_gemm_32_conversion_output, AI_STATIC,
  279, 0x0,
  AI_SHAPE_INIT(4, 1, 100, 100, 4), AI_STRIDE_INIT(4, 4, 4, 400, 40000),
  1, &nl_26_0_0_gemm_32_conversion_output_array, NULL)

/* Tensor #280 */
AI_TENSOR_OBJ_DECLARE(
  nl_26_output, AI_STATIC,
  280, 0x1,
  AI_SHAPE_INIT(4, 1, 100, 100, 4), AI_STRIDE_INIT(4, 1, 1, 100, 10000),
  1, &nl_26_output_array, &nl_26_output_array_intq)

/* Tensor #281 */
AI_TENSOR_OBJ_DECLARE(
  nl_26_scratch0, AI_STATIC,
  281, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 4, 4, 1024, 1024),
  1, &nl_26_scratch0_array, NULL)

/* Tensor #282 */
AI_TENSOR_OBJ_DECLARE(
  nl_287_0_0_gemm_293_conversion_output, AI_STATIC,
  282, 0x0,
  AI_SHAPE_INIT(4, 1, 100, 100, 4), AI_STRIDE_INIT(4, 4, 4, 400, 40000),
  1, &nl_287_0_0_gemm_293_conversion_output_array, NULL)

/* Tensor #283 */
AI_TENSOR_OBJ_DECLARE(
  nl_287_output, AI_STATIC,
  283, 0x1,
  AI_SHAPE_INIT(4, 1, 100, 100, 4), AI_STRIDE_INIT(4, 1, 1, 100, 10000),
  1, &nl_287_output_array, &nl_287_output_array_intq)

/* Tensor #284 */
AI_TENSOR_OBJ_DECLARE(
  nl_287_scratch0, AI_STATIC,
  284, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 4, 4, 1024, 1024),
  1, &nl_287_scratch0_array, NULL)

/* Tensor #285 */
AI_TENSOR_OBJ_DECLARE(
  nl_312_output, AI_STATIC,
  285, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &nl_312_output_array, &nl_312_output_array_intq)

/* Tensor #286 */
AI_TENSOR_OBJ_DECLARE(
  nl_345_output, AI_STATIC,
  286, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &nl_345_output_array, &nl_345_output_array_intq)

/* Tensor #287 */
AI_TENSOR_OBJ_DECLARE(
  nl_51_output, AI_STATIC,
  287, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &nl_51_output_array, &nl_51_output_array_intq)

/* Tensor #288 */
AI_TENSOR_OBJ_DECLARE(
  nl_84_output, AI_STATIC,
  288, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &nl_84_output_array, &nl_84_output_array_intq)

/* Tensor #289 */
AI_TENSOR_OBJ_DECLARE(
  reduce_134_Mul_0_1_eltwise_135_conversion_0_1_eltwise_135_conversion_output, AI_STATIC,
  289, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &reduce_134_Mul_0_1_eltwise_135_conversion_0_1_eltwise_135_conversion_output_array, NULL)

/* Tensor #290 */
AI_TENSOR_OBJ_DECLARE(
  reduce_134_Mul_0_1_eltwise_135_conversion_output, AI_STATIC,
  290, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &reduce_134_Mul_0_1_eltwise_135_conversion_output_array, &reduce_134_Mul_0_1_eltwise_135_conversion_output_array_intq)

/* Tensor #291 */
AI_TENSOR_OBJ_DECLARE(
  reduce_134_Mul_output, AI_STATIC,
  291, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &reduce_134_Mul_output_array, NULL)

/* Tensor #292 */
AI_TENSOR_OBJ_DECLARE(
  reduce_134_output, AI_STATIC,
  292, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &reduce_134_output_array, NULL)

/* Tensor #293 */
AI_TENSOR_OBJ_DECLARE(
  reduce_136_Mul_0_0_eltwise_137_conversion_output, AI_STATIC,
  293, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &reduce_136_Mul_0_0_eltwise_137_conversion_output_array, &reduce_136_Mul_0_0_eltwise_137_conversion_output_array_intq)

/* Tensor #294 */
AI_TENSOR_OBJ_DECLARE(
  reduce_136_Mul_output, AI_STATIC,
  294, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &reduce_136_Mul_output_array, NULL)

/* Tensor #295 */
AI_TENSOR_OBJ_DECLARE(
  reduce_136_output, AI_STATIC,
  295, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &reduce_136_output_array, NULL)

/* Tensor #296 */
AI_TENSOR_OBJ_DECLARE(
  reduce_167_Mul_0_1_eltwise_168_conversion_0_1_eltwise_168_conversion_output, AI_STATIC,
  296, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &reduce_167_Mul_0_1_eltwise_168_conversion_0_1_eltwise_168_conversion_output_array, NULL)

/* Tensor #297 */
AI_TENSOR_OBJ_DECLARE(
  reduce_167_Mul_0_1_eltwise_168_conversion_output, AI_STATIC,
  297, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &reduce_167_Mul_0_1_eltwise_168_conversion_output_array, &reduce_167_Mul_0_1_eltwise_168_conversion_output_array_intq)

/* Tensor #298 */
AI_TENSOR_OBJ_DECLARE(
  reduce_167_Mul_output, AI_STATIC,
  298, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &reduce_167_Mul_output_array, NULL)

/* Tensor #299 */
AI_TENSOR_OBJ_DECLARE(
  reduce_167_output, AI_STATIC,
  299, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &reduce_167_output_array, NULL)

/* Tensor #300 */
AI_TENSOR_OBJ_DECLARE(
  reduce_169_Mul_0_0_eltwise_170_conversion_output, AI_STATIC,
  300, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &reduce_169_Mul_0_0_eltwise_170_conversion_output_array, &reduce_169_Mul_0_0_eltwise_170_conversion_output_array_intq)

/* Tensor #301 */
AI_TENSOR_OBJ_DECLARE(
  reduce_169_Mul_output, AI_STATIC,
  301, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &reduce_169_Mul_output_array, NULL)

/* Tensor #302 */
AI_TENSOR_OBJ_DECLARE(
  reduce_169_output, AI_STATIC,
  302, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &reduce_169_output_array, NULL)

/* Tensor #303 */
AI_TENSOR_OBJ_DECLARE(
  reduce_221_Mul_0_1_eltwise_222_conversion_0_1_eltwise_222_conversion_output, AI_STATIC,
  303, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &reduce_221_Mul_0_1_eltwise_222_conversion_0_1_eltwise_222_conversion_output_array, NULL)

/* Tensor #304 */
AI_TENSOR_OBJ_DECLARE(
  reduce_221_Mul_0_1_eltwise_222_conversion_output, AI_STATIC,
  304, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &reduce_221_Mul_0_1_eltwise_222_conversion_output_array, &reduce_221_Mul_0_1_eltwise_222_conversion_output_array_intq)

/* Tensor #305 */
AI_TENSOR_OBJ_DECLARE(
  reduce_221_Mul_output, AI_STATIC,
  305, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &reduce_221_Mul_output_array, NULL)

/* Tensor #306 */
AI_TENSOR_OBJ_DECLARE(
  reduce_221_output, AI_STATIC,
  306, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &reduce_221_output_array, NULL)

/* Tensor #307 */
AI_TENSOR_OBJ_DECLARE(
  reduce_223_Mul_0_0_eltwise_224_conversion_output, AI_STATIC,
  307, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &reduce_223_Mul_0_0_eltwise_224_conversion_output_array, &reduce_223_Mul_0_0_eltwise_224_conversion_output_array_intq)

/* Tensor #308 */
AI_TENSOR_OBJ_DECLARE(
  reduce_223_Mul_output, AI_STATIC,
  308, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &reduce_223_Mul_output_array, NULL)

/* Tensor #309 */
AI_TENSOR_OBJ_DECLARE(
  reduce_223_output, AI_STATIC,
  309, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &reduce_223_output_array, NULL)

/* Tensor #310 */
AI_TENSOR_OBJ_DECLARE(
  reduce_254_Mul_0_1_eltwise_255_conversion_0_1_eltwise_255_conversion_output, AI_STATIC,
  310, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &reduce_254_Mul_0_1_eltwise_255_conversion_0_1_eltwise_255_conversion_output_array, NULL)

/* Tensor #311 */
AI_TENSOR_OBJ_DECLARE(
  reduce_254_Mul_0_1_eltwise_255_conversion_output, AI_STATIC,
  311, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &reduce_254_Mul_0_1_eltwise_255_conversion_output_array, &reduce_254_Mul_0_1_eltwise_255_conversion_output_array_intq)

/* Tensor #312 */
AI_TENSOR_OBJ_DECLARE(
  reduce_254_Mul_output, AI_STATIC,
  312, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &reduce_254_Mul_output_array, NULL)

/* Tensor #313 */
AI_TENSOR_OBJ_DECLARE(
  reduce_254_output, AI_STATIC,
  313, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &reduce_254_output_array, NULL)

/* Tensor #314 */
AI_TENSOR_OBJ_DECLARE(
  reduce_256_Mul_0_0_eltwise_257_conversion_output, AI_STATIC,
  314, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &reduce_256_Mul_0_0_eltwise_257_conversion_output_array, &reduce_256_Mul_0_0_eltwise_257_conversion_output_array_intq)

/* Tensor #315 */
AI_TENSOR_OBJ_DECLARE(
  reduce_256_Mul_output, AI_STATIC,
  315, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &reduce_256_Mul_output_array, NULL)

/* Tensor #316 */
AI_TENSOR_OBJ_DECLARE(
  reduce_256_output, AI_STATIC,
  316, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &reduce_256_output_array, NULL)

/* Tensor #317 */
AI_TENSOR_OBJ_DECLARE(
  reduce_308_Mul_0_1_eltwise_309_conversion_0_1_eltwise_309_conversion_output, AI_STATIC,
  317, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &reduce_308_Mul_0_1_eltwise_309_conversion_0_1_eltwise_309_conversion_output_array, NULL)

/* Tensor #318 */
AI_TENSOR_OBJ_DECLARE(
  reduce_308_Mul_0_1_eltwise_309_conversion_output, AI_STATIC,
  318, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &reduce_308_Mul_0_1_eltwise_309_conversion_output_array, &reduce_308_Mul_0_1_eltwise_309_conversion_output_array_intq)

/* Tensor #319 */
AI_TENSOR_OBJ_DECLARE(
  reduce_308_Mul_output, AI_STATIC,
  319, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &reduce_308_Mul_output_array, NULL)

/* Tensor #320 */
AI_TENSOR_OBJ_DECLARE(
  reduce_308_output, AI_STATIC,
  320, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &reduce_308_output_array, NULL)

/* Tensor #321 */
AI_TENSOR_OBJ_DECLARE(
  reduce_310_Mul_0_0_eltwise_311_conversion_output, AI_STATIC,
  321, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &reduce_310_Mul_0_0_eltwise_311_conversion_output_array, &reduce_310_Mul_0_0_eltwise_311_conversion_output_array_intq)

/* Tensor #322 */
AI_TENSOR_OBJ_DECLARE(
  reduce_310_Mul_output, AI_STATIC,
  322, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &reduce_310_Mul_output_array, NULL)

/* Tensor #323 */
AI_TENSOR_OBJ_DECLARE(
  reduce_310_output, AI_STATIC,
  323, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &reduce_310_output_array, NULL)

/* Tensor #324 */
AI_TENSOR_OBJ_DECLARE(
  reduce_341_Mul_0_1_eltwise_342_conversion_0_1_eltwise_342_conversion_output, AI_STATIC,
  324, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &reduce_341_Mul_0_1_eltwise_342_conversion_0_1_eltwise_342_conversion_output_array, NULL)

/* Tensor #325 */
AI_TENSOR_OBJ_DECLARE(
  reduce_341_Mul_0_1_eltwise_342_conversion_output, AI_STATIC,
  325, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &reduce_341_Mul_0_1_eltwise_342_conversion_output_array, &reduce_341_Mul_0_1_eltwise_342_conversion_output_array_intq)

/* Tensor #326 */
AI_TENSOR_OBJ_DECLARE(
  reduce_341_Mul_output, AI_STATIC,
  326, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &reduce_341_Mul_output_array, NULL)

/* Tensor #327 */
AI_TENSOR_OBJ_DECLARE(
  reduce_341_output, AI_STATIC,
  327, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &reduce_341_output_array, NULL)

/* Tensor #328 */
AI_TENSOR_OBJ_DECLARE(
  reduce_343_Mul_0_0_eltwise_344_conversion_output, AI_STATIC,
  328, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &reduce_343_Mul_0_0_eltwise_344_conversion_output_array, &reduce_343_Mul_0_0_eltwise_344_conversion_output_array_intq)

/* Tensor #329 */
AI_TENSOR_OBJ_DECLARE(
  reduce_343_Mul_output, AI_STATIC,
  329, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &reduce_343_Mul_output_array, NULL)

/* Tensor #330 */
AI_TENSOR_OBJ_DECLARE(
  reduce_343_output, AI_STATIC,
  330, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &reduce_343_output_array, NULL)

/* Tensor #331 */
AI_TENSOR_OBJ_DECLARE(
  reduce_47_Mul_0_1_eltwise_48_conversion_0_1_eltwise_48_conversion_output, AI_STATIC,
  331, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &reduce_47_Mul_0_1_eltwise_48_conversion_0_1_eltwise_48_conversion_output_array, NULL)

/* Tensor #332 */
AI_TENSOR_OBJ_DECLARE(
  reduce_47_Mul_0_1_eltwise_48_conversion_output, AI_STATIC,
  332, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &reduce_47_Mul_0_1_eltwise_48_conversion_output_array, &reduce_47_Mul_0_1_eltwise_48_conversion_output_array_intq)

/* Tensor #333 */
AI_TENSOR_OBJ_DECLARE(
  reduce_47_Mul_bias, AI_STATIC,
  333, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &reduce_47_Mul_bias_array, NULL)

/* Tensor #334 */
AI_TENSOR_OBJ_DECLARE(
  reduce_47_Mul_output, AI_STATIC,
  334, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &reduce_47_Mul_output_array, NULL)

/* Tensor #335 */
AI_TENSOR_OBJ_DECLARE(
  reduce_47_Mul_scale, AI_STATIC,
  335, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &reduce_47_Mul_scale_array, NULL)

/* Tensor #336 */
AI_TENSOR_OBJ_DECLARE(
  reduce_47_output, AI_STATIC,
  336, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &reduce_47_output_array, NULL)

/* Tensor #337 */
AI_TENSOR_OBJ_DECLARE(
  reduce_49_Mul_0_0_eltwise_50_conversion_output, AI_STATIC,
  337, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &reduce_49_Mul_0_0_eltwise_50_conversion_output_array, &reduce_49_Mul_0_0_eltwise_50_conversion_output_array_intq)

/* Tensor #338 */
AI_TENSOR_OBJ_DECLARE(
  reduce_49_Mul_output, AI_STATIC,
  338, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &reduce_49_Mul_output_array, NULL)

/* Tensor #339 */
AI_TENSOR_OBJ_DECLARE(
  reduce_49_output, AI_STATIC,
  339, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &reduce_49_output_array, NULL)

/* Tensor #340 */
AI_TENSOR_OBJ_DECLARE(
  reduce_80_Mul_0_1_eltwise_81_conversion_0_1_eltwise_81_conversion_output, AI_STATIC,
  340, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &reduce_80_Mul_0_1_eltwise_81_conversion_0_1_eltwise_81_conversion_output_array, NULL)

/* Tensor #341 */
AI_TENSOR_OBJ_DECLARE(
  reduce_80_Mul_0_1_eltwise_81_conversion_output, AI_STATIC,
  341, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &reduce_80_Mul_0_1_eltwise_81_conversion_output_array, &reduce_80_Mul_0_1_eltwise_81_conversion_output_array_intq)

/* Tensor #342 */
AI_TENSOR_OBJ_DECLARE(
  reduce_80_Mul_output, AI_STATIC,
  342, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &reduce_80_Mul_output_array, NULL)

/* Tensor #343 */
AI_TENSOR_OBJ_DECLARE(
  reduce_80_output, AI_STATIC,
  343, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &reduce_80_output_array, NULL)

/* Tensor #344 */
AI_TENSOR_OBJ_DECLARE(
  reduce_82_Mul_0_0_eltwise_83_conversion_output, AI_STATIC,
  344, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &reduce_82_Mul_0_0_eltwise_83_conversion_output_array, &reduce_82_Mul_0_0_eltwise_83_conversion_output_array_intq)

/* Tensor #345 */
AI_TENSOR_OBJ_DECLARE(
  reduce_82_Mul_output, AI_STATIC,
  345, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &reduce_82_Mul_output_array, NULL)

/* Tensor #346 */
AI_TENSOR_OBJ_DECLARE(
  reduce_82_output, AI_STATIC,
  346, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 100), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &reduce_82_output_array, NULL)

/* Tensor #347 */
AI_TENSOR_OBJ_DECLARE(
  serving_default_input_10_output, AI_STATIC,
  347, 0x0,
  AI_SHAPE_INIT(4, 1, 100, 1, 1), AI_STRIDE_INIT(4, 4, 4, 400, 400),
  1, &serving_default_input_10_output_array, NULL)

/* Tensor #348 */
AI_TENSOR_OBJ_DECLARE(
  tiny_bert_generator_encoder_layer_3_multi_head_attention_11_truedivtiny_bert_generator_encoder_layer_3_multi_head_attention_11_Sqrt_4D, AI_STATIC,
  348, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &tiny_bert_generator_encoder_layer_3_multi_head_attention_11_truedivtiny_bert_generator_encoder_layer_3_multi_head_attention_11_Sqrt_4D_array, &tiny_bert_generator_encoder_layer_3_multi_head_attention_11_truedivtiny_bert_generator_encoder_layer_3_multi_head_attention_11_Sqrt_4D_array_intq)

/* Tensor #349 */
AI_TENSOR_OBJ_DECLARE(
  tiny_bert_generator_encoder_layer_layer_normalization_16_batchnorm_add_y_3D, AI_STATIC,
  349, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &tiny_bert_generator_encoder_layer_layer_normalization_16_batchnorm_add_y_3D_array, &tiny_bert_generator_encoder_layer_layer_normalization_16_batchnorm_add_y_3D_array_intq)

/* Tensor #350 */
AI_TENSOR_OBJ_DECLARE(
  tiny_bert_generator_tf___operators___add_y, AI_STATIC,
  350, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &tiny_bert_generator_tf___operators___add_y_array, &tiny_bert_generator_tf___operators___add_y_array_intq)

/* Tensor #351 */
AI_TENSOR_OBJ_DECLARE(
  tiny_bert_generator_tf_math_multiply_Mul_y_3D, AI_STATIC,
  351, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &tiny_bert_generator_tf_math_multiply_Mul_y_3D_array, &tiny_bert_generator_tf_math_multiply_Mul_y_3D_array_intq)

/* Tensor #352 */
AI_TENSOR_OBJ_DECLARE(
  transpose_105_0_0_gemm_111_conversion_output, AI_STATIC,
  352, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 100, 4), AI_STRIDE_INIT(4, 4, 4, 256, 25600),
  1, &transpose_105_0_0_gemm_111_conversion_output_array, NULL)

/* Tensor #353 */
AI_TENSOR_OBJ_DECLARE(
  transpose_105_output, AI_STATIC,
  353, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 100, 4), AI_STRIDE_INIT(4, 1, 1, 64, 6400),
  1, &transpose_105_output_array, &transpose_105_output_array_intq)

/* Tensor #354 */
AI_TENSOR_OBJ_DECLARE(
  transpose_118_0_1_gemm_119_conversion_output, AI_STATIC,
  354, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 100, 4), AI_STRIDE_INIT(4, 4, 4, 256, 25600),
  1, &transpose_118_0_1_gemm_119_conversion_output_array, NULL)

/* Tensor #355 */
AI_TENSOR_OBJ_DECLARE(
  transpose_118_output, AI_STATIC,
  355, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 100, 4), AI_STRIDE_INIT(4, 1, 1, 64, 6400),
  1, &transpose_118_output_array, &transpose_118_output_array_intq)

/* Tensor #356 */
AI_TENSOR_OBJ_DECLARE(
  transpose_120_output, AI_STATIC,
  356, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 4, 100), AI_STRIDE_INIT(4, 1, 1, 64, 256),
  1, &transpose_120_output_array, &transpose_120_output_array_intq)

/* Tensor #357 */
AI_TENSOR_OBJ_DECLARE(
  transpose_120_output0, AI_STATIC,
  357, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &transpose_120_output_array, &transpose_120_output_array_intq)

/* Tensor #358 */
AI_TENSOR_OBJ_DECLARE(
  transpose_18_0_0_gemm_24_conversion_output, AI_STATIC,
  358, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 100, 4), AI_STRIDE_INIT(4, 4, 4, 256, 25600),
  1, &transpose_18_0_0_gemm_24_conversion_output_array, NULL)

/* Tensor #359 */
AI_TENSOR_OBJ_DECLARE(
  transpose_18_output, AI_STATIC,
  359, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 100, 4), AI_STRIDE_INIT(4, 1, 1, 64, 6400),
  1, &transpose_18_output_array, &transpose_18_output_array_intq)

/* Tensor #360 */
AI_TENSOR_OBJ_DECLARE(
  transpose_192_0_0_gemm_198_conversion_output, AI_STATIC,
  360, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 100, 4), AI_STRIDE_INIT(4, 4, 4, 256, 25600),
  1, &transpose_192_0_0_gemm_198_conversion_output_array, NULL)

/* Tensor #361 */
AI_TENSOR_OBJ_DECLARE(
  transpose_192_output, AI_STATIC,
  361, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 100, 4), AI_STRIDE_INIT(4, 1, 1, 64, 6400),
  1, &transpose_192_output_array, &transpose_192_output_array_intq)

/* Tensor #362 */
AI_TENSOR_OBJ_DECLARE(
  transpose_205_0_1_gemm_206_conversion_output, AI_STATIC,
  362, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 100, 4), AI_STRIDE_INIT(4, 4, 4, 256, 25600),
  1, &transpose_205_0_1_gemm_206_conversion_output_array, NULL)

/* Tensor #363 */
AI_TENSOR_OBJ_DECLARE(
  transpose_205_output, AI_STATIC,
  363, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 100, 4), AI_STRIDE_INIT(4, 1, 1, 64, 6400),
  1, &transpose_205_output_array, &transpose_205_output_array_intq)

/* Tensor #364 */
AI_TENSOR_OBJ_DECLARE(
  transpose_207_output, AI_STATIC,
  364, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 4, 100), AI_STRIDE_INIT(4, 1, 1, 64, 256),
  1, &transpose_207_output_array, &transpose_207_output_array_intq)

/* Tensor #365 */
AI_TENSOR_OBJ_DECLARE(
  transpose_207_output0, AI_STATIC,
  365, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &transpose_207_output_array, &transpose_207_output_array_intq)

/* Tensor #366 */
AI_TENSOR_OBJ_DECLARE(
  transpose_279_0_0_gemm_285_conversion_output, AI_STATIC,
  366, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 100, 4), AI_STRIDE_INIT(4, 4, 4, 256, 25600),
  1, &transpose_279_0_0_gemm_285_conversion_output_array, NULL)

/* Tensor #367 */
AI_TENSOR_OBJ_DECLARE(
  transpose_279_output, AI_STATIC,
  367, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 100, 4), AI_STRIDE_INIT(4, 1, 1, 64, 6400),
  1, &transpose_279_output_array, &transpose_279_output_array_intq)

/* Tensor #368 */
AI_TENSOR_OBJ_DECLARE(
  transpose_292_0_1_gemm_293_conversion_output, AI_STATIC,
  368, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 100, 4), AI_STRIDE_INIT(4, 4, 4, 256, 25600),
  1, &transpose_292_0_1_gemm_293_conversion_output_array, NULL)

/* Tensor #369 */
AI_TENSOR_OBJ_DECLARE(
  transpose_292_output, AI_STATIC,
  369, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 100, 4), AI_STRIDE_INIT(4, 1, 1, 64, 6400),
  1, &transpose_292_output_array, &transpose_292_output_array_intq)

/* Tensor #370 */
AI_TENSOR_OBJ_DECLARE(
  transpose_294_output, AI_STATIC,
  370, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 4, 100), AI_STRIDE_INIT(4, 1, 1, 64, 256),
  1, &transpose_294_output_array, &transpose_294_output_array_intq)

/* Tensor #371 */
AI_TENSOR_OBJ_DECLARE(
  transpose_294_output0, AI_STATIC,
  371, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &transpose_294_output_array, &transpose_294_output_array_intq)

/* Tensor #372 */
AI_TENSOR_OBJ_DECLARE(
  transpose_31_0_1_gemm_32_conversion_output, AI_STATIC,
  372, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 100, 4), AI_STRIDE_INIT(4, 4, 4, 256, 25600),
  1, &transpose_31_0_1_gemm_32_conversion_output_array, NULL)

/* Tensor #373 */
AI_TENSOR_OBJ_DECLARE(
  transpose_31_output, AI_STATIC,
  373, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 100, 4), AI_STRIDE_INIT(4, 1, 1, 64, 6400),
  1, &transpose_31_output_array, &transpose_31_output_array_intq)

/* Tensor #374 */
AI_TENSOR_OBJ_DECLARE(
  transpose_33_output, AI_STATIC,
  374, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 4, 100), AI_STRIDE_INIT(4, 1, 1, 64, 256),
  1, &transpose_33_output_array, &transpose_33_output_array_intq)

/* Tensor #375 */
AI_TENSOR_OBJ_DECLARE(
  transpose_33_output0, AI_STATIC,
  375, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 100), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &transpose_33_output_array, &transpose_33_output_array_intq)

/* Tensor #376 */
AI_TENSOR_OBJ_DECLARE(
  transpose_bgemm_111_out_0_1_gemm_111_conversion_output, AI_STATIC,
  376, 0x0,
  AI_SHAPE_INIT(4, 1, 100, 64, 4), AI_STRIDE_INIT(4, 4, 4, 400, 25600),
  1, &transpose_bgemm_111_out_0_1_gemm_111_conversion_output_array, NULL)

/* Tensor #377 */
AI_TENSOR_OBJ_DECLARE(
  transpose_bgemm_111_out_output, AI_STATIC,
  377, 0x1,
  AI_SHAPE_INIT(4, 1, 100, 64, 4), AI_STRIDE_INIT(4, 1, 1, 100, 6400),
  1, &transpose_bgemm_111_out_output_array, &transpose_bgemm_111_out_output_array_intq)

/* Tensor #378 */
AI_TENSOR_OBJ_DECLARE(
  transpose_bgemm_198_out_0_1_gemm_198_conversion_output, AI_STATIC,
  378, 0x0,
  AI_SHAPE_INIT(4, 1, 100, 64, 4), AI_STRIDE_INIT(4, 4, 4, 400, 25600),
  1, &transpose_bgemm_198_out_0_1_gemm_198_conversion_output_array, NULL)

/* Tensor #379 */
AI_TENSOR_OBJ_DECLARE(
  transpose_bgemm_198_out_output, AI_STATIC,
  379, 0x1,
  AI_SHAPE_INIT(4, 1, 100, 64, 4), AI_STRIDE_INIT(4, 1, 1, 100, 6400),
  1, &transpose_bgemm_198_out_output_array, &transpose_bgemm_198_out_output_array_intq)

/* Tensor #380 */
AI_TENSOR_OBJ_DECLARE(
  transpose_bgemm_24_out_0_1_gemm_24_conversion_output, AI_STATIC,
  380, 0x0,
  AI_SHAPE_INIT(4, 1, 100, 64, 4), AI_STRIDE_INIT(4, 4, 4, 400, 25600),
  1, &transpose_bgemm_24_out_0_1_gemm_24_conversion_output_array, NULL)

/* Tensor #381 */
AI_TENSOR_OBJ_DECLARE(
  transpose_bgemm_24_out_output, AI_STATIC,
  381, 0x1,
  AI_SHAPE_INIT(4, 1, 100, 64, 4), AI_STRIDE_INIT(4, 1, 1, 100, 6400),
  1, &transpose_bgemm_24_out_output_array, &transpose_bgemm_24_out_output_array_intq)

/* Tensor #382 */
AI_TENSOR_OBJ_DECLARE(
  transpose_bgemm_285_out_0_1_gemm_285_conversion_output, AI_STATIC,
  382, 0x0,
  AI_SHAPE_INIT(4, 1, 100, 64, 4), AI_STRIDE_INIT(4, 4, 4, 400, 25600),
  1, &transpose_bgemm_285_out_0_1_gemm_285_conversion_output_array, NULL)

/* Tensor #383 */
AI_TENSOR_OBJ_DECLARE(
  transpose_bgemm_285_out_output, AI_STATIC,
  383, 0x1,
  AI_SHAPE_INIT(4, 1, 100, 64, 4), AI_STRIDE_INIT(4, 1, 1, 100, 6400),
  1, &transpose_bgemm_285_out_output_array, &transpose_bgemm_285_out_output_array_intq)



/**  Layer declarations section  **********************************************/


AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_361_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_359_output, &dense_24_bias_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_361_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_361_layer, 361,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_361_chain,
  NULL, &eltwise_361_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_359_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_350_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_359_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &gemm_359_weights, &gemm_359_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_359_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_359_layer, 359,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &gemm_359_chain,
  NULL, &eltwise_361_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_350_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_347_output, &eltwise_349_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_350_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_350_layer, 350,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_350_chain,
  NULL, &gemm_359_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_347_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_340_output, &eltwise_346_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_347_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_347_layer, 347,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_347_chain,
  NULL, &eltwise_350_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_349_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &encoder_layer_3_layer_normalization_23_beta_3D, &eltwise_348_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_349_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_349_layer, 349,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_349_chain,
  NULL, &eltwise_347_layer, AI_STATIC, 
  .operation = ai_sub_f32, 
  .buffer_operation = ai_sub_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_348_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &reduce_341_Mul_0_1_eltwise_342_conversion_output, &eltwise_346_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_348_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_348_layer, 348,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_348_chain,
  NULL, &eltwise_349_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_346_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &nl_345_output, &encoder_layer_3_layer_normalization_23_gamma_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_346_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_346_layer, 346,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_346_chain,
  NULL, &eltwise_348_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)


AI_STATIC_CONST ai_i8 nl_345_nl_params_data[] = { -128, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 126 };
AI_ARRAY_OBJ_DECLARE(
    nl_345_nl_params, AI_ARRAY_FORMAT_S8,
    nl_345_nl_params_data, nl_345_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_345_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_344_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_345_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_345_layer, 345,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_345_chain,
  NULL, &eltwise_346_layer, AI_STATIC, 
  .nl_params = &nl_345_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_344_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &reduce_343_Mul_0_0_eltwise_344_conversion_output, &tiny_bert_generator_encoder_layer_layer_normalization_16_batchnorm_add_y_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_344_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_344_layer, 344,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_344_chain,
  NULL, &nl_345_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_343_Mul_0_0_eltwise_344_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_343_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_343_Mul_0_0_eltwise_344_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_343_Mul_0_0_eltwise_344_conversion_layer, 343,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &reduce_343_Mul_0_0_eltwise_344_conversion_chain,
  NULL, &eltwise_344_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_343_Mul_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_343_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_343_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &reduce_47_Mul_scale, &reduce_47_Mul_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_343_Mul_layer, 343,
  BN_TYPE, 0x0, NULL,
  bn, forward_bn,
  &reduce_343_Mul_chain,
  NULL, &reduce_343_Mul_0_0_eltwise_344_conversion_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_float reduce_343_neutral_value_data[] = { 0.0f };
AI_ARRAY_OBJ_DECLARE(
    reduce_343_neutral_value, AI_ARRAY_FORMAT_FLOAT,
    reduce_343_neutral_value_data, reduce_343_neutral_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_343_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_342_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_343_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_343_layer, 343,
  REDUCE_TYPE, 0x0, NULL,
  reduce, forward_reduce,
  &reduce_343_chain,
  NULL, &reduce_343_Mul_layer, AI_STATIC, 
  .operation = ai_sum, 
  .neutral_value = &reduce_343_neutral_value, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_342_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_340_0_0_reduce_341_conversion_output, &reduce_341_Mul_0_1_eltwise_342_conversion_0_1_eltwise_342_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_342_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_342_layer, 342,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_342_chain,
  NULL, &reduce_343_layer, AI_STATIC, 
  .operation = ai_squared_diff, 
  .buffer_operation = ai_squared_diff_buffer, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_341_Mul_0_1_eltwise_342_conversion_0_1_eltwise_342_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_341_Mul_0_1_eltwise_342_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_341_Mul_0_1_eltwise_342_conversion_0_1_eltwise_342_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_341_Mul_0_1_eltwise_342_conversion_0_1_eltwise_342_conversion_layer, 341,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &reduce_341_Mul_0_1_eltwise_342_conversion_0_1_eltwise_342_conversion_chain,
  NULL, &eltwise_342_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_341_Mul_0_1_eltwise_342_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_341_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_341_Mul_0_1_eltwise_342_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_341_Mul_0_1_eltwise_342_conversion_layer, 341,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &reduce_341_Mul_0_1_eltwise_342_conversion_chain,
  NULL, &reduce_341_Mul_0_1_eltwise_342_conversion_0_1_eltwise_342_conversion_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_341_Mul_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_341_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_341_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &reduce_47_Mul_scale, &reduce_47_Mul_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_341_Mul_layer, 341,
  BN_TYPE, 0x0, NULL,
  bn, forward_bn,
  &reduce_341_Mul_chain,
  NULL, &reduce_341_Mul_0_1_eltwise_342_conversion_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_float reduce_341_neutral_value_data[] = { 0.0f };
AI_ARRAY_OBJ_DECLARE(
    reduce_341_neutral_value, AI_ARRAY_FORMAT_FLOAT,
    reduce_341_neutral_value_data, reduce_341_neutral_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_341_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_340_0_0_reduce_341_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_341_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_341_layer, 341,
  REDUCE_TYPE, 0x0, NULL,
  reduce, forward_reduce,
  &reduce_341_chain,
  NULL, &reduce_341_Mul_layer, AI_STATIC, 
  .operation = ai_sum, 
  .neutral_value = &reduce_341_neutral_value, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_340_0_0_reduce_341_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_340_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_340_0_0_reduce_341_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_340_0_0_reduce_341_conversion_layer, 340,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &eltwise_340_0_0_reduce_341_conversion_chain,
  NULL, &reduce_341_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_340_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_317_output, &eltwise_339_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_340_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_340_layer, 340,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_340_chain,
  NULL, &eltwise_340_0_0_reduce_341_conversion_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_339_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_337_output, &dense_72_bias_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_339_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_339_layer, 339,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_339_chain,
  NULL, &eltwise_340_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_337_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_328_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_337_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &gemm_337_weights, &gemm_27_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_337_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_337_layer, 337,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &gemm_337_chain,
  NULL, &eltwise_339_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_328_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_326_output, &dense_71_bias_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_328_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_328_layer, 328,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_328_chain,
  NULL, &gemm_337_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_326_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_317_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_326_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &gemm_326_weights, &gemm_65_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_326_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_326_layer, 326,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &gemm_326_chain,
  NULL, &eltwise_328_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_317_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_314_output, &eltwise_316_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_317_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_317_layer, 317,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_317_chain,
  NULL, &gemm_326_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_314_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_307_output, &eltwise_313_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_314_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_314_layer, 314,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_314_chain,
  NULL, &eltwise_317_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_316_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &encoder_layer_3_layer_normalization_22_beta_3D, &eltwise_315_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_316_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_316_layer, 316,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_316_chain,
  NULL, &eltwise_314_layer, AI_STATIC, 
  .operation = ai_sub_f32, 
  .buffer_operation = ai_sub_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_315_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &reduce_308_Mul_0_1_eltwise_309_conversion_output, &eltwise_313_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_315_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_315_layer, 315,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_315_chain,
  NULL, &eltwise_316_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_313_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &nl_312_output, &encoder_layer_3_layer_normalization_22_gamma_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_313_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_313_layer, 313,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_313_chain,
  NULL, &eltwise_315_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)


AI_STATIC_CONST ai_i8 nl_312_nl_params_data[] = { -128, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 126 };
AI_ARRAY_OBJ_DECLARE(
    nl_312_nl_params, AI_ARRAY_FORMAT_S8,
    nl_312_nl_params_data, nl_312_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_312_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_311_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_312_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_312_layer, 312,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_312_chain,
  NULL, &eltwise_313_layer, AI_STATIC, 
  .nl_params = &nl_312_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_311_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &reduce_310_Mul_0_0_eltwise_311_conversion_output, &tiny_bert_generator_encoder_layer_layer_normalization_16_batchnorm_add_y_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_311_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_311_layer, 311,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_311_chain,
  NULL, &nl_312_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_310_Mul_0_0_eltwise_311_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_310_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_310_Mul_0_0_eltwise_311_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_310_Mul_0_0_eltwise_311_conversion_layer, 310,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &reduce_310_Mul_0_0_eltwise_311_conversion_chain,
  NULL, &eltwise_311_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_310_Mul_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_310_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_310_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &reduce_47_Mul_scale, &reduce_47_Mul_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_310_Mul_layer, 310,
  BN_TYPE, 0x0, NULL,
  bn, forward_bn,
  &reduce_310_Mul_chain,
  NULL, &reduce_310_Mul_0_0_eltwise_311_conversion_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_float reduce_310_neutral_value_data[] = { 0.0f };
AI_ARRAY_OBJ_DECLARE(
    reduce_310_neutral_value, AI_ARRAY_FORMAT_FLOAT,
    reduce_310_neutral_value_data, reduce_310_neutral_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_310_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_309_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_310_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_310_layer, 310,
  REDUCE_TYPE, 0x0, NULL,
  reduce, forward_reduce,
  &reduce_310_chain,
  NULL, &reduce_310_Mul_layer, AI_STATIC, 
  .operation = ai_sum, 
  .neutral_value = &reduce_310_neutral_value, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_309_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_307_0_0_reduce_308_conversion_output, &reduce_308_Mul_0_1_eltwise_309_conversion_0_1_eltwise_309_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_309_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_309_layer, 309,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_309_chain,
  NULL, &reduce_310_layer, AI_STATIC, 
  .operation = ai_squared_diff, 
  .buffer_operation = ai_squared_diff_buffer, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_308_Mul_0_1_eltwise_309_conversion_0_1_eltwise_309_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_308_Mul_0_1_eltwise_309_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_308_Mul_0_1_eltwise_309_conversion_0_1_eltwise_309_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_308_Mul_0_1_eltwise_309_conversion_0_1_eltwise_309_conversion_layer, 308,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &reduce_308_Mul_0_1_eltwise_309_conversion_0_1_eltwise_309_conversion_chain,
  NULL, &eltwise_309_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_308_Mul_0_1_eltwise_309_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_308_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_308_Mul_0_1_eltwise_309_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_308_Mul_0_1_eltwise_309_conversion_layer, 308,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &reduce_308_Mul_0_1_eltwise_309_conversion_chain,
  NULL, &reduce_308_Mul_0_1_eltwise_309_conversion_0_1_eltwise_309_conversion_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_308_Mul_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_308_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_308_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &reduce_47_Mul_scale, &reduce_47_Mul_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_308_Mul_layer, 308,
  BN_TYPE, 0x0, NULL,
  bn, forward_bn,
  &reduce_308_Mul_chain,
  NULL, &reduce_308_Mul_0_1_eltwise_309_conversion_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_float reduce_308_neutral_value_data[] = { 0.0f };
AI_ARRAY_OBJ_DECLARE(
    reduce_308_neutral_value, AI_ARRAY_FORMAT_FLOAT,
    reduce_308_neutral_value_data, reduce_308_neutral_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_308_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_307_0_0_reduce_308_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_308_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_308_layer, 308,
  REDUCE_TYPE, 0x0, NULL,
  reduce, forward_reduce,
  &reduce_308_chain,
  NULL, &reduce_308_Mul_layer, AI_STATIC, 
  .operation = ai_sum, 
  .neutral_value = &reduce_308_neutral_value, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_307_0_0_reduce_308_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_307_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_307_0_0_reduce_308_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_307_0_0_reduce_308_conversion_layer, 307,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &eltwise_307_0_0_reduce_308_conversion_chain,
  NULL, &reduce_308_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_307_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_263_output, &eltwise_306_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_307_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_307_layer, 307,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_307_chain,
  NULL, &eltwise_307_0_0_reduce_308_conversion_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_306_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_304_output, &encoder_layer_3_multi_head_attention_11_dense_70_bias_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_306_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_306_layer, 306,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_306_chain,
  NULL, &eltwise_307_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_304_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_294_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_304_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &gemm_304_weights, &gemm_27_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_304_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_304_layer, 304,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &gemm_304_chain,
  NULL, &eltwise_306_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  transpose_294_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_293_0_0_transpose_294_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_294_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  transpose_294_layer, 294,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &transpose_294_chain,
  NULL, &gemm_304_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_CHANNEL, AI_SHAPE_HEIGHT, AI_SHAPE_WIDTH, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_293_0_0_transpose_294_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_293_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_293_0_0_transpose_294_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_293_0_0_transpose_294_conversion_layer, 293,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &gemm_293_0_0_transpose_294_conversion_chain,
  NULL, &transpose_294_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_293_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &nl_287_0_0_gemm_293_conversion_output, &transpose_292_0_1_gemm_293_conversion_output, &gemm_293_bias_0_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_293_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_293_layer, 293,
  MATMUL_TYPE, 0x0, NULL,
  matmul, forward_matmul,
  &gemm_293_chain,
  NULL, &gemm_293_0_0_transpose_294_conversion_layer, AI_STATIC, 
  .alpha = 1.0, 
  .beta = 1.0, 
  .tA = 0, 
  .tB = 0, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_287_0_0_gemm_293_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_287_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_287_0_0_gemm_293_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_287_0_0_gemm_293_conversion_layer, 287,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &nl_287_0_0_gemm_293_conversion_chain,
  NULL, &gemm_293_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i32 nl_287_nl_params_data[] = { 1103115776, 20, -1984 };
AI_ARRAY_OBJ_DECLARE(
    nl_287_nl_params, AI_ARRAY_FORMAT_S32,
    nl_287_nl_params_data, nl_287_nl_params_data, 3, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_287_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_286_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_287_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_287_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  nl_287_layer, 287,
  SM_TYPE, 0x0, NULL,
  sm, forward_sm_integer,
  &nl_287_chain,
  NULL, &nl_287_0_0_gemm_293_conversion_layer, AI_STATIC, 
  .nl_params = &nl_287_nl_params, 
  .axis = AI_SHAPE_CHANNEL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_286_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_285_0_0_eltwise_286_conversion_output, &tiny_bert_generator_encoder_layer_3_multi_head_attention_11_truedivtiny_bert_generator_encoder_layer_3_multi_head_attention_11_Sqrt_4D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_286_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_286_layer, 286,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_286_chain,
  NULL, &nl_287_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_285_0_0_eltwise_286_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_285_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_285_0_0_eltwise_286_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_285_0_0_eltwise_286_conversion_layer, 285,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &gemm_285_0_0_eltwise_286_conversion_chain,
  NULL, &eltwise_286_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_285_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &transpose_279_0_0_gemm_285_conversion_output, &transpose_bgemm_285_out_0_1_gemm_285_conversion_output, &gemm_285_bias_0_2_gemm_285_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_285_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_285_layer, 285,
  MATMUL_TYPE, 0x0, NULL,
  matmul, forward_matmul,
  &gemm_285_chain,
  NULL, &gemm_285_0_0_eltwise_286_conversion_layer, AI_STATIC, 
  .alpha = 1.0, 
  .beta = 1.0, 
  .tA = 0, 
  .tB = 0, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  transpose_279_0_0_gemm_285_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_279_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_279_0_0_gemm_285_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  transpose_279_0_0_gemm_285_conversion_layer, 279,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &transpose_279_0_0_gemm_285_conversion_chain,
  NULL, &gemm_285_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  transpose_279_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_277_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_279_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  transpose_279_layer, 279,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &transpose_279_chain,
  NULL, &transpose_279_0_0_gemm_285_conversion_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_CHANNEL, AI_SHAPE_HEIGHT, AI_SHAPE_WIDTH, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_277_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_275_output, &encoder_layer_3_multi_head_attention_11_dense_67_bias_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_277_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_277_layer, 277,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_277_chain,
  NULL, &transpose_279_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_275_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_263_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_275_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &gemm_275_weights, &gemm_27_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_275_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_275_layer, 275,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &gemm_275_chain,
  NULL, &eltwise_277_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  transpose_bgemm_285_out_0_1_gemm_285_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_bgemm_285_out_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_bgemm_285_out_0_1_gemm_285_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  transpose_bgemm_285_out_0_1_gemm_285_conversion_layer, 285,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &transpose_bgemm_285_out_0_1_gemm_285_conversion_chain,
  NULL, &gemm_275_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  transpose_bgemm_285_out_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_282_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_bgemm_285_out_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  transpose_bgemm_285_out_layer, 285,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &transpose_bgemm_285_out_chain,
  NULL, &transpose_bgemm_285_out_0_1_gemm_285_conversion_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_HEIGHT, AI_SHAPE_CHANNEL, AI_SHAPE_WIDTH, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_282_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_280_output, &encoder_layer_3_multi_head_attention_11_dense_68_bias_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_282_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_282_layer, 282,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_282_chain,
  NULL, &transpose_bgemm_285_out_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_280_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_263_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_280_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &gemm_280_weights, &gemm_27_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_280_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_280_layer, 280,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &gemm_280_chain,
  NULL, &eltwise_282_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  transpose_292_0_1_gemm_293_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_292_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_292_0_1_gemm_293_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  transpose_292_0_1_gemm_293_conversion_layer, 292,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &transpose_292_0_1_gemm_293_conversion_chain,
  NULL, &gemm_280_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  transpose_292_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_290_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_292_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  transpose_292_layer, 292,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &transpose_292_chain,
  NULL, &transpose_292_0_1_gemm_293_conversion_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_CHANNEL, AI_SHAPE_HEIGHT, AI_SHAPE_WIDTH, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_290_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_288_output, &encoder_layer_3_multi_head_attention_11_dense_69_bias_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_290_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_290_layer, 290,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_290_chain,
  NULL, &transpose_292_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_288_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_263_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_288_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &gemm_288_weights, &gemm_27_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_288_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_288_layer, 288,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &gemm_288_chain,
  NULL, &eltwise_290_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_263_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_260_output, &eltwise_262_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_263_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_263_layer, 263,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_263_chain,
  NULL, &gemm_288_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_260_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_253_output, &eltwise_259_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_260_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_260_layer, 260,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_260_chain,
  NULL, &eltwise_263_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_262_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &encoder_layer_2_layer_normalization_21_beta_3D, &eltwise_261_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_262_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_262_layer, 262,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_262_chain,
  NULL, &eltwise_260_layer, AI_STATIC, 
  .operation = ai_sub_f32, 
  .buffer_operation = ai_sub_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_261_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &reduce_254_Mul_0_1_eltwise_255_conversion_output, &eltwise_259_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_261_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_261_layer, 261,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_261_chain,
  NULL, &eltwise_262_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_259_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &nl_258_output, &encoder_layer_2_layer_normalization_21_gamma_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_259_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_259_layer, 259,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_259_chain,
  NULL, &eltwise_261_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)


AI_STATIC_CONST ai_i8 nl_258_nl_params_data[] = { -128, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 126, 126, 125, 125 };
AI_ARRAY_OBJ_DECLARE(
    nl_258_nl_params, AI_ARRAY_FORMAT_S8,
    nl_258_nl_params_data, nl_258_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_258_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_257_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_258_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_258_layer, 258,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_258_chain,
  NULL, &eltwise_259_layer, AI_STATIC, 
  .nl_params = &nl_258_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_257_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &reduce_256_Mul_0_0_eltwise_257_conversion_output, &tiny_bert_generator_encoder_layer_layer_normalization_16_batchnorm_add_y_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_257_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_257_layer, 257,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_257_chain,
  NULL, &nl_258_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_256_Mul_0_0_eltwise_257_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_256_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_256_Mul_0_0_eltwise_257_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_256_Mul_0_0_eltwise_257_conversion_layer, 256,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &reduce_256_Mul_0_0_eltwise_257_conversion_chain,
  NULL, &eltwise_257_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_256_Mul_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_256_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_256_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &reduce_47_Mul_scale, &reduce_47_Mul_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_256_Mul_layer, 256,
  BN_TYPE, 0x0, NULL,
  bn, forward_bn,
  &reduce_256_Mul_chain,
  NULL, &reduce_256_Mul_0_0_eltwise_257_conversion_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_float reduce_256_neutral_value_data[] = { 0.0f };
AI_ARRAY_OBJ_DECLARE(
    reduce_256_neutral_value, AI_ARRAY_FORMAT_FLOAT,
    reduce_256_neutral_value_data, reduce_256_neutral_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_256_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_255_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_256_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_256_layer, 256,
  REDUCE_TYPE, 0x0, NULL,
  reduce, forward_reduce,
  &reduce_256_chain,
  NULL, &reduce_256_Mul_layer, AI_STATIC, 
  .operation = ai_sum, 
  .neutral_value = &reduce_256_neutral_value, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_255_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_253_0_0_reduce_254_conversion_output, &reduce_254_Mul_0_1_eltwise_255_conversion_0_1_eltwise_255_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_255_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_255_layer, 255,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_255_chain,
  NULL, &reduce_256_layer, AI_STATIC, 
  .operation = ai_squared_diff, 
  .buffer_operation = ai_squared_diff_buffer, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_254_Mul_0_1_eltwise_255_conversion_0_1_eltwise_255_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_254_Mul_0_1_eltwise_255_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_254_Mul_0_1_eltwise_255_conversion_0_1_eltwise_255_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_254_Mul_0_1_eltwise_255_conversion_0_1_eltwise_255_conversion_layer, 254,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &reduce_254_Mul_0_1_eltwise_255_conversion_0_1_eltwise_255_conversion_chain,
  NULL, &eltwise_255_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_254_Mul_0_1_eltwise_255_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_254_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_254_Mul_0_1_eltwise_255_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_254_Mul_0_1_eltwise_255_conversion_layer, 254,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &reduce_254_Mul_0_1_eltwise_255_conversion_chain,
  NULL, &reduce_254_Mul_0_1_eltwise_255_conversion_0_1_eltwise_255_conversion_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_254_Mul_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_254_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_254_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &reduce_47_Mul_scale, &reduce_47_Mul_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_254_Mul_layer, 254,
  BN_TYPE, 0x0, NULL,
  bn, forward_bn,
  &reduce_254_Mul_chain,
  NULL, &reduce_254_Mul_0_1_eltwise_255_conversion_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_float reduce_254_neutral_value_data[] = { 0.0f };
AI_ARRAY_OBJ_DECLARE(
    reduce_254_neutral_value, AI_ARRAY_FORMAT_FLOAT,
    reduce_254_neutral_value_data, reduce_254_neutral_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_254_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_253_0_0_reduce_254_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_254_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_254_layer, 254,
  REDUCE_TYPE, 0x0, NULL,
  reduce, forward_reduce,
  &reduce_254_chain,
  NULL, &reduce_254_Mul_layer, AI_STATIC, 
  .operation = ai_sum, 
  .neutral_value = &reduce_254_neutral_value, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_253_0_0_reduce_254_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_253_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_253_0_0_reduce_254_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_253_0_0_reduce_254_conversion_layer, 253,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &eltwise_253_0_0_reduce_254_conversion_chain,
  NULL, &reduce_254_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_253_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_230_output, &eltwise_252_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_253_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_253_layer, 253,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_253_chain,
  NULL, &eltwise_253_0_0_reduce_254_conversion_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_252_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_250_output, &dense_66_bias_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_252_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_252_layer, 252,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_252_chain,
  NULL, &eltwise_253_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_250_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_241_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_250_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &gemm_250_weights, &gemm_27_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_250_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_250_layer, 250,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &gemm_250_chain,
  NULL, &eltwise_252_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_241_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_239_output, &dense_65_bias_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_241_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_241_layer, 241,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_241_chain,
  NULL, &gemm_250_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_239_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_230_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_239_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &gemm_239_weights, &gemm_65_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_239_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_239_layer, 239,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &gemm_239_chain,
  NULL, &eltwise_241_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_230_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_227_output, &eltwise_229_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_230_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_230_layer, 230,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_230_chain,
  NULL, &gemm_239_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_227_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_220_output, &eltwise_226_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_227_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_227_layer, 227,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_227_chain,
  NULL, &eltwise_230_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_229_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &encoder_layer_2_layer_normalization_20_beta_3D, &eltwise_228_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_229_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_229_layer, 229,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_229_chain,
  NULL, &eltwise_227_layer, AI_STATIC, 
  .operation = ai_sub_f32, 
  .buffer_operation = ai_sub_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_228_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &reduce_221_Mul_0_1_eltwise_222_conversion_output, &eltwise_226_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_228_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_228_layer, 228,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_228_chain,
  NULL, &eltwise_229_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_226_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &nl_225_output, &encoder_layer_2_layer_normalization_20_gamma_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_226_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_226_layer, 226,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_226_chain,
  NULL, &eltwise_228_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)


AI_STATIC_CONST ai_i8 nl_225_nl_params_data[] = { -128, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 126, 126, 125, 125 };
AI_ARRAY_OBJ_DECLARE(
    nl_225_nl_params, AI_ARRAY_FORMAT_S8,
    nl_225_nl_params_data, nl_225_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_225_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_224_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_225_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_225_layer, 225,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_225_chain,
  NULL, &eltwise_226_layer, AI_STATIC, 
  .nl_params = &nl_225_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_224_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &reduce_223_Mul_0_0_eltwise_224_conversion_output, &tiny_bert_generator_encoder_layer_layer_normalization_16_batchnorm_add_y_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_224_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_224_layer, 224,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_224_chain,
  NULL, &nl_225_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_223_Mul_0_0_eltwise_224_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_223_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_223_Mul_0_0_eltwise_224_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_223_Mul_0_0_eltwise_224_conversion_layer, 223,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &reduce_223_Mul_0_0_eltwise_224_conversion_chain,
  NULL, &eltwise_224_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_223_Mul_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_223_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_223_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &reduce_47_Mul_scale, &reduce_47_Mul_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_223_Mul_layer, 223,
  BN_TYPE, 0x0, NULL,
  bn, forward_bn,
  &reduce_223_Mul_chain,
  NULL, &reduce_223_Mul_0_0_eltwise_224_conversion_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_float reduce_223_neutral_value_data[] = { 0.0f };
AI_ARRAY_OBJ_DECLARE(
    reduce_223_neutral_value, AI_ARRAY_FORMAT_FLOAT,
    reduce_223_neutral_value_data, reduce_223_neutral_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_223_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_222_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_223_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_223_layer, 223,
  REDUCE_TYPE, 0x0, NULL,
  reduce, forward_reduce,
  &reduce_223_chain,
  NULL, &reduce_223_Mul_layer, AI_STATIC, 
  .operation = ai_sum, 
  .neutral_value = &reduce_223_neutral_value, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_222_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_220_0_0_reduce_221_conversion_output, &reduce_221_Mul_0_1_eltwise_222_conversion_0_1_eltwise_222_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_222_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_222_layer, 222,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_222_chain,
  NULL, &reduce_223_layer, AI_STATIC, 
  .operation = ai_squared_diff, 
  .buffer_operation = ai_squared_diff_buffer, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_221_Mul_0_1_eltwise_222_conversion_0_1_eltwise_222_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_221_Mul_0_1_eltwise_222_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_221_Mul_0_1_eltwise_222_conversion_0_1_eltwise_222_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_221_Mul_0_1_eltwise_222_conversion_0_1_eltwise_222_conversion_layer, 221,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &reduce_221_Mul_0_1_eltwise_222_conversion_0_1_eltwise_222_conversion_chain,
  NULL, &eltwise_222_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_221_Mul_0_1_eltwise_222_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_221_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_221_Mul_0_1_eltwise_222_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_221_Mul_0_1_eltwise_222_conversion_layer, 221,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &reduce_221_Mul_0_1_eltwise_222_conversion_chain,
  NULL, &reduce_221_Mul_0_1_eltwise_222_conversion_0_1_eltwise_222_conversion_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_221_Mul_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_221_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_221_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &reduce_47_Mul_scale, &reduce_47_Mul_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_221_Mul_layer, 221,
  BN_TYPE, 0x0, NULL,
  bn, forward_bn,
  &reduce_221_Mul_chain,
  NULL, &reduce_221_Mul_0_1_eltwise_222_conversion_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_float reduce_221_neutral_value_data[] = { 0.0f };
AI_ARRAY_OBJ_DECLARE(
    reduce_221_neutral_value, AI_ARRAY_FORMAT_FLOAT,
    reduce_221_neutral_value_data, reduce_221_neutral_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_221_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_220_0_0_reduce_221_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_221_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_221_layer, 221,
  REDUCE_TYPE, 0x0, NULL,
  reduce, forward_reduce,
  &reduce_221_chain,
  NULL, &reduce_221_Mul_layer, AI_STATIC, 
  .operation = ai_sum, 
  .neutral_value = &reduce_221_neutral_value, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_220_0_0_reduce_221_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_220_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_220_0_0_reduce_221_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_220_0_0_reduce_221_conversion_layer, 220,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &eltwise_220_0_0_reduce_221_conversion_chain,
  NULL, &reduce_221_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_220_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_176_output, &eltwise_219_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_220_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_220_layer, 220,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_220_chain,
  NULL, &eltwise_220_0_0_reduce_221_conversion_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_219_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_217_output, &encoder_layer_2_multi_head_attention_10_dense_64_bias_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_219_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_219_layer, 219,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_219_chain,
  NULL, &eltwise_220_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_217_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_207_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_217_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &gemm_217_weights, &gemm_27_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_217_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_217_layer, 217,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &gemm_217_chain,
  NULL, &eltwise_219_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  transpose_207_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_206_0_0_transpose_207_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_207_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  transpose_207_layer, 207,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &transpose_207_chain,
  NULL, &gemm_217_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_CHANNEL, AI_SHAPE_HEIGHT, AI_SHAPE_WIDTH, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_206_0_0_transpose_207_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_206_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_206_0_0_transpose_207_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_206_0_0_transpose_207_conversion_layer, 206,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &gemm_206_0_0_transpose_207_conversion_chain,
  NULL, &transpose_207_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_206_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &nl_200_0_0_gemm_206_conversion_output, &transpose_205_0_1_gemm_206_conversion_output, &gemm_206_bias_0_2_gemm_206_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_206_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_206_layer, 206,
  MATMUL_TYPE, 0x0, NULL,
  matmul, forward_matmul,
  &gemm_206_chain,
  NULL, &gemm_206_0_0_transpose_207_conversion_layer, AI_STATIC, 
  .alpha = 1.0, 
  .beta = 1.0, 
  .tA = 0, 
  .tB = 0, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_200_0_0_gemm_206_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_200_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_200_0_0_gemm_206_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_200_0_0_gemm_206_conversion_layer, 200,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &nl_200_0_0_gemm_206_conversion_chain,
  NULL, &gemm_206_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i32 nl_200_nl_params_data[] = { 1493978240, 20, -1984 };
AI_ARRAY_OBJ_DECLARE(
    nl_200_nl_params, AI_ARRAY_FORMAT_S32,
    nl_200_nl_params_data, nl_200_nl_params_data, 3, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_200_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_199_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_200_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_200_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  nl_200_layer, 200,
  SM_TYPE, 0x0, NULL,
  sm, forward_sm_integer,
  &nl_200_chain,
  NULL, &nl_200_0_0_gemm_206_conversion_layer, AI_STATIC, 
  .nl_params = &nl_200_nl_params, 
  .axis = AI_SHAPE_CHANNEL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_199_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_198_0_0_eltwise_199_conversion_output, &tiny_bert_generator_encoder_layer_3_multi_head_attention_11_truedivtiny_bert_generator_encoder_layer_3_multi_head_attention_11_Sqrt_4D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_199_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_199_layer, 199,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_199_chain,
  NULL, &nl_200_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_198_0_0_eltwise_199_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_198_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_198_0_0_eltwise_199_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_198_0_0_eltwise_199_conversion_layer, 198,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &gemm_198_0_0_eltwise_199_conversion_chain,
  NULL, &eltwise_199_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_198_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &transpose_192_0_0_gemm_198_conversion_output, &transpose_bgemm_198_out_0_1_gemm_198_conversion_output, &gemm_198_bias_0_2_gemm_198_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_198_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_198_layer, 198,
  MATMUL_TYPE, 0x0, NULL,
  matmul, forward_matmul,
  &gemm_198_chain,
  NULL, &gemm_198_0_0_eltwise_199_conversion_layer, AI_STATIC, 
  .alpha = 1.0, 
  .beta = 1.0, 
  .tA = 0, 
  .tB = 0, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  transpose_192_0_0_gemm_198_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_192_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_192_0_0_gemm_198_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  transpose_192_0_0_gemm_198_conversion_layer, 192,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &transpose_192_0_0_gemm_198_conversion_chain,
  NULL, &gemm_198_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  transpose_192_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_190_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_192_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  transpose_192_layer, 192,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &transpose_192_chain,
  NULL, &transpose_192_0_0_gemm_198_conversion_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_CHANNEL, AI_SHAPE_HEIGHT, AI_SHAPE_WIDTH, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_190_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_188_output, &encoder_layer_2_multi_head_attention_10_dense_61_bias_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_190_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_190_layer, 190,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_190_chain,
  NULL, &transpose_192_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_188_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_176_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_188_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &gemm_188_weights, &gemm_27_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_188_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_188_layer, 188,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &gemm_188_chain,
  NULL, &eltwise_190_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  transpose_bgemm_198_out_0_1_gemm_198_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_bgemm_198_out_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_bgemm_198_out_0_1_gemm_198_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  transpose_bgemm_198_out_0_1_gemm_198_conversion_layer, 198,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &transpose_bgemm_198_out_0_1_gemm_198_conversion_chain,
  NULL, &gemm_188_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  transpose_bgemm_198_out_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_195_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_bgemm_198_out_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  transpose_bgemm_198_out_layer, 198,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &transpose_bgemm_198_out_chain,
  NULL, &transpose_bgemm_198_out_0_1_gemm_198_conversion_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_HEIGHT, AI_SHAPE_CHANNEL, AI_SHAPE_WIDTH, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_195_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_193_output, &encoder_layer_2_multi_head_attention_10_dense_62_bias_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_195_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_195_layer, 195,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_195_chain,
  NULL, &transpose_bgemm_198_out_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_193_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_176_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_193_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &gemm_193_weights, &gemm_27_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_193_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_193_layer, 193,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &gemm_193_chain,
  NULL, &eltwise_195_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  transpose_205_0_1_gemm_206_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_205_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_205_0_1_gemm_206_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  transpose_205_0_1_gemm_206_conversion_layer, 205,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &transpose_205_0_1_gemm_206_conversion_chain,
  NULL, &gemm_193_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  transpose_205_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_203_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_205_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  transpose_205_layer, 205,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &transpose_205_chain,
  NULL, &transpose_205_0_1_gemm_206_conversion_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_CHANNEL, AI_SHAPE_HEIGHT, AI_SHAPE_WIDTH, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_203_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_201_output, &encoder_layer_2_multi_head_attention_10_dense_63_bias_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_203_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_203_layer, 203,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_203_chain,
  NULL, &transpose_205_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_201_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_176_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_201_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &gemm_201_weights, &gemm_27_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_201_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_201_layer, 201,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &gemm_201_chain,
  NULL, &eltwise_203_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_176_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_173_output, &eltwise_175_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_176_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_176_layer, 176,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_176_chain,
  NULL, &gemm_201_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_173_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_166_output, &eltwise_172_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_173_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_173_layer, 173,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_173_chain,
  NULL, &eltwise_176_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_175_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &encoder_layer_1_layer_normalization_19_beta_3D, &eltwise_174_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_175_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_175_layer, 175,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_175_chain,
  NULL, &eltwise_173_layer, AI_STATIC, 
  .operation = ai_sub_f32, 
  .buffer_operation = ai_sub_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_174_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &reduce_167_Mul_0_1_eltwise_168_conversion_output, &eltwise_172_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_174_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_174_layer, 174,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_174_chain,
  NULL, &eltwise_175_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_172_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &nl_171_output, &encoder_layer_1_layer_normalization_19_gamma_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_172_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_172_layer, 172,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_172_chain,
  NULL, &eltwise_174_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)


AI_STATIC_CONST ai_i8 nl_171_nl_params_data[] = { -128, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 126, 126, 125, 125, 124, 124, 123, 123, 122, 122, 121, 121, 120, 120, 119, 119, 118 };
AI_ARRAY_OBJ_DECLARE(
    nl_171_nl_params, AI_ARRAY_FORMAT_S8,
    nl_171_nl_params_data, nl_171_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_171_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_170_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_171_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_171_layer, 171,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_171_chain,
  NULL, &eltwise_172_layer, AI_STATIC, 
  .nl_params = &nl_171_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_170_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &reduce_169_Mul_0_0_eltwise_170_conversion_output, &tiny_bert_generator_encoder_layer_layer_normalization_16_batchnorm_add_y_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_170_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_170_layer, 170,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_170_chain,
  NULL, &nl_171_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_169_Mul_0_0_eltwise_170_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_169_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_169_Mul_0_0_eltwise_170_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_169_Mul_0_0_eltwise_170_conversion_layer, 169,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &reduce_169_Mul_0_0_eltwise_170_conversion_chain,
  NULL, &eltwise_170_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_169_Mul_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_169_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_169_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &reduce_47_Mul_scale, &reduce_47_Mul_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_169_Mul_layer, 169,
  BN_TYPE, 0x0, NULL,
  bn, forward_bn,
  &reduce_169_Mul_chain,
  NULL, &reduce_169_Mul_0_0_eltwise_170_conversion_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_float reduce_169_neutral_value_data[] = { 0.0f };
AI_ARRAY_OBJ_DECLARE(
    reduce_169_neutral_value, AI_ARRAY_FORMAT_FLOAT,
    reduce_169_neutral_value_data, reduce_169_neutral_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_169_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_168_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_169_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_169_layer, 169,
  REDUCE_TYPE, 0x0, NULL,
  reduce, forward_reduce,
  &reduce_169_chain,
  NULL, &reduce_169_Mul_layer, AI_STATIC, 
  .operation = ai_sum, 
  .neutral_value = &reduce_169_neutral_value, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_168_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_166_0_0_reduce_167_conversion_output, &reduce_167_Mul_0_1_eltwise_168_conversion_0_1_eltwise_168_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_168_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_168_layer, 168,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_168_chain,
  NULL, &reduce_169_layer, AI_STATIC, 
  .operation = ai_squared_diff, 
  .buffer_operation = ai_squared_diff_buffer, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_167_Mul_0_1_eltwise_168_conversion_0_1_eltwise_168_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_167_Mul_0_1_eltwise_168_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_167_Mul_0_1_eltwise_168_conversion_0_1_eltwise_168_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_167_Mul_0_1_eltwise_168_conversion_0_1_eltwise_168_conversion_layer, 167,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &reduce_167_Mul_0_1_eltwise_168_conversion_0_1_eltwise_168_conversion_chain,
  NULL, &eltwise_168_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_167_Mul_0_1_eltwise_168_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_167_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_167_Mul_0_1_eltwise_168_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_167_Mul_0_1_eltwise_168_conversion_layer, 167,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &reduce_167_Mul_0_1_eltwise_168_conversion_chain,
  NULL, &reduce_167_Mul_0_1_eltwise_168_conversion_0_1_eltwise_168_conversion_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_167_Mul_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_167_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_167_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &reduce_47_Mul_scale, &reduce_47_Mul_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_167_Mul_layer, 167,
  BN_TYPE, 0x0, NULL,
  bn, forward_bn,
  &reduce_167_Mul_chain,
  NULL, &reduce_167_Mul_0_1_eltwise_168_conversion_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_float reduce_167_neutral_value_data[] = { 0.0f };
AI_ARRAY_OBJ_DECLARE(
    reduce_167_neutral_value, AI_ARRAY_FORMAT_FLOAT,
    reduce_167_neutral_value_data, reduce_167_neutral_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_167_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_166_0_0_reduce_167_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_167_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_167_layer, 167,
  REDUCE_TYPE, 0x0, NULL,
  reduce, forward_reduce,
  &reduce_167_chain,
  NULL, &reduce_167_Mul_layer, AI_STATIC, 
  .operation = ai_sum, 
  .neutral_value = &reduce_167_neutral_value, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_166_0_0_reduce_167_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_166_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_166_0_0_reduce_167_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_166_0_0_reduce_167_conversion_layer, 166,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &eltwise_166_0_0_reduce_167_conversion_chain,
  NULL, &reduce_167_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_166_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_143_output, &eltwise_165_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_166_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_166_layer, 166,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_166_chain,
  NULL, &eltwise_166_0_0_reduce_167_conversion_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_165_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_163_output, &dense_60_bias_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_165_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_165_layer, 165,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_165_chain,
  NULL, &eltwise_166_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_163_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_154_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_163_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &gemm_163_weights, &gemm_27_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_163_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_163_layer, 163,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &gemm_163_chain,
  NULL, &eltwise_165_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_154_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_152_output, &dense_59_bias_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_154_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_154_layer, 154,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_154_chain,
  NULL, &gemm_163_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_152_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_143_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_152_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &gemm_152_weights, &gemm_65_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_152_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_152_layer, 152,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &gemm_152_chain,
  NULL, &eltwise_154_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_143_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_140_output, &eltwise_142_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_143_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_143_layer, 143,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_143_chain,
  NULL, &gemm_152_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_140_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_133_output, &eltwise_139_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_140_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_140_layer, 140,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_140_chain,
  NULL, &eltwise_143_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_142_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &encoder_layer_1_layer_normalization_18_beta_3D, &eltwise_141_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_142_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_142_layer, 142,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_142_chain,
  NULL, &eltwise_140_layer, AI_STATIC, 
  .operation = ai_sub_f32, 
  .buffer_operation = ai_sub_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_141_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &reduce_134_Mul_0_1_eltwise_135_conversion_output, &eltwise_139_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_141_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_141_layer, 141,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_141_chain,
  NULL, &eltwise_142_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_139_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &nl_138_output, &encoder_layer_1_layer_normalization_18_gamma_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_139_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_139_layer, 139,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_139_chain,
  NULL, &eltwise_141_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)


AI_STATIC_CONST ai_i8 nl_138_nl_params_data[] = { -128, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 126, 126, 125, 125, 124, 123, 123, 122, 122, 121, 121, 120, 120, 119, 119, 118, 118, 117 };
AI_ARRAY_OBJ_DECLARE(
    nl_138_nl_params, AI_ARRAY_FORMAT_S8,
    nl_138_nl_params_data, nl_138_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_138_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_137_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_138_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_138_layer, 138,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_138_chain,
  NULL, &eltwise_139_layer, AI_STATIC, 
  .nl_params = &nl_138_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_137_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &reduce_136_Mul_0_0_eltwise_137_conversion_output, &tiny_bert_generator_encoder_layer_layer_normalization_16_batchnorm_add_y_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_137_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_137_layer, 137,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_137_chain,
  NULL, &nl_138_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_136_Mul_0_0_eltwise_137_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_136_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_136_Mul_0_0_eltwise_137_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_136_Mul_0_0_eltwise_137_conversion_layer, 136,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &reduce_136_Mul_0_0_eltwise_137_conversion_chain,
  NULL, &eltwise_137_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_136_Mul_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_136_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_136_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &reduce_47_Mul_scale, &reduce_47_Mul_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_136_Mul_layer, 136,
  BN_TYPE, 0x0, NULL,
  bn, forward_bn,
  &reduce_136_Mul_chain,
  NULL, &reduce_136_Mul_0_0_eltwise_137_conversion_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_float reduce_136_neutral_value_data[] = { 0.0f };
AI_ARRAY_OBJ_DECLARE(
    reduce_136_neutral_value, AI_ARRAY_FORMAT_FLOAT,
    reduce_136_neutral_value_data, reduce_136_neutral_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_136_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_135_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_136_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_136_layer, 136,
  REDUCE_TYPE, 0x0, NULL,
  reduce, forward_reduce,
  &reduce_136_chain,
  NULL, &reduce_136_Mul_layer, AI_STATIC, 
  .operation = ai_sum, 
  .neutral_value = &reduce_136_neutral_value, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_135_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_133_0_0_reduce_134_conversion_output, &reduce_134_Mul_0_1_eltwise_135_conversion_0_1_eltwise_135_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_135_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_135_layer, 135,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_135_chain,
  NULL, &reduce_136_layer, AI_STATIC, 
  .operation = ai_squared_diff, 
  .buffer_operation = ai_squared_diff_buffer, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_134_Mul_0_1_eltwise_135_conversion_0_1_eltwise_135_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_134_Mul_0_1_eltwise_135_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_134_Mul_0_1_eltwise_135_conversion_0_1_eltwise_135_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_134_Mul_0_1_eltwise_135_conversion_0_1_eltwise_135_conversion_layer, 134,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &reduce_134_Mul_0_1_eltwise_135_conversion_0_1_eltwise_135_conversion_chain,
  NULL, &eltwise_135_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_134_Mul_0_1_eltwise_135_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_134_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_134_Mul_0_1_eltwise_135_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_134_Mul_0_1_eltwise_135_conversion_layer, 134,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &reduce_134_Mul_0_1_eltwise_135_conversion_chain,
  NULL, &reduce_134_Mul_0_1_eltwise_135_conversion_0_1_eltwise_135_conversion_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_134_Mul_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_134_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_134_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &reduce_47_Mul_scale, &reduce_47_Mul_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_134_Mul_layer, 134,
  BN_TYPE, 0x0, NULL,
  bn, forward_bn,
  &reduce_134_Mul_chain,
  NULL, &reduce_134_Mul_0_1_eltwise_135_conversion_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_float reduce_134_neutral_value_data[] = { 0.0f };
AI_ARRAY_OBJ_DECLARE(
    reduce_134_neutral_value, AI_ARRAY_FORMAT_FLOAT,
    reduce_134_neutral_value_data, reduce_134_neutral_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_134_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_133_0_0_reduce_134_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_134_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_134_layer, 134,
  REDUCE_TYPE, 0x0, NULL,
  reduce, forward_reduce,
  &reduce_134_chain,
  NULL, &reduce_134_Mul_layer, AI_STATIC, 
  .operation = ai_sum, 
  .neutral_value = &reduce_134_neutral_value, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_133_0_0_reduce_134_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_133_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_133_0_0_reduce_134_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_133_0_0_reduce_134_conversion_layer, 133,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &eltwise_133_0_0_reduce_134_conversion_chain,
  NULL, &reduce_134_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_133_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_89_output, &eltwise_132_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_133_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_133_layer, 133,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_133_chain,
  NULL, &eltwise_133_0_0_reduce_134_conversion_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_132_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_130_output, &encoder_layer_1_multi_head_attention_9_dense_58_bias_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_132_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_132_layer, 132,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_132_chain,
  NULL, &eltwise_133_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_130_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_120_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_130_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &gemm_130_weights, &gemm_27_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_130_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_130_layer, 130,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &gemm_130_chain,
  NULL, &eltwise_132_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  transpose_120_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_119_0_0_transpose_120_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_120_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  transpose_120_layer, 120,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &transpose_120_chain,
  NULL, &gemm_130_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_CHANNEL, AI_SHAPE_HEIGHT, AI_SHAPE_WIDTH, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_119_0_0_transpose_120_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_119_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_119_0_0_transpose_120_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_119_0_0_transpose_120_conversion_layer, 119,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &gemm_119_0_0_transpose_120_conversion_chain,
  NULL, &transpose_120_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_119_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &nl_113_0_0_gemm_119_conversion_output, &transpose_118_0_1_gemm_119_conversion_output, &gemm_119_bias_0_2_gemm_119_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_119_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_119_layer, 119,
  MATMUL_TYPE, 0x0, NULL,
  matmul, forward_matmul,
  &gemm_119_chain,
  NULL, &gemm_119_0_0_transpose_120_conversion_layer, AI_STATIC, 
  .alpha = 1.0, 
  .beta = 1.0, 
  .tA = 0, 
  .tB = 0, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_113_0_0_gemm_119_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_113_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_113_0_0_gemm_119_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_113_0_0_gemm_119_conversion_layer, 113,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &nl_113_0_0_gemm_119_conversion_chain,
  NULL, &gemm_119_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i32 nl_113_nl_params_data[] = { 1706367616, 21, -992 };
AI_ARRAY_OBJ_DECLARE(
    nl_113_nl_params, AI_ARRAY_FORMAT_S32,
    nl_113_nl_params_data, nl_113_nl_params_data, 3, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_113_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_112_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_113_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_113_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  nl_113_layer, 113,
  SM_TYPE, 0x0, NULL,
  sm, forward_sm_integer,
  &nl_113_chain,
  NULL, &nl_113_0_0_gemm_119_conversion_layer, AI_STATIC, 
  .nl_params = &nl_113_nl_params, 
  .axis = AI_SHAPE_CHANNEL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_112_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_111_0_0_eltwise_112_conversion_output, &tiny_bert_generator_encoder_layer_3_multi_head_attention_11_truedivtiny_bert_generator_encoder_layer_3_multi_head_attention_11_Sqrt_4D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_112_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_112_layer, 112,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_112_chain,
  NULL, &nl_113_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_111_0_0_eltwise_112_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_111_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_111_0_0_eltwise_112_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_111_0_0_eltwise_112_conversion_layer, 111,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &gemm_111_0_0_eltwise_112_conversion_chain,
  NULL, &eltwise_112_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_111_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &transpose_105_0_0_gemm_111_conversion_output, &transpose_bgemm_111_out_0_1_gemm_111_conversion_output, &gemm_111_bias_0_2_gemm_111_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_111_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_111_layer, 111,
  MATMUL_TYPE, 0x0, NULL,
  matmul, forward_matmul,
  &gemm_111_chain,
  NULL, &gemm_111_0_0_eltwise_112_conversion_layer, AI_STATIC, 
  .alpha = 1.0, 
  .beta = 1.0, 
  .tA = 0, 
  .tB = 0, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  transpose_105_0_0_gemm_111_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_105_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_105_0_0_gemm_111_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  transpose_105_0_0_gemm_111_conversion_layer, 105,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &transpose_105_0_0_gemm_111_conversion_chain,
  NULL, &gemm_111_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  transpose_105_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_103_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_105_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  transpose_105_layer, 105,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &transpose_105_chain,
  NULL, &transpose_105_0_0_gemm_111_conversion_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_CHANNEL, AI_SHAPE_HEIGHT, AI_SHAPE_WIDTH, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_103_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_101_output, &encoder_layer_1_multi_head_attention_9_dense_55_bias_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_103_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_103_layer, 103,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_103_chain,
  NULL, &transpose_105_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_101_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_89_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_101_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &gemm_101_weights, &gemm_27_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_101_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_101_layer, 101,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &gemm_101_chain,
  NULL, &eltwise_103_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  transpose_bgemm_111_out_0_1_gemm_111_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_bgemm_111_out_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_bgemm_111_out_0_1_gemm_111_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  transpose_bgemm_111_out_0_1_gemm_111_conversion_layer, 111,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &transpose_bgemm_111_out_0_1_gemm_111_conversion_chain,
  NULL, &gemm_101_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  transpose_bgemm_111_out_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_108_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_bgemm_111_out_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  transpose_bgemm_111_out_layer, 111,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &transpose_bgemm_111_out_chain,
  NULL, &transpose_bgemm_111_out_0_1_gemm_111_conversion_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_HEIGHT, AI_SHAPE_CHANNEL, AI_SHAPE_WIDTH, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_108_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_106_output, &encoder_layer_1_multi_head_attention_9_dense_56_bias_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_108_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_108_layer, 108,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_108_chain,
  NULL, &transpose_bgemm_111_out_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_106_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_89_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_106_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &gemm_106_weights, &gemm_27_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_106_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_106_layer, 106,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &gemm_106_chain,
  NULL, &eltwise_108_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  transpose_118_0_1_gemm_119_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_118_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_118_0_1_gemm_119_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  transpose_118_0_1_gemm_119_conversion_layer, 118,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &transpose_118_0_1_gemm_119_conversion_chain,
  NULL, &gemm_106_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  transpose_118_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_116_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_118_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  transpose_118_layer, 118,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &transpose_118_chain,
  NULL, &transpose_118_0_1_gemm_119_conversion_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_CHANNEL, AI_SHAPE_HEIGHT, AI_SHAPE_WIDTH, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_116_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_114_output, &encoder_layer_1_multi_head_attention_9_dense_57_bias_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_116_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_116_layer, 116,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_116_chain,
  NULL, &transpose_118_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_114_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_89_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_114_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &gemm_114_weights, &gemm_27_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_114_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_114_layer, 114,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &gemm_114_chain,
  NULL, &eltwise_116_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_89_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_86_output, &eltwise_88_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_89_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_89_layer, 89,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_89_chain,
  NULL, &gemm_114_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_86_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_79_output, &eltwise_85_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_86_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_86_layer, 86,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_86_chain,
  NULL, &eltwise_89_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_88_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &encoder_layer_layer_normalization_17_beta_3D, &eltwise_87_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_88_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_88_layer, 88,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_88_chain,
  NULL, &eltwise_86_layer, AI_STATIC, 
  .operation = ai_sub_f32, 
  .buffer_operation = ai_sub_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_87_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &reduce_80_Mul_0_1_eltwise_81_conversion_output, &eltwise_85_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_87_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_87_layer, 87,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_87_chain,
  NULL, &eltwise_88_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_85_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &nl_84_output, &encoder_layer_layer_normalization_17_gamma_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_85_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_85_layer, 85,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_85_chain,
  NULL, &eltwise_87_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)


AI_STATIC_CONST ai_i8 nl_84_nl_params_data[] = { -128, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 126, 126, 125, 124, 124, 123, 122, 122, 121, 121, 120, 119, 119, 118, 118, 117, 117, 116, 115, 115, 114, 114, 113, 113, 112, 112, 111, 111, 110, 109, 109, 108, 108, 107, 107, 106, 106, 105, 105, 104, 104, 103, 103, 102, 102, 102, 101, 101, 100, 100, 99, 99, 98, 98, 97, 97, 97 };
AI_ARRAY_OBJ_DECLARE(
    nl_84_nl_params, AI_ARRAY_FORMAT_S8,
    nl_84_nl_params_data, nl_84_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_84_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_83_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_84_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_84_layer, 84,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_84_chain,
  NULL, &eltwise_85_layer, AI_STATIC, 
  .nl_params = &nl_84_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_83_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &reduce_82_Mul_0_0_eltwise_83_conversion_output, &tiny_bert_generator_encoder_layer_layer_normalization_16_batchnorm_add_y_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_83_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_83_layer, 83,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_83_chain,
  NULL, &nl_84_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_82_Mul_0_0_eltwise_83_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_82_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_82_Mul_0_0_eltwise_83_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_82_Mul_0_0_eltwise_83_conversion_layer, 82,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &reduce_82_Mul_0_0_eltwise_83_conversion_chain,
  NULL, &eltwise_83_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_82_Mul_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_82_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_82_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &reduce_47_Mul_scale, &reduce_47_Mul_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_82_Mul_layer, 82,
  BN_TYPE, 0x0, NULL,
  bn, forward_bn,
  &reduce_82_Mul_chain,
  NULL, &reduce_82_Mul_0_0_eltwise_83_conversion_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_float reduce_82_neutral_value_data[] = { 0.0f };
AI_ARRAY_OBJ_DECLARE(
    reduce_82_neutral_value, AI_ARRAY_FORMAT_FLOAT,
    reduce_82_neutral_value_data, reduce_82_neutral_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_82_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_81_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_82_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_82_layer, 82,
  REDUCE_TYPE, 0x0, NULL,
  reduce, forward_reduce,
  &reduce_82_chain,
  NULL, &reduce_82_Mul_layer, AI_STATIC, 
  .operation = ai_sum, 
  .neutral_value = &reduce_82_neutral_value, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_81_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_79_0_0_reduce_80_conversion_output, &reduce_80_Mul_0_1_eltwise_81_conversion_0_1_eltwise_81_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_81_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_81_layer, 81,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_81_chain,
  NULL, &reduce_82_layer, AI_STATIC, 
  .operation = ai_squared_diff, 
  .buffer_operation = ai_squared_diff_buffer, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_80_Mul_0_1_eltwise_81_conversion_0_1_eltwise_81_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_80_Mul_0_1_eltwise_81_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_80_Mul_0_1_eltwise_81_conversion_0_1_eltwise_81_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_80_Mul_0_1_eltwise_81_conversion_0_1_eltwise_81_conversion_layer, 80,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &reduce_80_Mul_0_1_eltwise_81_conversion_0_1_eltwise_81_conversion_chain,
  NULL, &eltwise_81_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_80_Mul_0_1_eltwise_81_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_80_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_80_Mul_0_1_eltwise_81_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_80_Mul_0_1_eltwise_81_conversion_layer, 80,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &reduce_80_Mul_0_1_eltwise_81_conversion_chain,
  NULL, &reduce_80_Mul_0_1_eltwise_81_conversion_0_1_eltwise_81_conversion_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_80_Mul_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_80_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_80_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &reduce_47_Mul_scale, &reduce_47_Mul_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_80_Mul_layer, 80,
  BN_TYPE, 0x0, NULL,
  bn, forward_bn,
  &reduce_80_Mul_chain,
  NULL, &reduce_80_Mul_0_1_eltwise_81_conversion_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_float reduce_80_neutral_value_data[] = { 0.0f };
AI_ARRAY_OBJ_DECLARE(
    reduce_80_neutral_value, AI_ARRAY_FORMAT_FLOAT,
    reduce_80_neutral_value_data, reduce_80_neutral_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_80_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_79_0_0_reduce_80_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_80_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_80_layer, 80,
  REDUCE_TYPE, 0x0, NULL,
  reduce, forward_reduce,
  &reduce_80_chain,
  NULL, &reduce_80_Mul_layer, AI_STATIC, 
  .operation = ai_sum, 
  .neutral_value = &reduce_80_neutral_value, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_79_0_0_reduce_80_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_79_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_79_0_0_reduce_80_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_79_0_0_reduce_80_conversion_layer, 79,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &eltwise_79_0_0_reduce_80_conversion_chain,
  NULL, &reduce_80_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_79_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_56_output, &eltwise_78_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_79_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_79_layer, 79,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_79_chain,
  NULL, &eltwise_79_0_0_reduce_80_conversion_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_78_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_76_output, &dense_54_bias_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_78_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_78_layer, 78,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_78_chain,
  NULL, &eltwise_79_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_76_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_67_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_76_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &gemm_76_weights, &gemm_27_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_76_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_76_layer, 76,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &gemm_76_chain,
  NULL, &eltwise_78_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_67_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_65_output, &dense_53_bias_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_67_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_67_layer, 67,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_67_chain,
  NULL, &gemm_76_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_65_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_56_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_65_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &gemm_65_weights, &gemm_65_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_65_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_65_layer, 65,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &gemm_65_chain,
  NULL, &eltwise_67_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_56_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_53_output, &eltwise_55_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_56_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_56_layer, 56,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_56_chain,
  NULL, &gemm_65_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_53_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_46_output, &eltwise_52_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_53_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_53_layer, 53,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_53_chain,
  NULL, &eltwise_56_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_55_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &encoder_layer_layer_normalization_16_beta_3D, &eltwise_54_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_55_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_55_layer, 55,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_55_chain,
  NULL, &eltwise_53_layer, AI_STATIC, 
  .operation = ai_sub_f32, 
  .buffer_operation = ai_sub_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_54_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &reduce_47_Mul_0_1_eltwise_48_conversion_output, &eltwise_52_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_54_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_54_layer, 54,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_54_chain,
  NULL, &eltwise_55_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_52_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &nl_51_output, &encoder_layer_layer_normalization_16_gamma_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_52_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_52_layer, 52,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_52_chain,
  NULL, &eltwise_54_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)


AI_STATIC_CONST ai_i8 nl_51_nl_params_data[] = { -128, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 126, 125, 125, 124, 123, 122, 122, 121, 120, 120, 119, 118, 118, 117, 116, 116, 115, 114, 114, 113, 113, 112, 111, 111, 110, 109, 109, 108, 108, 107, 107, 106, 105, 105, 104, 104, 103, 103, 102, 102, 101, 100, 100, 99, 99, 98, 98, 97, 97, 96, 96, 95, 95, 94, 94, 93, 93, 92, 92, 91, 91, 91, 90, 90, 89, 89, 88, 88, 87, 87, 86, 86, 86, 85, 85, 84, 84, 83, 83, 83, 82, 82 };
AI_ARRAY_OBJ_DECLARE(
    nl_51_nl_params, AI_ARRAY_FORMAT_S8,
    nl_51_nl_params_data, nl_51_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_51_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_50_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_51_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_51_layer, 51,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_51_chain,
  NULL, &eltwise_52_layer, AI_STATIC, 
  .nl_params = &nl_51_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_50_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &reduce_49_Mul_0_0_eltwise_50_conversion_output, &tiny_bert_generator_encoder_layer_layer_normalization_16_batchnorm_add_y_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_50_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_50_layer, 50,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_50_chain,
  NULL, &nl_51_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_49_Mul_0_0_eltwise_50_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_49_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_49_Mul_0_0_eltwise_50_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_49_Mul_0_0_eltwise_50_conversion_layer, 49,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &reduce_49_Mul_0_0_eltwise_50_conversion_chain,
  NULL, &eltwise_50_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_49_Mul_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_49_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_49_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &reduce_47_Mul_scale, &reduce_47_Mul_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_49_Mul_layer, 49,
  BN_TYPE, 0x0, NULL,
  bn, forward_bn,
  &reduce_49_Mul_chain,
  NULL, &reduce_49_Mul_0_0_eltwise_50_conversion_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_float reduce_49_neutral_value_data[] = { 0.0f };
AI_ARRAY_OBJ_DECLARE(
    reduce_49_neutral_value, AI_ARRAY_FORMAT_FLOAT,
    reduce_49_neutral_value_data, reduce_49_neutral_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_49_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_48_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_49_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_49_layer, 49,
  REDUCE_TYPE, 0x0, NULL,
  reduce, forward_reduce,
  &reduce_49_chain,
  NULL, &reduce_49_Mul_layer, AI_STATIC, 
  .operation = ai_sum, 
  .neutral_value = &reduce_49_neutral_value, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_48_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_46_0_0_reduce_47_conversion_output, &reduce_47_Mul_0_1_eltwise_48_conversion_0_1_eltwise_48_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_48_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_48_layer, 48,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_48_chain,
  NULL, &reduce_49_layer, AI_STATIC, 
  .operation = ai_squared_diff, 
  .buffer_operation = ai_squared_diff_buffer, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_47_Mul_0_1_eltwise_48_conversion_0_1_eltwise_48_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_47_Mul_0_1_eltwise_48_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_47_Mul_0_1_eltwise_48_conversion_0_1_eltwise_48_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_47_Mul_0_1_eltwise_48_conversion_0_1_eltwise_48_conversion_layer, 47,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &reduce_47_Mul_0_1_eltwise_48_conversion_0_1_eltwise_48_conversion_chain,
  NULL, &eltwise_48_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_47_Mul_0_1_eltwise_48_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_47_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_47_Mul_0_1_eltwise_48_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_47_Mul_0_1_eltwise_48_conversion_layer, 47,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &reduce_47_Mul_0_1_eltwise_48_conversion_chain,
  NULL, &reduce_47_Mul_0_1_eltwise_48_conversion_0_1_eltwise_48_conversion_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_47_Mul_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_47_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_47_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &reduce_47_Mul_scale, &reduce_47_Mul_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_47_Mul_layer, 47,
  BN_TYPE, 0x0, NULL,
  bn, forward_bn,
  &reduce_47_Mul_chain,
  NULL, &reduce_47_Mul_0_1_eltwise_48_conversion_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_float reduce_47_neutral_value_data[] = { 0.0f };
AI_ARRAY_OBJ_DECLARE(
    reduce_47_neutral_value, AI_ARRAY_FORMAT_FLOAT,
    reduce_47_neutral_value_data, reduce_47_neutral_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_47_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_46_0_0_reduce_47_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_47_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_47_layer, 47,
  REDUCE_TYPE, 0x0, NULL,
  reduce, forward_reduce,
  &reduce_47_chain,
  NULL, &reduce_47_Mul_layer, AI_STATIC, 
  .operation = ai_sum, 
  .neutral_value = &reduce_47_neutral_value, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_46_0_0_reduce_47_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_46_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_46_0_0_reduce_47_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_46_0_0_reduce_47_conversion_layer, 46,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &eltwise_46_0_0_reduce_47_conversion_chain,
  NULL, &reduce_47_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_46_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_2_output, &eltwise_45_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_46_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_46_layer, 46,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_46_chain,
  NULL, &eltwise_46_0_0_reduce_47_conversion_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_45_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_43_output, &encoder_layer_multi_head_attention_8_dense_52_bias_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_45_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_45_layer, 45,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_45_chain,
  NULL, &eltwise_46_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_43_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_33_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_43_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &gemm_43_weights, &gemm_27_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_43_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_43_layer, 43,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &gemm_43_chain,
  NULL, &eltwise_45_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  transpose_33_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_32_0_0_transpose_33_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_33_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  transpose_33_layer, 33,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &transpose_33_chain,
  NULL, &gemm_43_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_CHANNEL, AI_SHAPE_HEIGHT, AI_SHAPE_WIDTH, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_32_0_0_transpose_33_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_32_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_32_0_0_transpose_33_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_32_0_0_transpose_33_conversion_layer, 32,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &gemm_32_0_0_transpose_33_conversion_chain,
  NULL, &transpose_33_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_32_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &nl_26_0_0_gemm_32_conversion_output, &transpose_31_0_1_gemm_32_conversion_output, &gemm_32_bias_0_2_gemm_32_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_32_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_32_layer, 32,
  MATMUL_TYPE, 0x0, NULL,
  matmul, forward_matmul,
  &gemm_32_chain,
  NULL, &gemm_32_0_0_transpose_33_conversion_layer, AI_STATIC, 
  .alpha = 1.0, 
  .beta = 1.0, 
  .tA = 0, 
  .tB = 0, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_26_0_0_gemm_32_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_26_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_26_0_0_gemm_32_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_26_0_0_gemm_32_conversion_layer, 26,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &nl_26_0_0_gemm_32_conversion_chain,
  NULL, &gemm_32_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i32 nl_26_nl_params_data[] = { 1226403968, 22, -496 };
AI_ARRAY_OBJ_DECLARE(
    nl_26_nl_params, AI_ARRAY_FORMAT_S32,
    nl_26_nl_params_data, nl_26_nl_params_data, 3, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_26_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_25_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_26_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_26_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  nl_26_layer, 26,
  SM_TYPE, 0x0, NULL,
  sm, forward_sm_integer,
  &nl_26_chain,
  NULL, &nl_26_0_0_gemm_32_conversion_layer, AI_STATIC, 
  .nl_params = &nl_26_nl_params, 
  .axis = AI_SHAPE_CHANNEL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_25_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_24_0_0_eltwise_25_conversion_output, &tiny_bert_generator_encoder_layer_3_multi_head_attention_11_truedivtiny_bert_generator_encoder_layer_3_multi_head_attention_11_Sqrt_4D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_25_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_25_layer, 25,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_25_chain,
  NULL, &nl_26_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_24_0_0_eltwise_25_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_24_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_24_0_0_eltwise_25_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_24_0_0_eltwise_25_conversion_layer, 24,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &gemm_24_0_0_eltwise_25_conversion_chain,
  NULL, &eltwise_25_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_24_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &transpose_18_0_0_gemm_24_conversion_output, &transpose_bgemm_24_out_0_1_gemm_24_conversion_output, &gemm_24_bias_0_2_gemm_24_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_24_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_24_layer, 24,
  MATMUL_TYPE, 0x0, NULL,
  matmul, forward_matmul,
  &gemm_24_chain,
  NULL, &gemm_24_0_0_eltwise_25_conversion_layer, AI_STATIC, 
  .alpha = 1.0, 
  .beta = 1.0, 
  .tA = 0, 
  .tB = 0, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  transpose_18_0_0_gemm_24_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_18_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_18_0_0_gemm_24_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  transpose_18_0_0_gemm_24_conversion_layer, 18,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &transpose_18_0_0_gemm_24_conversion_chain,
  NULL, &gemm_24_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  transpose_18_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_16_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_18_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  transpose_18_layer, 18,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &transpose_18_chain,
  NULL, &transpose_18_0_0_gemm_24_conversion_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_CHANNEL, AI_SHAPE_HEIGHT, AI_SHAPE_WIDTH, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_16_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_14_output, &encoder_layer_multi_head_attention_8_dense_49_bias_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_16_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_16_layer, 16,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_16_chain,
  NULL, &transpose_18_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_14_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_2_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_14_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &gemm_14_weights, &gemm_27_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_14_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_14_layer, 14,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &gemm_14_chain,
  NULL, &eltwise_16_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  transpose_bgemm_24_out_0_1_gemm_24_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_bgemm_24_out_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_bgemm_24_out_0_1_gemm_24_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  transpose_bgemm_24_out_0_1_gemm_24_conversion_layer, 24,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &transpose_bgemm_24_out_0_1_gemm_24_conversion_chain,
  NULL, &gemm_14_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  transpose_bgemm_24_out_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_21_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_bgemm_24_out_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  transpose_bgemm_24_out_layer, 24,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &transpose_bgemm_24_out_chain,
  NULL, &transpose_bgemm_24_out_0_1_gemm_24_conversion_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_HEIGHT, AI_SHAPE_CHANNEL, AI_SHAPE_WIDTH, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_21_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_19_output, &encoder_layer_multi_head_attention_8_dense_50_bias_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_21_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_21_layer, 21,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_21_chain,
  NULL, &transpose_bgemm_24_out_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_19_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_2_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_19_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &gemm_19_weights, &gemm_27_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_19_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_19_layer, 19,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &gemm_19_chain,
  NULL, &eltwise_21_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  transpose_31_0_1_gemm_32_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_31_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_31_0_1_gemm_32_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  transpose_31_0_1_gemm_32_conversion_layer, 31,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &transpose_31_0_1_gemm_32_conversion_chain,
  NULL, &gemm_19_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  transpose_31_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_29_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_31_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  transpose_31_layer, 31,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &transpose_31_chain,
  NULL, &transpose_31_0_1_gemm_32_conversion_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_CHANNEL, AI_SHAPE_HEIGHT, AI_SHAPE_WIDTH, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_29_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_27_output, &encoder_layer_multi_head_attention_8_dense_51_bias_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_29_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_29_layer, 29,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_29_chain,
  NULL, &transpose_31_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_27_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_2_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_27_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &gemm_27_weights, &gemm_27_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_27_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_27_layer, 27,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &gemm_27_chain,
  NULL, &eltwise_29_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_2_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_1_output, &tiny_bert_generator_tf___operators___add_y),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_2_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_2_layer, 2,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_2_chain,
  NULL, &gemm_27_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_1_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gather_0_output0, &tiny_bert_generator_tf_math_multiply_Mul_y_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_1_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_1_layer, 1,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_1_chain,
  NULL, &eltwise_2_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gather_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &embedding_embeddings, &serving_default_input_10_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gather_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gather_0_layer, 0,
  GATHER_TYPE, 0x0, NULL,
  gather, forward_gather,
  &gather_0_chain,
  NULL, &eltwise_1_layer, AI_STATIC, 
  .axis = AI_SHAPE_HEIGHT, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_24_bias_0_2_gemm_24_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_24_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_24_bias_0_2_gemm_24_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_24_bias_0_2_gemm_24_conversion_layer, 24,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &gemm_24_bias_0_2_gemm_24_conversion_chain,
  NULL, &gather_0_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_32_bias_0_2_gemm_32_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_32_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_32_bias_0_2_gemm_32_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_32_bias_0_2_gemm_32_conversion_layer, 32,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &gemm_32_bias_0_2_gemm_32_conversion_chain,
  NULL, &gemm_24_bias_0_2_gemm_24_conversion_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_111_bias_0_2_gemm_111_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_111_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_111_bias_0_2_gemm_111_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_111_bias_0_2_gemm_111_conversion_layer, 111,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &gemm_111_bias_0_2_gemm_111_conversion_chain,
  NULL, &gemm_32_bias_0_2_gemm_32_conversion_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_119_bias_0_2_gemm_119_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_119_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_119_bias_0_2_gemm_119_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_119_bias_0_2_gemm_119_conversion_layer, 119,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &gemm_119_bias_0_2_gemm_119_conversion_chain,
  NULL, &gemm_111_bias_0_2_gemm_111_conversion_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_198_bias_0_2_gemm_198_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_198_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_198_bias_0_2_gemm_198_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_198_bias_0_2_gemm_198_conversion_layer, 198,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &gemm_198_bias_0_2_gemm_198_conversion_chain,
  NULL, &gemm_119_bias_0_2_gemm_119_conversion_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_206_bias_0_2_gemm_206_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_206_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_206_bias_0_2_gemm_206_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_206_bias_0_2_gemm_206_conversion_layer, 206,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &gemm_206_bias_0_2_gemm_206_conversion_chain,
  NULL, &gemm_198_bias_0_2_gemm_198_conversion_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_285_bias_0_2_gemm_285_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_285_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_285_bias_0_2_gemm_285_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_285_bias_0_2_gemm_285_conversion_layer, 285,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &gemm_285_bias_0_2_gemm_285_conversion_chain,
  NULL, &gemm_206_bias_0_2_gemm_206_conversion_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_293_bias_0_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_293_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_293_bias_0_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_293_bias_0_conversion_layer, 293,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &gemm_293_bias_0_conversion_chain,
  NULL, &gemm_285_bias_0_2_gemm_285_conversion_layer, AI_STATIC, 
)


#if (AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 2171264, 1, 1),
    2171264, NULL, NULL),
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 550828, 1, 1),
    550828, NULL, NULL),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_IN_NUM, &serving_default_input_10_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_OUT_NUM, &eltwise_361_output),
  &gemm_293_bias_0_conversion_layer, 0xa6b1f45d, NULL)

#else

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 2171264, 1, 1),
      2171264, NULL, NULL)
  ),
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 550828, 1, 1),
      550828, NULL, NULL)
  ),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_IN_NUM, &serving_default_input_10_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_OUT_NUM, &eltwise_361_output),
  &gemm_293_bias_0_conversion_layer, 0xa6b1f45d, NULL)

#endif	/*(AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)*/



/******************************************************************************/
AI_DECLARE_STATIC
ai_bool network_configure_activations(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_activations_map(g_network_activations_map, 1, params)) {
    /* Updating activations (byte) offsets */
    
    serving_default_input_10_output_array.data = AI_PTR(g_network_activations_map[0] + 390416);
    serving_default_input_10_output_array.data_start = AI_PTR(g_network_activations_map[0] + 390416);
    gemm_293_bias_0_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 390412);
    gemm_293_bias_0_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 390412);
    gemm_285_bias_0_2_gemm_285_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 390816);
    gemm_285_bias_0_2_gemm_285_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 390816);
    gemm_206_bias_0_2_gemm_206_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 390408);
    gemm_206_bias_0_2_gemm_206_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 390408);
    gemm_198_bias_0_2_gemm_198_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 390820);
    gemm_198_bias_0_2_gemm_198_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 390820);
    gemm_119_bias_0_2_gemm_119_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 390404);
    gemm_119_bias_0_2_gemm_119_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 390404);
    gemm_111_bias_0_2_gemm_111_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 390824);
    gemm_111_bias_0_2_gemm_111_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 390824);
    gemm_32_bias_0_2_gemm_32_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 390400);
    gemm_32_bias_0_2_gemm_32_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 390400);
    gemm_24_bias_0_2_gemm_24_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 390828);
    gemm_24_bias_0_2_gemm_24_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 390828);
    gather_0_output_array.data = AI_PTR(g_network_activations_map[0] + 364800);
    gather_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 364800);
    eltwise_1_output_array.data = AI_PTR(g_network_activations_map[0] + 364800);
    eltwise_1_output_array.data_start = AI_PTR(g_network_activations_map[0] + 364800);
    eltwise_2_output_array.data = AI_PTR(g_network_activations_map[0] + 364800);
    eltwise_2_output_array.data_start = AI_PTR(g_network_activations_map[0] + 364800);
    gemm_27_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 390832);
    gemm_27_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 390832);
    gemm_27_output_array.data = AI_PTR(g_network_activations_map[0] + 339200);
    gemm_27_output_array.data_start = AI_PTR(g_network_activations_map[0] + 339200);
    eltwise_29_output_array.data = AI_PTR(g_network_activations_map[0] + 390832);
    eltwise_29_output_array.data_start = AI_PTR(g_network_activations_map[0] + 390832);
    transpose_31_output_array.data = AI_PTR(g_network_activations_map[0] + 339200);
    transpose_31_output_array.data_start = AI_PTR(g_network_activations_map[0] + 339200);
    transpose_31_0_1_gemm_32_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 262400);
    transpose_31_0_1_gemm_32_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 262400);
    gemm_19_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 261376);
    gemm_19_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 261376);
    gemm_19_output_array.data = AI_PTR(g_network_activations_map[0] + 390832);
    gemm_19_output_array.data_start = AI_PTR(g_network_activations_map[0] + 390832);
    eltwise_21_output_array.data = AI_PTR(g_network_activations_map[0] + 236800);
    eltwise_21_output_array.data_start = AI_PTR(g_network_activations_map[0] + 236800);
    transpose_bgemm_24_out_output_array.data = AI_PTR(g_network_activations_map[0] + 390832);
    transpose_bgemm_24_out_output_array.data_start = AI_PTR(g_network_activations_map[0] + 390832);
    transpose_bgemm_24_out_0_1_gemm_24_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 160000);
    transpose_bgemm_24_out_0_1_gemm_24_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 160000);
    gemm_14_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 158976);
    gemm_14_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 158976);
    gemm_14_output_array.data = AI_PTR(g_network_activations_map[0] + 390832);
    gemm_14_output_array.data_start = AI_PTR(g_network_activations_map[0] + 390832);
    eltwise_16_output_array.data = AI_PTR(g_network_activations_map[0] + 134400);
    eltwise_16_output_array.data_start = AI_PTR(g_network_activations_map[0] + 134400);
    transpose_18_output_array.data = AI_PTR(g_network_activations_map[0] + 108800);
    transpose_18_output_array.data_start = AI_PTR(g_network_activations_map[0] + 108800);
    transpose_18_0_0_gemm_24_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 390832);
    transpose_18_0_0_gemm_24_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 390832);
    gemm_24_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    gemm_24_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    gemm_24_0_0_eltwise_25_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 160000);
    gemm_24_0_0_eltwise_25_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 160000);
    eltwise_25_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    eltwise_25_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    nl_26_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 40000);
    nl_26_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 40000);
    nl_26_output_array.data = AI_PTR(g_network_activations_map[0] + 41024);
    nl_26_output_array.data_start = AI_PTR(g_network_activations_map[0] + 41024);
    nl_26_0_0_gemm_32_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 81024);
    nl_26_0_0_gemm_32_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 81024);
    gemm_32_output_array.data = AI_PTR(g_network_activations_map[0] + 390828);
    gemm_32_output_array.data_start = AI_PTR(g_network_activations_map[0] + 390828);
    gemm_32_0_0_transpose_33_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    gemm_32_0_0_transpose_33_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    transpose_33_output_array.data = AI_PTR(g_network_activations_map[0] + 25600);
    transpose_33_output_array.data_start = AI_PTR(g_network_activations_map[0] + 25600);
    gemm_43_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    gemm_43_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    gemm_43_output_array.data = AI_PTR(g_network_activations_map[0] + 51200);
    gemm_43_output_array.data_start = AI_PTR(g_network_activations_map[0] + 51200);
    eltwise_45_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    eltwise_45_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    eltwise_46_output_array.data = AI_PTR(g_network_activations_map[0] + 25600);
    eltwise_46_output_array.data_start = AI_PTR(g_network_activations_map[0] + 25600);
    eltwise_46_0_0_reduce_47_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 51200);
    eltwise_46_0_0_reduce_47_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 51200);
    reduce_47_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    reduce_47_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    reduce_47_Mul_output_array.data = AI_PTR(g_network_activations_map[0] + 400);
    reduce_47_Mul_output_array.data_start = AI_PTR(g_network_activations_map[0] + 400);
    reduce_47_Mul_0_1_eltwise_48_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    reduce_47_Mul_0_1_eltwise_48_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    reduce_47_Mul_0_1_eltwise_48_conversion_0_1_eltwise_48_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 100);
    reduce_47_Mul_0_1_eltwise_48_conversion_0_1_eltwise_48_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 100);
    eltwise_48_output_array.data = AI_PTR(g_network_activations_map[0] + 153600);
    eltwise_48_output_array.data_start = AI_PTR(g_network_activations_map[0] + 153600);
    reduce_49_output_array.data = AI_PTR(g_network_activations_map[0] + 100);
    reduce_49_output_array.data_start = AI_PTR(g_network_activations_map[0] + 100);
    reduce_49_Mul_output_array.data = AI_PTR(g_network_activations_map[0] + 500);
    reduce_49_Mul_output_array.data_start = AI_PTR(g_network_activations_map[0] + 500);
    reduce_49_Mul_0_0_eltwise_50_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 100);
    reduce_49_Mul_0_0_eltwise_50_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 100);
    eltwise_50_output_array.data = AI_PTR(g_network_activations_map[0] + 200);
    eltwise_50_output_array.data_start = AI_PTR(g_network_activations_map[0] + 200);
    nl_51_output_array.data = AI_PTR(g_network_activations_map[0] + 100);
    nl_51_output_array.data_start = AI_PTR(g_network_activations_map[0] + 100);
    eltwise_52_output_array.data = AI_PTR(g_network_activations_map[0] + 51200);
    eltwise_52_output_array.data_start = AI_PTR(g_network_activations_map[0] + 51200);
    eltwise_54_output_array.data = AI_PTR(g_network_activations_map[0] + 76800);
    eltwise_54_output_array.data_start = AI_PTR(g_network_activations_map[0] + 76800);
    eltwise_55_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    eltwise_55_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    eltwise_53_output_array.data = AI_PTR(g_network_activations_map[0] + 76800);
    eltwise_53_output_array.data_start = AI_PTR(g_network_activations_map[0] + 76800);
    eltwise_56_output_array.data = AI_PTR(g_network_activations_map[0] + 25600);
    eltwise_56_output_array.data_start = AI_PTR(g_network_activations_map[0] + 25600);
    gemm_65_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    gemm_65_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    gemm_65_output_array.data = AI_PTR(g_network_activations_map[0] + 51200);
    gemm_65_output_array.data_start = AI_PTR(g_network_activations_map[0] + 51200);
    eltwise_67_output_array.data = AI_PTR(g_network_activations_map[0] + 102400);
    eltwise_67_output_array.data_start = AI_PTR(g_network_activations_map[0] + 102400);
    gemm_76_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    gemm_76_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    gemm_76_output_array.data = AI_PTR(g_network_activations_map[0] + 51200);
    gemm_76_output_array.data_start = AI_PTR(g_network_activations_map[0] + 51200);
    eltwise_78_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    eltwise_78_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    eltwise_79_output_array.data = AI_PTR(g_network_activations_map[0] + 51200);
    eltwise_79_output_array.data_start = AI_PTR(g_network_activations_map[0] + 51200);
    eltwise_79_0_0_reduce_80_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 76800);
    eltwise_79_0_0_reduce_80_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 76800);
    reduce_80_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    reduce_80_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    reduce_80_Mul_output_array.data = AI_PTR(g_network_activations_map[0] + 400);
    reduce_80_Mul_output_array.data_start = AI_PTR(g_network_activations_map[0] + 400);
    reduce_80_Mul_0_1_eltwise_81_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    reduce_80_Mul_0_1_eltwise_81_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    reduce_80_Mul_0_1_eltwise_81_conversion_0_1_eltwise_81_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 100);
    reduce_80_Mul_0_1_eltwise_81_conversion_0_1_eltwise_81_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 100);
    eltwise_81_output_array.data = AI_PTR(g_network_activations_map[0] + 179200);
    eltwise_81_output_array.data_start = AI_PTR(g_network_activations_map[0] + 179200);
    reduce_82_output_array.data = AI_PTR(g_network_activations_map[0] + 100);
    reduce_82_output_array.data_start = AI_PTR(g_network_activations_map[0] + 100);
    reduce_82_Mul_output_array.data = AI_PTR(g_network_activations_map[0] + 500);
    reduce_82_Mul_output_array.data_start = AI_PTR(g_network_activations_map[0] + 500);
    reduce_82_Mul_0_0_eltwise_83_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 100);
    reduce_82_Mul_0_0_eltwise_83_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 100);
    eltwise_83_output_array.data = AI_PTR(g_network_activations_map[0] + 200);
    eltwise_83_output_array.data_start = AI_PTR(g_network_activations_map[0] + 200);
    nl_84_output_array.data = AI_PTR(g_network_activations_map[0] + 100);
    nl_84_output_array.data_start = AI_PTR(g_network_activations_map[0] + 100);
    eltwise_85_output_array.data = AI_PTR(g_network_activations_map[0] + 200);
    eltwise_85_output_array.data_start = AI_PTR(g_network_activations_map[0] + 200);
    eltwise_87_output_array.data = AI_PTR(g_network_activations_map[0] + 76800);
    eltwise_87_output_array.data_start = AI_PTR(g_network_activations_map[0] + 76800);
    eltwise_88_output_array.data = AI_PTR(g_network_activations_map[0] + 102400);
    eltwise_88_output_array.data_start = AI_PTR(g_network_activations_map[0] + 102400);
    eltwise_86_output_array.data = AI_PTR(g_network_activations_map[0] + 76800);
    eltwise_86_output_array.data_start = AI_PTR(g_network_activations_map[0] + 76800);
    eltwise_89_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    eltwise_89_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    gemm_114_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 25600);
    gemm_114_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 25600);
    gemm_114_output_array.data = AI_PTR(g_network_activations_map[0] + 26624);
    gemm_114_output_array.data_start = AI_PTR(g_network_activations_map[0] + 26624);
    eltwise_116_output_array.data = AI_PTR(g_network_activations_map[0] + 52224);
    eltwise_116_output_array.data_start = AI_PTR(g_network_activations_map[0] + 52224);
    transpose_118_output_array.data = AI_PTR(g_network_activations_map[0] + 25600);
    transpose_118_output_array.data_start = AI_PTR(g_network_activations_map[0] + 25600);
    transpose_118_0_1_gemm_119_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 51200);
    transpose_118_0_1_gemm_119_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 51200);
    gemm_106_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 25600);
    gemm_106_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 25600);
    gemm_106_output_array.data = AI_PTR(g_network_activations_map[0] + 153600);
    gemm_106_output_array.data_start = AI_PTR(g_network_activations_map[0] + 153600);
    eltwise_108_output_array.data = AI_PTR(g_network_activations_map[0] + 25600);
    eltwise_108_output_array.data_start = AI_PTR(g_network_activations_map[0] + 25600);
    transpose_bgemm_111_out_output_array.data = AI_PTR(g_network_activations_map[0] + 153600);
    transpose_bgemm_111_out_output_array.data_start = AI_PTR(g_network_activations_map[0] + 153600);
    transpose_bgemm_111_out_0_1_gemm_111_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 179200);
    transpose_bgemm_111_out_0_1_gemm_111_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 179200);
    gemm_101_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 492208);
    gemm_101_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 492208);
    gemm_101_output_array.data = AI_PTR(g_network_activations_map[0] + 466608);
    gemm_101_output_array.data_start = AI_PTR(g_network_activations_map[0] + 466608);
    eltwise_103_output_array.data = AI_PTR(g_network_activations_map[0] + 492208);
    eltwise_103_output_array.data_start = AI_PTR(g_network_activations_map[0] + 492208);
    transpose_105_output_array.data = AI_PTR(g_network_activations_map[0] + 517808);
    transpose_105_output_array.data_start = AI_PTR(g_network_activations_map[0] + 517808);
    transpose_105_0_0_gemm_111_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 288004);
    transpose_105_0_0_gemm_111_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 288004);
    gemm_111_output_array.data = AI_PTR(g_network_activations_map[0] + 390828);
    gemm_111_output_array.data_start = AI_PTR(g_network_activations_map[0] + 390828);
    gemm_111_0_0_eltwise_112_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 153600);
    gemm_111_0_0_eltwise_112_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 153600);
    eltwise_112_output_array.data = AI_PTR(g_network_activations_map[0] + 193600);
    eltwise_112_output_array.data_start = AI_PTR(g_network_activations_map[0] + 193600);
    nl_113_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 25600);
    nl_113_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 25600);
    nl_113_output_array.data = AI_PTR(g_network_activations_map[0] + 153600);
    nl_113_output_array.data_start = AI_PTR(g_network_activations_map[0] + 153600);
    nl_113_0_0_gemm_119_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 193600);
    nl_113_0_0_gemm_119_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 193600);
    gemm_119_output_array.data = AI_PTR(g_network_activations_map[0] + 390824);
    gemm_119_output_array.data_start = AI_PTR(g_network_activations_map[0] + 390824);
    gemm_119_0_0_transpose_120_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 25600);
    gemm_119_0_0_transpose_120_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 25600);
    transpose_120_output_array.data = AI_PTR(g_network_activations_map[0] + 51200);
    transpose_120_output_array.data_start = AI_PTR(g_network_activations_map[0] + 51200);
    gemm_130_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 25600);
    gemm_130_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 25600);
    gemm_130_output_array.data = AI_PTR(g_network_activations_map[0] + 76800);
    gemm_130_output_array.data_start = AI_PTR(g_network_activations_map[0] + 76800);
    eltwise_132_output_array.data = AI_PTR(g_network_activations_map[0] + 25600);
    eltwise_132_output_array.data_start = AI_PTR(g_network_activations_map[0] + 25600);
    eltwise_133_output_array.data = AI_PTR(g_network_activations_map[0] + 51200);
    eltwise_133_output_array.data_start = AI_PTR(g_network_activations_map[0] + 51200);
    eltwise_133_0_0_reduce_134_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 76800);
    eltwise_133_0_0_reduce_134_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 76800);
    reduce_134_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    reduce_134_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    reduce_134_Mul_output_array.data = AI_PTR(g_network_activations_map[0] + 400);
    reduce_134_Mul_output_array.data_start = AI_PTR(g_network_activations_map[0] + 400);
    reduce_134_Mul_0_1_eltwise_135_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    reduce_134_Mul_0_1_eltwise_135_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    reduce_134_Mul_0_1_eltwise_135_conversion_0_1_eltwise_135_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 100);
    reduce_134_Mul_0_1_eltwise_135_conversion_0_1_eltwise_135_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 100);
    eltwise_135_output_array.data = AI_PTR(g_network_activations_map[0] + 179200);
    eltwise_135_output_array.data_start = AI_PTR(g_network_activations_map[0] + 179200);
    reduce_136_output_array.data = AI_PTR(g_network_activations_map[0] + 100);
    reduce_136_output_array.data_start = AI_PTR(g_network_activations_map[0] + 100);
    reduce_136_Mul_output_array.data = AI_PTR(g_network_activations_map[0] + 500);
    reduce_136_Mul_output_array.data_start = AI_PTR(g_network_activations_map[0] + 500);
    reduce_136_Mul_0_0_eltwise_137_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 100);
    reduce_136_Mul_0_0_eltwise_137_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 100);
    eltwise_137_output_array.data = AI_PTR(g_network_activations_map[0] + 200);
    eltwise_137_output_array.data_start = AI_PTR(g_network_activations_map[0] + 200);
    nl_138_output_array.data = AI_PTR(g_network_activations_map[0] + 100);
    nl_138_output_array.data_start = AI_PTR(g_network_activations_map[0] + 100);
    eltwise_139_output_array.data = AI_PTR(g_network_activations_map[0] + 200);
    eltwise_139_output_array.data_start = AI_PTR(g_network_activations_map[0] + 200);
    eltwise_141_output_array.data = AI_PTR(g_network_activations_map[0] + 76800);
    eltwise_141_output_array.data_start = AI_PTR(g_network_activations_map[0] + 76800);
    eltwise_142_output_array.data = AI_PTR(g_network_activations_map[0] + 102400);
    eltwise_142_output_array.data_start = AI_PTR(g_network_activations_map[0] + 102400);
    eltwise_140_output_array.data = AI_PTR(g_network_activations_map[0] + 76800);
    eltwise_140_output_array.data_start = AI_PTR(g_network_activations_map[0] + 76800);
    eltwise_143_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    eltwise_143_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    gemm_152_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 25600);
    gemm_152_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 25600);
    gemm_152_output_array.data = AI_PTR(g_network_activations_map[0] + 26624);
    gemm_152_output_array.data_start = AI_PTR(g_network_activations_map[0] + 26624);
    eltwise_154_output_array.data = AI_PTR(g_network_activations_map[0] + 77824);
    eltwise_154_output_array.data_start = AI_PTR(g_network_activations_map[0] + 77824);
    gemm_163_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 25600);
    gemm_163_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 25600);
    gemm_163_output_array.data = AI_PTR(g_network_activations_map[0] + 27648);
    gemm_163_output_array.data_start = AI_PTR(g_network_activations_map[0] + 27648);
    eltwise_165_output_array.data = AI_PTR(g_network_activations_map[0] + 53248);
    eltwise_165_output_array.data_start = AI_PTR(g_network_activations_map[0] + 53248);
    eltwise_166_output_array.data = AI_PTR(g_network_activations_map[0] + 25600);
    eltwise_166_output_array.data_start = AI_PTR(g_network_activations_map[0] + 25600);
    eltwise_166_0_0_reduce_167_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 51200);
    eltwise_166_0_0_reduce_167_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 51200);
    reduce_167_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    reduce_167_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    reduce_167_Mul_output_array.data = AI_PTR(g_network_activations_map[0] + 400);
    reduce_167_Mul_output_array.data_start = AI_PTR(g_network_activations_map[0] + 400);
    reduce_167_Mul_0_1_eltwise_168_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    reduce_167_Mul_0_1_eltwise_168_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    reduce_167_Mul_0_1_eltwise_168_conversion_0_1_eltwise_168_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 100);
    reduce_167_Mul_0_1_eltwise_168_conversion_0_1_eltwise_168_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 100);
    eltwise_168_output_array.data = AI_PTR(g_network_activations_map[0] + 153600);
    eltwise_168_output_array.data_start = AI_PTR(g_network_activations_map[0] + 153600);
    reduce_169_output_array.data = AI_PTR(g_network_activations_map[0] + 100);
    reduce_169_output_array.data_start = AI_PTR(g_network_activations_map[0] + 100);
    reduce_169_Mul_output_array.data = AI_PTR(g_network_activations_map[0] + 500);
    reduce_169_Mul_output_array.data_start = AI_PTR(g_network_activations_map[0] + 500);
    reduce_169_Mul_0_0_eltwise_170_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 100);
    reduce_169_Mul_0_0_eltwise_170_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 100);
    eltwise_170_output_array.data = AI_PTR(g_network_activations_map[0] + 200);
    eltwise_170_output_array.data_start = AI_PTR(g_network_activations_map[0] + 200);
    nl_171_output_array.data = AI_PTR(g_network_activations_map[0] + 100);
    nl_171_output_array.data_start = AI_PTR(g_network_activations_map[0] + 100);
    eltwise_172_output_array.data = AI_PTR(g_network_activations_map[0] + 51200);
    eltwise_172_output_array.data_start = AI_PTR(g_network_activations_map[0] + 51200);
    eltwise_174_output_array.data = AI_PTR(g_network_activations_map[0] + 76800);
    eltwise_174_output_array.data_start = AI_PTR(g_network_activations_map[0] + 76800);
    eltwise_175_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    eltwise_175_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    eltwise_173_output_array.data = AI_PTR(g_network_activations_map[0] + 76800);
    eltwise_173_output_array.data_start = AI_PTR(g_network_activations_map[0] + 76800);
    eltwise_176_output_array.data = AI_PTR(g_network_activations_map[0] + 25600);
    eltwise_176_output_array.data_start = AI_PTR(g_network_activations_map[0] + 25600);
    gemm_201_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    gemm_201_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    gemm_201_output_array.data = AI_PTR(g_network_activations_map[0] + 51200);
    gemm_201_output_array.data_start = AI_PTR(g_network_activations_map[0] + 51200);
    eltwise_203_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    eltwise_203_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    transpose_205_output_array.data = AI_PTR(g_network_activations_map[0] + 51200);
    transpose_205_output_array.data_start = AI_PTR(g_network_activations_map[0] + 51200);
    transpose_205_0_1_gemm_206_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 76800);
    transpose_205_0_1_gemm_206_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 76800);
    gemm_193_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    gemm_193_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    gemm_193_output_array.data = AI_PTR(g_network_activations_map[0] + 51200);
    gemm_193_output_array.data_start = AI_PTR(g_network_activations_map[0] + 51200);
    eltwise_195_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    eltwise_195_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    transpose_bgemm_198_out_output_array.data = AI_PTR(g_network_activations_map[0] + 51200);
    transpose_bgemm_198_out_output_array.data_start = AI_PTR(g_network_activations_map[0] + 51200);
    transpose_bgemm_198_out_0_1_gemm_198_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 179200);
    transpose_bgemm_198_out_0_1_gemm_198_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 179200);
    gemm_188_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    gemm_188_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    gemm_188_output_array.data = AI_PTR(g_network_activations_map[0] + 51200);
    gemm_188_output_array.data_start = AI_PTR(g_network_activations_map[0] + 51200);
    eltwise_190_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    eltwise_190_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    transpose_192_output_array.data = AI_PTR(g_network_activations_map[0] + 51200);
    transpose_192_output_array.data_start = AI_PTR(g_network_activations_map[0] + 51200);
    transpose_192_0_0_gemm_198_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 281600);
    transpose_192_0_0_gemm_198_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 281600);
    gemm_198_output_array.data = AI_PTR(g_network_activations_map[0] + 390824);
    gemm_198_output_array.data_start = AI_PTR(g_network_activations_map[0] + 390824);
    gemm_198_0_0_eltwise_199_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 179200);
    gemm_198_0_0_eltwise_199_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 179200);
    eltwise_199_output_array.data = AI_PTR(g_network_activations_map[0] + 219200);
    eltwise_199_output_array.data_start = AI_PTR(g_network_activations_map[0] + 219200);
    nl_200_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    nl_200_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    nl_200_output_array.data = AI_PTR(g_network_activations_map[0] + 179200);
    nl_200_output_array.data_start = AI_PTR(g_network_activations_map[0] + 179200);
    nl_200_0_0_gemm_206_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 219200);
    nl_200_0_0_gemm_206_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 219200);
    gemm_206_output_array.data = AI_PTR(g_network_activations_map[0] + 390820);
    gemm_206_output_array.data_start = AI_PTR(g_network_activations_map[0] + 390820);
    gemm_206_0_0_transpose_207_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    gemm_206_0_0_transpose_207_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    transpose_207_output_array.data = AI_PTR(g_network_activations_map[0] + 51200);
    transpose_207_output_array.data_start = AI_PTR(g_network_activations_map[0] + 51200);
    gemm_217_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    gemm_217_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    gemm_217_output_array.data = AI_PTR(g_network_activations_map[0] + 76800);
    gemm_217_output_array.data_start = AI_PTR(g_network_activations_map[0] + 76800);
    eltwise_219_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    eltwise_219_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    eltwise_220_output_array.data = AI_PTR(g_network_activations_map[0] + 51200);
    eltwise_220_output_array.data_start = AI_PTR(g_network_activations_map[0] + 51200);
    eltwise_220_0_0_reduce_221_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 76800);
    eltwise_220_0_0_reduce_221_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 76800);
    reduce_221_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    reduce_221_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    reduce_221_Mul_output_array.data = AI_PTR(g_network_activations_map[0] + 400);
    reduce_221_Mul_output_array.data_start = AI_PTR(g_network_activations_map[0] + 400);
    reduce_221_Mul_0_1_eltwise_222_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    reduce_221_Mul_0_1_eltwise_222_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    reduce_221_Mul_0_1_eltwise_222_conversion_0_1_eltwise_222_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 100);
    reduce_221_Mul_0_1_eltwise_222_conversion_0_1_eltwise_222_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 100);
    eltwise_222_output_array.data = AI_PTR(g_network_activations_map[0] + 179200);
    eltwise_222_output_array.data_start = AI_PTR(g_network_activations_map[0] + 179200);
    reduce_223_output_array.data = AI_PTR(g_network_activations_map[0] + 100);
    reduce_223_output_array.data_start = AI_PTR(g_network_activations_map[0] + 100);
    reduce_223_Mul_output_array.data = AI_PTR(g_network_activations_map[0] + 500);
    reduce_223_Mul_output_array.data_start = AI_PTR(g_network_activations_map[0] + 500);
    reduce_223_Mul_0_0_eltwise_224_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 100);
    reduce_223_Mul_0_0_eltwise_224_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 100);
    eltwise_224_output_array.data = AI_PTR(g_network_activations_map[0] + 200);
    eltwise_224_output_array.data_start = AI_PTR(g_network_activations_map[0] + 200);
    nl_225_output_array.data = AI_PTR(g_network_activations_map[0] + 100);
    nl_225_output_array.data_start = AI_PTR(g_network_activations_map[0] + 100);
    eltwise_226_output_array.data = AI_PTR(g_network_activations_map[0] + 200);
    eltwise_226_output_array.data_start = AI_PTR(g_network_activations_map[0] + 200);
    eltwise_228_output_array.data = AI_PTR(g_network_activations_map[0] + 76800);
    eltwise_228_output_array.data_start = AI_PTR(g_network_activations_map[0] + 76800);
    eltwise_229_output_array.data = AI_PTR(g_network_activations_map[0] + 102400);
    eltwise_229_output_array.data_start = AI_PTR(g_network_activations_map[0] + 102400);
    eltwise_227_output_array.data = AI_PTR(g_network_activations_map[0] + 76800);
    eltwise_227_output_array.data_start = AI_PTR(g_network_activations_map[0] + 76800);
    eltwise_230_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    eltwise_230_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    gemm_239_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 25600);
    gemm_239_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 25600);
    gemm_239_output_array.data = AI_PTR(g_network_activations_map[0] + 26624);
    gemm_239_output_array.data_start = AI_PTR(g_network_activations_map[0] + 26624);
    eltwise_241_output_array.data = AI_PTR(g_network_activations_map[0] + 77824);
    eltwise_241_output_array.data_start = AI_PTR(g_network_activations_map[0] + 77824);
    gemm_250_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 25600);
    gemm_250_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 25600);
    gemm_250_output_array.data = AI_PTR(g_network_activations_map[0] + 27648);
    gemm_250_output_array.data_start = AI_PTR(g_network_activations_map[0] + 27648);
    eltwise_252_output_array.data = AI_PTR(g_network_activations_map[0] + 53248);
    eltwise_252_output_array.data_start = AI_PTR(g_network_activations_map[0] + 53248);
    eltwise_253_output_array.data = AI_PTR(g_network_activations_map[0] + 25600);
    eltwise_253_output_array.data_start = AI_PTR(g_network_activations_map[0] + 25600);
    eltwise_253_0_0_reduce_254_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 51200);
    eltwise_253_0_0_reduce_254_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 51200);
    reduce_254_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    reduce_254_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    reduce_254_Mul_output_array.data = AI_PTR(g_network_activations_map[0] + 400);
    reduce_254_Mul_output_array.data_start = AI_PTR(g_network_activations_map[0] + 400);
    reduce_254_Mul_0_1_eltwise_255_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    reduce_254_Mul_0_1_eltwise_255_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    reduce_254_Mul_0_1_eltwise_255_conversion_0_1_eltwise_255_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 100);
    reduce_254_Mul_0_1_eltwise_255_conversion_0_1_eltwise_255_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 100);
    eltwise_255_output_array.data = AI_PTR(g_network_activations_map[0] + 153600);
    eltwise_255_output_array.data_start = AI_PTR(g_network_activations_map[0] + 153600);
    reduce_256_output_array.data = AI_PTR(g_network_activations_map[0] + 100);
    reduce_256_output_array.data_start = AI_PTR(g_network_activations_map[0] + 100);
    reduce_256_Mul_output_array.data = AI_PTR(g_network_activations_map[0] + 500);
    reduce_256_Mul_output_array.data_start = AI_PTR(g_network_activations_map[0] + 500);
    reduce_256_Mul_0_0_eltwise_257_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 100);
    reduce_256_Mul_0_0_eltwise_257_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 100);
    eltwise_257_output_array.data = AI_PTR(g_network_activations_map[0] + 200);
    eltwise_257_output_array.data_start = AI_PTR(g_network_activations_map[0] + 200);
    nl_258_output_array.data = AI_PTR(g_network_activations_map[0] + 100);
    nl_258_output_array.data_start = AI_PTR(g_network_activations_map[0] + 100);
    eltwise_259_output_array.data = AI_PTR(g_network_activations_map[0] + 51200);
    eltwise_259_output_array.data_start = AI_PTR(g_network_activations_map[0] + 51200);
    eltwise_261_output_array.data = AI_PTR(g_network_activations_map[0] + 76800);
    eltwise_261_output_array.data_start = AI_PTR(g_network_activations_map[0] + 76800);
    eltwise_262_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    eltwise_262_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    eltwise_260_output_array.data = AI_PTR(g_network_activations_map[0] + 76800);
    eltwise_260_output_array.data_start = AI_PTR(g_network_activations_map[0] + 76800);
    eltwise_263_output_array.data = AI_PTR(g_network_activations_map[0] + 25600);
    eltwise_263_output_array.data_start = AI_PTR(g_network_activations_map[0] + 25600);
    gemm_288_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    gemm_288_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    gemm_288_output_array.data = AI_PTR(g_network_activations_map[0] + 51200);
    gemm_288_output_array.data_start = AI_PTR(g_network_activations_map[0] + 51200);
    eltwise_290_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    eltwise_290_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    transpose_292_output_array.data = AI_PTR(g_network_activations_map[0] + 51200);
    transpose_292_output_array.data_start = AI_PTR(g_network_activations_map[0] + 51200);
    transpose_292_0_1_gemm_293_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 76800);
    transpose_292_0_1_gemm_293_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 76800);
    gemm_280_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    gemm_280_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    gemm_280_output_array.data = AI_PTR(g_network_activations_map[0] + 51200);
    gemm_280_output_array.data_start = AI_PTR(g_network_activations_map[0] + 51200);
    eltwise_282_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    eltwise_282_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    transpose_bgemm_285_out_output_array.data = AI_PTR(g_network_activations_map[0] + 51200);
    transpose_bgemm_285_out_output_array.data_start = AI_PTR(g_network_activations_map[0] + 51200);
    transpose_bgemm_285_out_0_1_gemm_285_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 179200);
    transpose_bgemm_285_out_0_1_gemm_285_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 179200);
    gemm_275_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    gemm_275_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    gemm_275_output_array.data = AI_PTR(g_network_activations_map[0] + 51200);
    gemm_275_output_array.data_start = AI_PTR(g_network_activations_map[0] + 51200);
    eltwise_277_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    eltwise_277_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    transpose_279_output_array.data = AI_PTR(g_network_activations_map[0] + 51200);
    transpose_279_output_array.data_start = AI_PTR(g_network_activations_map[0] + 51200);
    transpose_279_0_0_gemm_285_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 281600);
    transpose_279_0_0_gemm_285_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 281600);
    gemm_285_output_array.data = AI_PTR(g_network_activations_map[0] + 390820);
    gemm_285_output_array.data_start = AI_PTR(g_network_activations_map[0] + 390820);
    gemm_285_0_0_eltwise_286_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 179200);
    gemm_285_0_0_eltwise_286_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 179200);
    eltwise_286_output_array.data = AI_PTR(g_network_activations_map[0] + 219200);
    eltwise_286_output_array.data_start = AI_PTR(g_network_activations_map[0] + 219200);
    nl_287_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    nl_287_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    nl_287_output_array.data = AI_PTR(g_network_activations_map[0] + 179200);
    nl_287_output_array.data_start = AI_PTR(g_network_activations_map[0] + 179200);
    nl_287_0_0_gemm_293_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 219200);
    nl_287_0_0_gemm_293_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 219200);
    gemm_293_output_array.data = AI_PTR(g_network_activations_map[0] + 390416);
    gemm_293_output_array.data_start = AI_PTR(g_network_activations_map[0] + 390416);
    gemm_293_0_0_transpose_294_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    gemm_293_0_0_transpose_294_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    transpose_294_output_array.data = AI_PTR(g_network_activations_map[0] + 51200);
    transpose_294_output_array.data_start = AI_PTR(g_network_activations_map[0] + 51200);
    gemm_304_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    gemm_304_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    gemm_304_output_array.data = AI_PTR(g_network_activations_map[0] + 76800);
    gemm_304_output_array.data_start = AI_PTR(g_network_activations_map[0] + 76800);
    eltwise_306_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    eltwise_306_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    eltwise_307_output_array.data = AI_PTR(g_network_activations_map[0] + 51200);
    eltwise_307_output_array.data_start = AI_PTR(g_network_activations_map[0] + 51200);
    eltwise_307_0_0_reduce_308_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 76800);
    eltwise_307_0_0_reduce_308_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 76800);
    reduce_308_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    reduce_308_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    reduce_308_Mul_output_array.data = AI_PTR(g_network_activations_map[0] + 400);
    reduce_308_Mul_output_array.data_start = AI_PTR(g_network_activations_map[0] + 400);
    reduce_308_Mul_0_1_eltwise_309_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    reduce_308_Mul_0_1_eltwise_309_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    reduce_308_Mul_0_1_eltwise_309_conversion_0_1_eltwise_309_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 100);
    reduce_308_Mul_0_1_eltwise_309_conversion_0_1_eltwise_309_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 100);
    eltwise_309_output_array.data = AI_PTR(g_network_activations_map[0] + 179200);
    eltwise_309_output_array.data_start = AI_PTR(g_network_activations_map[0] + 179200);
    reduce_310_output_array.data = AI_PTR(g_network_activations_map[0] + 100);
    reduce_310_output_array.data_start = AI_PTR(g_network_activations_map[0] + 100);
    reduce_310_Mul_output_array.data = AI_PTR(g_network_activations_map[0] + 500);
    reduce_310_Mul_output_array.data_start = AI_PTR(g_network_activations_map[0] + 500);
    reduce_310_Mul_0_0_eltwise_311_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 100);
    reduce_310_Mul_0_0_eltwise_311_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 100);
    eltwise_311_output_array.data = AI_PTR(g_network_activations_map[0] + 200);
    eltwise_311_output_array.data_start = AI_PTR(g_network_activations_map[0] + 200);
    nl_312_output_array.data = AI_PTR(g_network_activations_map[0] + 100);
    nl_312_output_array.data_start = AI_PTR(g_network_activations_map[0] + 100);
    eltwise_313_output_array.data = AI_PTR(g_network_activations_map[0] + 200);
    eltwise_313_output_array.data_start = AI_PTR(g_network_activations_map[0] + 200);
    eltwise_315_output_array.data = AI_PTR(g_network_activations_map[0] + 76800);
    eltwise_315_output_array.data_start = AI_PTR(g_network_activations_map[0] + 76800);
    eltwise_316_output_array.data = AI_PTR(g_network_activations_map[0] + 102400);
    eltwise_316_output_array.data_start = AI_PTR(g_network_activations_map[0] + 102400);
    eltwise_314_output_array.data = AI_PTR(g_network_activations_map[0] + 76800);
    eltwise_314_output_array.data_start = AI_PTR(g_network_activations_map[0] + 76800);
    eltwise_317_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    eltwise_317_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    gemm_326_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 25600);
    gemm_326_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 25600);
    gemm_326_output_array.data = AI_PTR(g_network_activations_map[0] + 26624);
    gemm_326_output_array.data_start = AI_PTR(g_network_activations_map[0] + 26624);
    eltwise_328_output_array.data = AI_PTR(g_network_activations_map[0] + 77824);
    eltwise_328_output_array.data_start = AI_PTR(g_network_activations_map[0] + 77824);
    gemm_337_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 25600);
    gemm_337_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 25600);
    gemm_337_output_array.data = AI_PTR(g_network_activations_map[0] + 27648);
    gemm_337_output_array.data_start = AI_PTR(g_network_activations_map[0] + 27648);
    eltwise_339_output_array.data = AI_PTR(g_network_activations_map[0] + 53248);
    eltwise_339_output_array.data_start = AI_PTR(g_network_activations_map[0] + 53248);
    eltwise_340_output_array.data = AI_PTR(g_network_activations_map[0] + 25600);
    eltwise_340_output_array.data_start = AI_PTR(g_network_activations_map[0] + 25600);
    eltwise_340_0_0_reduce_341_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 51200);
    eltwise_340_0_0_reduce_341_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 51200);
    reduce_341_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    reduce_341_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    reduce_341_Mul_output_array.data = AI_PTR(g_network_activations_map[0] + 400);
    reduce_341_Mul_output_array.data_start = AI_PTR(g_network_activations_map[0] + 400);
    reduce_341_Mul_0_1_eltwise_342_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    reduce_341_Mul_0_1_eltwise_342_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    reduce_341_Mul_0_1_eltwise_342_conversion_0_1_eltwise_342_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 100);
    reduce_341_Mul_0_1_eltwise_342_conversion_0_1_eltwise_342_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 100);
    eltwise_342_output_array.data = AI_PTR(g_network_activations_map[0] + 153600);
    eltwise_342_output_array.data_start = AI_PTR(g_network_activations_map[0] + 153600);
    reduce_343_output_array.data = AI_PTR(g_network_activations_map[0] + 100);
    reduce_343_output_array.data_start = AI_PTR(g_network_activations_map[0] + 100);
    reduce_343_Mul_output_array.data = AI_PTR(g_network_activations_map[0] + 500);
    reduce_343_Mul_output_array.data_start = AI_PTR(g_network_activations_map[0] + 500);
    reduce_343_Mul_0_0_eltwise_344_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 100);
    reduce_343_Mul_0_0_eltwise_344_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 100);
    eltwise_344_output_array.data = AI_PTR(g_network_activations_map[0] + 200);
    eltwise_344_output_array.data_start = AI_PTR(g_network_activations_map[0] + 200);
    nl_345_output_array.data = AI_PTR(g_network_activations_map[0] + 100);
    nl_345_output_array.data_start = AI_PTR(g_network_activations_map[0] + 100);
    eltwise_346_output_array.data = AI_PTR(g_network_activations_map[0] + 51200);
    eltwise_346_output_array.data_start = AI_PTR(g_network_activations_map[0] + 51200);
    eltwise_348_output_array.data = AI_PTR(g_network_activations_map[0] + 76800);
    eltwise_348_output_array.data_start = AI_PTR(g_network_activations_map[0] + 76800);
    eltwise_349_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    eltwise_349_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    eltwise_347_output_array.data = AI_PTR(g_network_activations_map[0] + 76800);
    eltwise_347_output_array.data_start = AI_PTR(g_network_activations_map[0] + 76800);
    eltwise_350_output_array.data = AI_PTR(g_network_activations_map[0] + 25600);
    eltwise_350_output_array.data_start = AI_PTR(g_network_activations_map[0] + 25600);
    gemm_359_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    gemm_359_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    gemm_359_output_array.data = AI_PTR(g_network_activations_map[0] + 1024);
    gemm_359_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1024);
    eltwise_361_output_array.data = AI_PTR(g_network_activations_map[0] + 7624);
    eltwise_361_output_array.data_start = AI_PTR(g_network_activations_map[0] + 7624);
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
    
    gemm_293_bias_array.format |= AI_FMT_FLAG_CONST;
    gemm_293_bias_array.data = AI_PTR(g_network_weights_map[0] + 0);
    gemm_293_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 0);
    gemm_285_bias_array.format |= AI_FMT_FLAG_CONST;
    gemm_285_bias_array.data = AI_PTR(g_network_weights_map[0] + 4);
    gemm_285_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 4);
    gemm_206_bias_array.format |= AI_FMT_FLAG_CONST;
    gemm_206_bias_array.data = AI_PTR(g_network_weights_map[0] + 8);
    gemm_206_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 8);
    gemm_198_bias_array.format |= AI_FMT_FLAG_CONST;
    gemm_198_bias_array.data = AI_PTR(g_network_weights_map[0] + 12);
    gemm_198_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 12);
    gemm_119_bias_array.format |= AI_FMT_FLAG_CONST;
    gemm_119_bias_array.data = AI_PTR(g_network_weights_map[0] + 16);
    gemm_119_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 16);
    gemm_111_bias_array.format |= AI_FMT_FLAG_CONST;
    gemm_111_bias_array.data = AI_PTR(g_network_weights_map[0] + 20);
    gemm_111_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 20);
    gemm_32_bias_array.format |= AI_FMT_FLAG_CONST;
    gemm_32_bias_array.data = AI_PTR(g_network_weights_map[0] + 24);
    gemm_32_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 24);
    gemm_24_bias_array.format |= AI_FMT_FLAG_CONST;
    gemm_24_bias_array.data = AI_PTR(g_network_weights_map[0] + 28);
    gemm_24_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 28);
    tiny_bert_generator_tf_math_multiply_Mul_y_3D_array.format |= AI_FMT_FLAG_CONST;
    tiny_bert_generator_tf_math_multiply_Mul_y_3D_array.data = AI_PTR(g_network_weights_map[0] + 32);
    tiny_bert_generator_tf_math_multiply_Mul_y_3D_array.data_start = AI_PTR(g_network_weights_map[0] + 32);
    encoder_layer_multi_head_attention_8_dense_49_bias_3D_array.format |= AI_FMT_FLAG_CONST;
    encoder_layer_multi_head_attention_8_dense_49_bias_3D_array.data = AI_PTR(g_network_weights_map[0] + 36);
    encoder_layer_multi_head_attention_8_dense_49_bias_3D_array.data_start = AI_PTR(g_network_weights_map[0] + 36);
    encoder_layer_multi_head_attention_8_dense_50_bias_3D_array.format |= AI_FMT_FLAG_CONST;
    encoder_layer_multi_head_attention_8_dense_50_bias_3D_array.data = AI_PTR(g_network_weights_map[0] + 292);
    encoder_layer_multi_head_attention_8_dense_50_bias_3D_array.data_start = AI_PTR(g_network_weights_map[0] + 292);
    tiny_bert_generator_encoder_layer_3_multi_head_attention_11_truedivtiny_bert_generator_encoder_layer_3_multi_head_attention_11_Sqrt_4D_array.format |= AI_FMT_FLAG_CONST;
    tiny_bert_generator_encoder_layer_3_multi_head_attention_11_truedivtiny_bert_generator_encoder_layer_3_multi_head_attention_11_Sqrt_4D_array.data = AI_PTR(g_network_weights_map[0] + 548);
    tiny_bert_generator_encoder_layer_3_multi_head_attention_11_truedivtiny_bert_generator_encoder_layer_3_multi_head_attention_11_Sqrt_4D_array.data_start = AI_PTR(g_network_weights_map[0] + 548);
    encoder_layer_multi_head_attention_8_dense_51_bias_3D_array.format |= AI_FMT_FLAG_CONST;
    encoder_layer_multi_head_attention_8_dense_51_bias_3D_array.data = AI_PTR(g_network_weights_map[0] + 552);
    encoder_layer_multi_head_attention_8_dense_51_bias_3D_array.data_start = AI_PTR(g_network_weights_map[0] + 552);
    encoder_layer_multi_head_attention_8_dense_52_bias_3D_array.format |= AI_FMT_FLAG_CONST;
    encoder_layer_multi_head_attention_8_dense_52_bias_3D_array.data = AI_PTR(g_network_weights_map[0] + 808);
    encoder_layer_multi_head_attention_8_dense_52_bias_3D_array.data_start = AI_PTR(g_network_weights_map[0] + 808);
    tiny_bert_generator_encoder_layer_layer_normalization_16_batchnorm_add_y_3D_array.format |= AI_FMT_FLAG_CONST;
    tiny_bert_generator_encoder_layer_layer_normalization_16_batchnorm_add_y_3D_array.data = AI_PTR(g_network_weights_map[0] + 1064);
    tiny_bert_generator_encoder_layer_layer_normalization_16_batchnorm_add_y_3D_array.data_start = AI_PTR(g_network_weights_map[0] + 1064);
    encoder_layer_layer_normalization_16_gamma_3D_array.format |= AI_FMT_FLAG_CONST;
    encoder_layer_layer_normalization_16_gamma_3D_array.data = AI_PTR(g_network_weights_map[0] + 1068);
    encoder_layer_layer_normalization_16_gamma_3D_array.data_start = AI_PTR(g_network_weights_map[0] + 1068);
    encoder_layer_layer_normalization_16_beta_3D_array.format |= AI_FMT_FLAG_CONST;
    encoder_layer_layer_normalization_16_beta_3D_array.data = AI_PTR(g_network_weights_map[0] + 1324);
    encoder_layer_layer_normalization_16_beta_3D_array.data_start = AI_PTR(g_network_weights_map[0] + 1324);
    dense_53_bias_3D_array.format |= AI_FMT_FLAG_CONST;
    dense_53_bias_3D_array.data = AI_PTR(g_network_weights_map[0] + 1580);
    dense_53_bias_3D_array.data_start = AI_PTR(g_network_weights_map[0] + 1580);
    dense_54_bias_3D_array.format |= AI_FMT_FLAG_CONST;
    dense_54_bias_3D_array.data = AI_PTR(g_network_weights_map[0] + 2092);
    dense_54_bias_3D_array.data_start = AI_PTR(g_network_weights_map[0] + 2092);
    encoder_layer_layer_normalization_17_gamma_3D_array.format |= AI_FMT_FLAG_CONST;
    encoder_layer_layer_normalization_17_gamma_3D_array.data = AI_PTR(g_network_weights_map[0] + 2348);
    encoder_layer_layer_normalization_17_gamma_3D_array.data_start = AI_PTR(g_network_weights_map[0] + 2348);
    encoder_layer_layer_normalization_17_beta_3D_array.format |= AI_FMT_FLAG_CONST;
    encoder_layer_layer_normalization_17_beta_3D_array.data = AI_PTR(g_network_weights_map[0] + 2604);
    encoder_layer_layer_normalization_17_beta_3D_array.data_start = AI_PTR(g_network_weights_map[0] + 2604);
    encoder_layer_1_multi_head_attention_9_dense_55_bias_3D_array.format |= AI_FMT_FLAG_CONST;
    encoder_layer_1_multi_head_attention_9_dense_55_bias_3D_array.data = AI_PTR(g_network_weights_map[0] + 2860);
    encoder_layer_1_multi_head_attention_9_dense_55_bias_3D_array.data_start = AI_PTR(g_network_weights_map[0] + 2860);
    encoder_layer_1_multi_head_attention_9_dense_56_bias_3D_array.format |= AI_FMT_FLAG_CONST;
    encoder_layer_1_multi_head_attention_9_dense_56_bias_3D_array.data = AI_PTR(g_network_weights_map[0] + 3116);
    encoder_layer_1_multi_head_attention_9_dense_56_bias_3D_array.data_start = AI_PTR(g_network_weights_map[0] + 3116);
    encoder_layer_1_multi_head_attention_9_dense_57_bias_3D_array.format |= AI_FMT_FLAG_CONST;
    encoder_layer_1_multi_head_attention_9_dense_57_bias_3D_array.data = AI_PTR(g_network_weights_map[0] + 3372);
    encoder_layer_1_multi_head_attention_9_dense_57_bias_3D_array.data_start = AI_PTR(g_network_weights_map[0] + 3372);
    encoder_layer_1_multi_head_attention_9_dense_58_bias_3D_array.format |= AI_FMT_FLAG_CONST;
    encoder_layer_1_multi_head_attention_9_dense_58_bias_3D_array.data = AI_PTR(g_network_weights_map[0] + 3628);
    encoder_layer_1_multi_head_attention_9_dense_58_bias_3D_array.data_start = AI_PTR(g_network_weights_map[0] + 3628);
    encoder_layer_1_layer_normalization_18_gamma_3D_array.format |= AI_FMT_FLAG_CONST;
    encoder_layer_1_layer_normalization_18_gamma_3D_array.data = AI_PTR(g_network_weights_map[0] + 3884);
    encoder_layer_1_layer_normalization_18_gamma_3D_array.data_start = AI_PTR(g_network_weights_map[0] + 3884);
    encoder_layer_1_layer_normalization_18_beta_3D_array.format |= AI_FMT_FLAG_CONST;
    encoder_layer_1_layer_normalization_18_beta_3D_array.data = AI_PTR(g_network_weights_map[0] + 4140);
    encoder_layer_1_layer_normalization_18_beta_3D_array.data_start = AI_PTR(g_network_weights_map[0] + 4140);
    dense_59_bias_3D_array.format |= AI_FMT_FLAG_CONST;
    dense_59_bias_3D_array.data = AI_PTR(g_network_weights_map[0] + 4396);
    dense_59_bias_3D_array.data_start = AI_PTR(g_network_weights_map[0] + 4396);
    dense_60_bias_3D_array.format |= AI_FMT_FLAG_CONST;
    dense_60_bias_3D_array.data = AI_PTR(g_network_weights_map[0] + 4908);
    dense_60_bias_3D_array.data_start = AI_PTR(g_network_weights_map[0] + 4908);
    encoder_layer_1_layer_normalization_19_gamma_3D_array.format |= AI_FMT_FLAG_CONST;
    encoder_layer_1_layer_normalization_19_gamma_3D_array.data = AI_PTR(g_network_weights_map[0] + 5164);
    encoder_layer_1_layer_normalization_19_gamma_3D_array.data_start = AI_PTR(g_network_weights_map[0] + 5164);
    encoder_layer_1_layer_normalization_19_beta_3D_array.format |= AI_FMT_FLAG_CONST;
    encoder_layer_1_layer_normalization_19_beta_3D_array.data = AI_PTR(g_network_weights_map[0] + 5420);
    encoder_layer_1_layer_normalization_19_beta_3D_array.data_start = AI_PTR(g_network_weights_map[0] + 5420);
    encoder_layer_2_multi_head_attention_10_dense_61_bias_3D_array.format |= AI_FMT_FLAG_CONST;
    encoder_layer_2_multi_head_attention_10_dense_61_bias_3D_array.data = AI_PTR(g_network_weights_map[0] + 5676);
    encoder_layer_2_multi_head_attention_10_dense_61_bias_3D_array.data_start = AI_PTR(g_network_weights_map[0] + 5676);
    encoder_layer_2_multi_head_attention_10_dense_62_bias_3D_array.format |= AI_FMT_FLAG_CONST;
    encoder_layer_2_multi_head_attention_10_dense_62_bias_3D_array.data = AI_PTR(g_network_weights_map[0] + 5932);
    encoder_layer_2_multi_head_attention_10_dense_62_bias_3D_array.data_start = AI_PTR(g_network_weights_map[0] + 5932);
    encoder_layer_2_multi_head_attention_10_dense_63_bias_3D_array.format |= AI_FMT_FLAG_CONST;
    encoder_layer_2_multi_head_attention_10_dense_63_bias_3D_array.data = AI_PTR(g_network_weights_map[0] + 6188);
    encoder_layer_2_multi_head_attention_10_dense_63_bias_3D_array.data_start = AI_PTR(g_network_weights_map[0] + 6188);
    encoder_layer_2_multi_head_attention_10_dense_64_bias_3D_array.format |= AI_FMT_FLAG_CONST;
    encoder_layer_2_multi_head_attention_10_dense_64_bias_3D_array.data = AI_PTR(g_network_weights_map[0] + 6444);
    encoder_layer_2_multi_head_attention_10_dense_64_bias_3D_array.data_start = AI_PTR(g_network_weights_map[0] + 6444);
    encoder_layer_2_layer_normalization_20_gamma_3D_array.format |= AI_FMT_FLAG_CONST;
    encoder_layer_2_layer_normalization_20_gamma_3D_array.data = AI_PTR(g_network_weights_map[0] + 6700);
    encoder_layer_2_layer_normalization_20_gamma_3D_array.data_start = AI_PTR(g_network_weights_map[0] + 6700);
    encoder_layer_2_layer_normalization_20_beta_3D_array.format |= AI_FMT_FLAG_CONST;
    encoder_layer_2_layer_normalization_20_beta_3D_array.data = AI_PTR(g_network_weights_map[0] + 6956);
    encoder_layer_2_layer_normalization_20_beta_3D_array.data_start = AI_PTR(g_network_weights_map[0] + 6956);
    dense_65_bias_3D_array.format |= AI_FMT_FLAG_CONST;
    dense_65_bias_3D_array.data = AI_PTR(g_network_weights_map[0] + 7212);
    dense_65_bias_3D_array.data_start = AI_PTR(g_network_weights_map[0] + 7212);
    dense_66_bias_3D_array.format |= AI_FMT_FLAG_CONST;
    dense_66_bias_3D_array.data = AI_PTR(g_network_weights_map[0] + 7724);
    dense_66_bias_3D_array.data_start = AI_PTR(g_network_weights_map[0] + 7724);
    encoder_layer_2_layer_normalization_21_gamma_3D_array.format |= AI_FMT_FLAG_CONST;
    encoder_layer_2_layer_normalization_21_gamma_3D_array.data = AI_PTR(g_network_weights_map[0] + 7980);
    encoder_layer_2_layer_normalization_21_gamma_3D_array.data_start = AI_PTR(g_network_weights_map[0] + 7980);
    encoder_layer_2_layer_normalization_21_beta_3D_array.format |= AI_FMT_FLAG_CONST;
    encoder_layer_2_layer_normalization_21_beta_3D_array.data = AI_PTR(g_network_weights_map[0] + 8236);
    encoder_layer_2_layer_normalization_21_beta_3D_array.data_start = AI_PTR(g_network_weights_map[0] + 8236);
    encoder_layer_3_multi_head_attention_11_dense_67_bias_3D_array.format |= AI_FMT_FLAG_CONST;
    encoder_layer_3_multi_head_attention_11_dense_67_bias_3D_array.data = AI_PTR(g_network_weights_map[0] + 8492);
    encoder_layer_3_multi_head_attention_11_dense_67_bias_3D_array.data_start = AI_PTR(g_network_weights_map[0] + 8492);
    encoder_layer_3_multi_head_attention_11_dense_68_bias_3D_array.format |= AI_FMT_FLAG_CONST;
    encoder_layer_3_multi_head_attention_11_dense_68_bias_3D_array.data = AI_PTR(g_network_weights_map[0] + 8748);
    encoder_layer_3_multi_head_attention_11_dense_68_bias_3D_array.data_start = AI_PTR(g_network_weights_map[0] + 8748);
    encoder_layer_3_multi_head_attention_11_dense_69_bias_3D_array.format |= AI_FMT_FLAG_CONST;
    encoder_layer_3_multi_head_attention_11_dense_69_bias_3D_array.data = AI_PTR(g_network_weights_map[0] + 9004);
    encoder_layer_3_multi_head_attention_11_dense_69_bias_3D_array.data_start = AI_PTR(g_network_weights_map[0] + 9004);
    encoder_layer_3_multi_head_attention_11_dense_70_bias_3D_array.format |= AI_FMT_FLAG_CONST;
    encoder_layer_3_multi_head_attention_11_dense_70_bias_3D_array.data = AI_PTR(g_network_weights_map[0] + 9260);
    encoder_layer_3_multi_head_attention_11_dense_70_bias_3D_array.data_start = AI_PTR(g_network_weights_map[0] + 9260);
    encoder_layer_3_layer_normalization_22_gamma_3D_array.format |= AI_FMT_FLAG_CONST;
    encoder_layer_3_layer_normalization_22_gamma_3D_array.data = AI_PTR(g_network_weights_map[0] + 9516);
    encoder_layer_3_layer_normalization_22_gamma_3D_array.data_start = AI_PTR(g_network_weights_map[0] + 9516);
    encoder_layer_3_layer_normalization_22_beta_3D_array.format |= AI_FMT_FLAG_CONST;
    encoder_layer_3_layer_normalization_22_beta_3D_array.data = AI_PTR(g_network_weights_map[0] + 9772);
    encoder_layer_3_layer_normalization_22_beta_3D_array.data_start = AI_PTR(g_network_weights_map[0] + 9772);
    dense_71_bias_3D_array.format |= AI_FMT_FLAG_CONST;
    dense_71_bias_3D_array.data = AI_PTR(g_network_weights_map[0] + 10028);
    dense_71_bias_3D_array.data_start = AI_PTR(g_network_weights_map[0] + 10028);
    dense_72_bias_3D_array.format |= AI_FMT_FLAG_CONST;
    dense_72_bias_3D_array.data = AI_PTR(g_network_weights_map[0] + 10540);
    dense_72_bias_3D_array.data_start = AI_PTR(g_network_weights_map[0] + 10540);
    encoder_layer_3_layer_normalization_23_gamma_3D_array.format |= AI_FMT_FLAG_CONST;
    encoder_layer_3_layer_normalization_23_gamma_3D_array.data = AI_PTR(g_network_weights_map[0] + 10796);
    encoder_layer_3_layer_normalization_23_gamma_3D_array.data_start = AI_PTR(g_network_weights_map[0] + 10796);
    encoder_layer_3_layer_normalization_23_beta_3D_array.format |= AI_FMT_FLAG_CONST;
    encoder_layer_3_layer_normalization_23_beta_3D_array.data = AI_PTR(g_network_weights_map[0] + 11052);
    encoder_layer_3_layer_normalization_23_beta_3D_array.data_start = AI_PTR(g_network_weights_map[0] + 11052);
    dense_24_bias_3D_array.format |= AI_FMT_FLAG_CONST;
    dense_24_bias_3D_array.data = AI_PTR(g_network_weights_map[0] + 11308);
    dense_24_bias_3D_array.data_start = AI_PTR(g_network_weights_map[0] + 11308);
    tiny_bert_generator_tf___operators___add_y_array.format |= AI_FMT_FLAG_CONST;
    tiny_bert_generator_tf___operators___add_y_array.data = AI_PTR(g_network_weights_map[0] + 11376);
    tiny_bert_generator_tf___operators___add_y_array.data_start = AI_PTR(g_network_weights_map[0] + 11376);
    embedding_embeddings_array.format |= AI_FMT_FLAG_CONST;
    embedding_embeddings_array.data = AI_PTR(g_network_weights_map[0] + 36976);
    embedding_embeddings_array.data_start = AI_PTR(g_network_weights_map[0] + 36976);
    gemm_27_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_27_weights_array.data = AI_PTR(g_network_weights_map[0] + 53872);
    gemm_27_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 53872);
    gemm_27_bias_array.format |= AI_FMT_FLAG_CONST;
    gemm_27_bias_array.data = AI_PTR(g_network_weights_map[0] + 119408);
    gemm_27_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 119408);
    gemm_19_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_19_weights_array.data = AI_PTR(g_network_weights_map[0] + 120432);
    gemm_19_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 120432);
    gemm_14_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_14_weights_array.data = AI_PTR(g_network_weights_map[0] + 185968);
    gemm_14_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 185968);
    gemm_43_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_43_weights_array.data = AI_PTR(g_network_weights_map[0] + 251504);
    gemm_43_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 251504);
    reduce_47_Mul_scale_array.format |= AI_FMT_FLAG_CONST;
    reduce_47_Mul_scale_array.data = AI_PTR(g_network_weights_map[0] + 317040);
    reduce_47_Mul_scale_array.data_start = AI_PTR(g_network_weights_map[0] + 317040);
    reduce_47_Mul_bias_array.format |= AI_FMT_FLAG_CONST;
    reduce_47_Mul_bias_array.data = AI_PTR(g_network_weights_map[0] + 317044);
    reduce_47_Mul_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 317044);
    gemm_65_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_65_weights_array.data = AI_PTR(g_network_weights_map[0] + 317048);
    gemm_65_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 317048);
    gemm_65_bias_array.format |= AI_FMT_FLAG_CONST;
    gemm_65_bias_array.data = AI_PTR(g_network_weights_map[0] + 448120);
    gemm_65_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 448120);
    gemm_76_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_76_weights_array.data = AI_PTR(g_network_weights_map[0] + 450168);
    gemm_76_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 450168);
    gemm_114_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_114_weights_array.data = AI_PTR(g_network_weights_map[0] + 581240);
    gemm_114_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 581240);
    gemm_106_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_106_weights_array.data = AI_PTR(g_network_weights_map[0] + 646776);
    gemm_106_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 646776);
    gemm_101_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_101_weights_array.data = AI_PTR(g_network_weights_map[0] + 712312);
    gemm_101_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 712312);
    gemm_130_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_130_weights_array.data = AI_PTR(g_network_weights_map[0] + 777848);
    gemm_130_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 777848);
    gemm_152_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_152_weights_array.data = AI_PTR(g_network_weights_map[0] + 843384);
    gemm_152_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 843384);
    gemm_163_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_163_weights_array.data = AI_PTR(g_network_weights_map[0] + 974456);
    gemm_163_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 974456);
    gemm_201_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_201_weights_array.data = AI_PTR(g_network_weights_map[0] + 1105528);
    gemm_201_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 1105528);
    gemm_193_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_193_weights_array.data = AI_PTR(g_network_weights_map[0] + 1171064);
    gemm_193_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 1171064);
    gemm_188_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_188_weights_array.data = AI_PTR(g_network_weights_map[0] + 1236600);
    gemm_188_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 1236600);
    gemm_217_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_217_weights_array.data = AI_PTR(g_network_weights_map[0] + 1302136);
    gemm_217_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 1302136);
    gemm_239_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_239_weights_array.data = AI_PTR(g_network_weights_map[0] + 1367672);
    gemm_239_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 1367672);
    gemm_250_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_250_weights_array.data = AI_PTR(g_network_weights_map[0] + 1498744);
    gemm_250_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 1498744);
    gemm_288_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_288_weights_array.data = AI_PTR(g_network_weights_map[0] + 1629816);
    gemm_288_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 1629816);
    gemm_280_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_280_weights_array.data = AI_PTR(g_network_weights_map[0] + 1695352);
    gemm_280_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 1695352);
    gemm_275_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_275_weights_array.data = AI_PTR(g_network_weights_map[0] + 1760888);
    gemm_275_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 1760888);
    gemm_304_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_304_weights_array.data = AI_PTR(g_network_weights_map[0] + 1826424);
    gemm_304_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 1826424);
    gemm_326_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_326_weights_array.data = AI_PTR(g_network_weights_map[0] + 1891960);
    gemm_326_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 1891960);
    gemm_337_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_337_weights_array.data = AI_PTR(g_network_weights_map[0] + 2023032);
    gemm_337_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 2023032);
    gemm_359_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_359_weights_array.data = AI_PTR(g_network_weights_map[0] + 2154104);
    gemm_359_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 2154104);
    gemm_359_bias_array.format |= AI_FMT_FLAG_CONST;
    gemm_359_bias_array.data = AI_PTR(g_network_weights_map[0] + 2171000);
    gemm_359_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 2171000);
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
      .signature         = 0xa6b1f45d,
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
      .signature         = 0xa6b1f45d,
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

