/**
  ******************************************************************************
  * @file    app_config.h
  * @author  GPM/AIS Application Team
  * @brief   APP configuration
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2023 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */

#ifndef __APP_CONFIG_H__
#define __APP_CONFIG_H__

#define VDDCORE_OVERDRIVE               0               // Use Overdrive mode (quick clocks) or not (normal clocks)

#define USE_UART_BAUDRATE               115200          /* 921600 115200 */
#define USE_MCU_DCACHE                  1
#define USE_MCU_ICACHE                  1
#define USE_EXTERNAL_MEMORY_DEVICES     1

#define USE_NPU_CACHE           // Used to open RISAFs for the NPU cache

#endif /* __APP_CONFIG_H__ */
