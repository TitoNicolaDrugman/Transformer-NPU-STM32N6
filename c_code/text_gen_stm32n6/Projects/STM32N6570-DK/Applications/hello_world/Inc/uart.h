/**
  ******************************************************************************
  * @file    uart.h 
  * @author  GPM/AIS Application Team
  * @brief   UART driver
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

#ifndef __UART_H__
#define __UART_H__

#include <stdint.h>
int uart_write(const uint8_t* buff, const int size);
int uart_read(uint8_t* buff, const int size, uint32_t timeout);
#endif /* __UART_H__ */