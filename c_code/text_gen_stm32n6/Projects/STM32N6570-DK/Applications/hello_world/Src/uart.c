/**
  ******************************************************************************
  * @file    uart.c 
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

#include "uart.h"
#include "app_config.h"

#include "stm32n6xx_hal.h"
#include "stm32n6xx_ll_usart.h"


extern UART_HandleTypeDef UartHandle;

#define _UART_INSTANCE UartHandle.Instance

/* DISCO & NUCLEO USART1 (PE5/PE6) */
int uart_write(const uint8_t* buff, const int size)
{
  int cpt = (size>0)?size:0;
  int idx = 0;
  
  if ((!buff) || (!cpt))
    return 0;
  
  while (cpt--) 
  {
    while (!LL_USART_IsActiveFlag_TXE(_UART_INSTANCE))
    {
    }
    LL_USART_TransmitData8(_UART_INSTANCE, buff[idx++]);
  }
  while (!LL_USART_IsActiveFlag_TC(_UART_INSTANCE)) {};
  
  return idx;
}


int uart_read(uint8_t* buff, const int size, uint32_t timeout)
{
  int cpt = (size>0)?size:0;
  int idx = 0;
  
  if ((!buff) || (!cpt)) 
    return 0;
  
  if (timeout == 0)
    timeout = HAL_MAX_DELAY;
  
  if (timeout != HAL_MAX_DELAY)
  {
    uint32_t Tickstart = HAL_GetTick();
    
    while (cpt) 
    {
      if (LL_USART_IsActiveFlag_RXNE(_UART_INSTANCE))
      {
        buff[idx++] = LL_USART_ReceiveData8(_UART_INSTANCE);
        cpt--;
      } else {
        if ((HAL_GetTick() - Tickstart) > timeout)
          return -1;
      }
    }
    
  } else {
    
    while (cpt--) 
    {
      while (!LL_USART_IsActiveFlag_RXNE(_UART_INSTANCE))
      {
      }
      buff[idx++] = LL_USART_ReceiveData8(_UART_INSTANCE);
    }
  }
  
  return idx;
}
