################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (13.3.rel1)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
C:/Users/drugm/stm32ai-modelzoo-services/dd2_text_gen_stm32n6v2/text_gen_stm32n6/Projects/STM32N6570-DK/Applications/hello_world/Drivers/BSP/Components/mx66uw1g45g/mx66uw1g45g.c 

OBJS += \
./Drivers/BSP/Components/mx66uw1g45g/mx66uw1g45g.o 

C_DEPS += \
./Drivers/BSP/Components/mx66uw1g45g/mx66uw1g45g.d 


# Each subdirectory must supply rules for building sources it contributes
Drivers/BSP/Components/mx66uw1g45g/mx66uw1g45g.o: C:/Users/drugm/stm32ai-modelzoo-services/dd2_text_gen_stm32n6v2/text_gen_stm32n6/Projects/STM32N6570-DK/Applications/hello_world/Drivers/BSP/Components/mx66uw1g45g/mx66uw1g45g.c Drivers/BSP/Components/mx66uw1g45g/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m55 -std=gnu11 -g3 -DSTM32N657xx -DUSE_FULL_ASSERT -DVECT_TAB_SRAM -DUSE_FULL_LL_DRIVER -DUSER_VECT_TAB_ADDRESS -DDEBUG -DLL_ATON_DUMP_DEBUG_API -DLL_ATON_PLATFORM=LL_ATON_PLAT_STM32N6 -DLL_ATON_OSAL=LL_ATON_OSAL_BARE_METAL -DLL_ATON_RT_MODE=LL_ATON_RT_ASYNC -DLL_ATON_SW_FALLBACK -DLL_ATON_EB_DBG_INFO -DNUCLEO_N6_CONFIG=0 -DLL_ATON_DBG_BUFFER_INFO_EXCLUDED=1 -c -IC:/Users/drugm/stm32ai-modelzoo-services/dd2_text_gen_stm32n6v2/text_gen_stm32n6/Projects/STM32N6570-DK/Applications/hello_world/cubeIDE/../Inc -IC:/Users/drugm/stm32ai-modelzoo-services/dd2_text_gen_stm32n6v2/text_gen_stm32n6/Projects/STM32N6570-DK/Applications/hello_world/cubeIDE/.. -IC:/Users/drugm/stm32ai-modelzoo-services/dd2_text_gen_stm32n6v2/text_gen_stm32n6/Projects/STM32N6570-DK/Applications/hello_world/cubeIDE/../AI/atonn -IC:/Users/drugm/stm32ai-modelzoo-services/dd2_text_gen_stm32n6v2/text_gen_stm32n6/Projects/STM32N6570-DK/Applications/hello_world/cubeIDE/../../../../../Middlewares/ST/AI/Npu/ll_aton -IC:/Users/drugm/stm32ai-modelzoo-services/dd2_text_gen_stm32n6v2/text_gen_stm32n6/Projects/STM32N6570-DK/Applications/hello_world/cubeIDE/../../../../../Middlewares/ST/AI/Inc -IC:/Users/drugm/stm32ai-modelzoo-services/dd2_text_gen_stm32n6v2/text_gen_stm32n6/Projects/STM32N6570-DK/Applications/hello_world/cubeIDE/../Drivers/BSP/Components/mx66uw1g45g -IC:/Users/drugm/stm32ai-modelzoo-services/dd2_text_gen_stm32n6v2/text_gen_stm32n6/Projects/STM32N6570-DK/Applications/hello_world/cubeIDE/../Drivers/BSP/STM32N6570-DK -IC:/Users/drugm/stm32ai-modelzoo-services/dd2_text_gen_stm32n6v2/text_gen_stm32n6/Projects/STM32N6570-DK/Applications/hello_world/cubeIDE/../Drivers/BSP/STM32N6xx_Nucleo -IC:/Users/drugm/stm32ai-modelzoo-services/dd2_text_gen_stm32n6v2/text_gen_stm32n6/Projects/STM32N6570-DK/Applications/hello_world/cubeIDE/../Drivers/CMSIS/Core/Include -IC:/Users/drugm/stm32ai-modelzoo-services/dd2_text_gen_stm32n6v2/text_gen_stm32n6/Projects/STM32N6570-DK/Applications/hello_world/cubeIDE/../Drivers/CMSIS/Device/ST/STM32N6xx/Include -IC:/Users/drugm/stm32ai-modelzoo-services/dd2_text_gen_stm32n6v2/text_gen_stm32n6/Projects/STM32N6570-DK/Applications/hello_world/cubeIDE/../Drivers/CMSIS/Device/ST/STM32N6xx/Include/Templates -IC:/Users/drugm/stm32ai-modelzoo-services/dd2_text_gen_stm32n6v2/text_gen_stm32n6/Projects/STM32N6570-DK/Applications/hello_world/cubeIDE/../Drivers/CMSIS/Core/DSP/Include -IC:/Users/drugm/stm32ai-modelzoo-services/dd2_text_gen_stm32n6v2/text_gen_stm32n6/Projects/STM32N6570-DK/Applications/hello_world/cubeIDE/../Drivers/STM32N6xx_HAL_Driver/Inc -O0 -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -mcmse -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv5-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-Drivers-2f-BSP-2f-Components-2f-mx66uw1g45g

clean-Drivers-2f-BSP-2f-Components-2f-mx66uw1g45g:
	-$(RM) ./Drivers/BSP/Components/mx66uw1g45g/mx66uw1g45g.cyclo ./Drivers/BSP/Components/mx66uw1g45g/mx66uw1g45g.d ./Drivers/BSP/Components/mx66uw1g45g/mx66uw1g45g.o ./Drivers/BSP/Components/mx66uw1g45g/mx66uw1g45g.su

.PHONY: clean-Drivers-2f-BSP-2f-Components-2f-mx66uw1g45g

