################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (13.3.rel1)
################################################################################

-include ../makefile.init

RM := rm -rf

# All of the sources participating in the build are defined here
-include sources.mk
-include Src/subdir.mk
-include Drivers/STM32N6xx_HAL_Driver/Src/subdir.mk
-include Drivers/CMSIS/Device/ST/STM32N6xx/Source/Templates/subdir.mk
-include Drivers/BSP/STM32N6570-DK/subdir.mk
-include Drivers/BSP/Components/mx66uw1g45g/subdir.mk
-include Drivers/BSP/Components/aps256xx/subdir.mk
-include AI/atonn/models/subdir.mk
-include AI/atonn/ll_aton/subdir.mk
-include subdir.mk
-include objects.mk

ifneq ($(MAKECMDGOALS),clean)
ifneq ($(strip $(S_DEPS)),)
-include $(S_DEPS)
endif
ifneq ($(strip $(S_UPPER_DEPS)),)
-include $(S_UPPER_DEPS)
endif
ifneq ($(strip $(C_DEPS)),)
-include $(C_DEPS)
endif
endif

-include ../makefile.defs

OPTIONAL_TOOL_DEPS := \
$(wildcard ../makefile.defs) \
$(wildcard ../makefile.init) \
$(wildcard ../makefile.targets) \


BUILD_ARTIFACT_NAME := hello_world
BUILD_ARTIFACT_EXTENSION := elf
BUILD_ARTIFACT_PREFIX :=
BUILD_ARTIFACT := $(BUILD_ARTIFACT_PREFIX)$(BUILD_ARTIFACT_NAME)$(if $(BUILD_ARTIFACT_EXTENSION),.$(BUILD_ARTIFACT_EXTENSION),)

# Add inputs and outputs from these tool invocations to the build variables 
EXECUTABLES += \
hello_world.elf \

MAP_FILES += \
hello_world.map \

SIZE_OUTPUT += \
default.size.stdout \

OBJDUMP_LIST += \
hello_world.list \

OBJCOPY_BIN += \
hello_world.bin \


# All Target
all: main-build

# Main-build Target
main-build: hello_world.elf secondary-outputs

# Tool invocations
hello_world.elf hello_world.map: $(OBJS) $(USER_OBJS) C:/Users/drugm/stm32ai-modelzoo-services/dd2_text_gen_stm32n6v2/text_gen_stm32n6/Projects/STM32N6570-DK/Applications/hello_world/cubeIDE/STM32N657xx.ld makefile objects.list $(OPTIONAL_TOOL_DEPS)
	arm-none-eabi-gcc -o "hello_world.elf" @"objects.list" $(USER_OBJS) $(LIBS) -mcpu=cortex-m55 -T"C:/Users/drugm/stm32ai-modelzoo-services/dd2_text_gen_stm32n6v2/text_gen_stm32n6/Projects/STM32N6570-DK/Applications/hello_world/cubeIDE/STM32N657xx.ld" --specs=nosys.specs -Wl,-Map="hello_world.map" -Wl,--gc-sections -static -Wl,--cmse-implib -Wl,--out-implib=./secure_nsclib.o -L"C:/Users/drugm/stm32ai-modelzoo-services/dd2_text_gen_stm32n6v2/text_gen_stm32n6/Projects/STM32N6570-DK/Applications/hello_world/cubeIDE/../../../../../Middlewares/ST/AI/Lib/GCC/ARMCortexM55" --specs=nano.specs -mfpu=fpv5-d16 -mfloat-abi=hard -mthumb -u _printf_float -Wl,--start-group -lc -lm -Wl,--end-group
	@echo 'Finished building target: $@'
	@echo ' '

default.size.stdout: $(EXECUTABLES) makefile objects.list $(OPTIONAL_TOOL_DEPS)
	arm-none-eabi-size  $(EXECUTABLES)
	@echo 'Finished building: $@'
	@echo ' '

hello_world.list: $(EXECUTABLES) makefile objects.list $(OPTIONAL_TOOL_DEPS)
	arm-none-eabi-objdump -h -S $(EXECUTABLES) > "hello_world.list"
	@echo 'Finished building: $@'
	@echo ' '

hello_world.bin: $(EXECUTABLES) makefile objects.list $(OPTIONAL_TOOL_DEPS)
	arm-none-eabi-objcopy  -O binary $(EXECUTABLES) "hello_world.bin"
	@echo 'Finished building: $@'
	@echo ' '

# Other Targets
clean:
	-$(RM) default.size.stdout hello_world.bin hello_world.elf hello_world.list hello_world.map
	-@echo ' '

secondary-outputs: $(SIZE_OUTPUT) $(OBJDUMP_LIST) $(OBJCOPY_BIN)

fail-specified-linker-script-missing:
	@echo 'Error: Cannot find the specified linker script. Check the linker settings in the build configuration.'
	@exit 2

warn-no-linker-script-specified:
	@echo 'Warning: No linker script specified. Check the linker settings in the build configuration.'

.PHONY: all clean dependents main-build fail-specified-linker-script-missing warn-no-linker-script-specified

-include ../makefile.targets
