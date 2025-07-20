/* Includes ------------------------------------------------------------------*/

#include <stdio.h>
#include <string.h>

#include "app_config.h"
#include "npu_cache.h"
#include "mcu_cache.h"
#include "ll_aton_runtime.h"
#include "uart.h"
#include "md5.h"
#include "main.h"
#include "misc_toolbox.h"

#include "quantized_embedding.h"






LL_ATON_DECLARE_NAMED_NN_INSTANCE_AND_INTERFACE(Default) // Defines NN_Instance_Default and NN_Interface_Default with network.c info
/* Private variables ---------------------------------------------------------*/
static uint32_t t_init;
static uint32_t t_out;
/* Private function prototypes -----------------------------------------------*/
void time_in(void);
uint32_t time_out(void);
#ifdef  USE_FULL_ASSERT
  void assert_failed(uint8_t* file, uint32_t line);
#endif

#if defined(__GNUC__)
#define PUTCHAR_PROTOTYPE int __io_putchar(int ch)
#endif /* __ICCARM__ */
/**
  * @brief  Retargets the C library printf function to the USART.
  */
PUTCHAR_PROTOTYPE
{
  /* Place your implementation of putchar here */
  /* e.g. write a character to the USART1 and Loop until the end of transmission */
  //HAL_UART_Transmit(&huart1, (uint8_t *)&ch, 1, 0xFFFF);
  uart_write( (uint8_t *)&ch, 1);
  return ch;
}

void init_text_generation();
void generate_one_step(int8_t *buffer_in,uint32_t new_token);
uint32_t softmax_with_temperature(int8_t *logits_q, float temperature);
uint32_t random_choice(float* probs, uint32_t n);
void perplexity(void);
float perplexity_softmax_with_temperature(int8_t *logits_q, float temperature,uint32_t token_id);


/* Private functions ---------------------------------------------------------*/
void init_dwt()
{
    /* Enable Trace */
  CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;

  /* Reset Cycle Counter and Event Counters */
  ARM_PMU_CYCCNT_Reset();

  /* Enable Cycle Counter */
  ARM_PMU_CNTR_Enable(PMU_CNTENSET_CCNTR_ENABLE_Msk);

  /* Enable the PMU */
  ARM_PMU_Enable();
}

void time_in(void)
{
  ARM_PMU_CYCCNT_Reset();
  t_init = ARM_PMU_Get_CCNTR();
}

uint32_t time_out(void)
{
  t_out = ARM_PMU_Get_CCNTR();
  return (t_out - t_init);
}

void init_external_memories(void)
{
#if defined(USE_EXTERNAL_MEMORY_DEVICES) && USE_EXTERNAL_MEMORY_DEVICES == 1
  BSP_XSPI_NOR_Init_t Flash;
  
#if (NUCLEO_N6_CONFIG == 0)
  BSP_XSPI_RAM_Init(0);
  BSP_XSPI_RAM_EnableMemoryMappedMode(0);
  /* Configure the memory in octal DTR */
  Flash.InterfaceMode = MX66UW1G45G_OPI_MODE;
  Flash.TransferRate = MX66UW1G45G_DTR_TRANSFER;
#else
  Flash.InterfaceMode = MX25UM51245G_OPI_MODE;
  Flash.TransferRate = MX25UM51245G_DTR_TRANSFER;
#endif
  
  if(BSP_XSPI_NOR_Init(0, &Flash) != BSP_ERROR_NONE)
  {
        __BKPT(0);
  }
  BSP_XSPI_NOR_EnableMemoryMappedMode(0);
#endif 
}

#ifdef  USE_FULL_ASSERT

/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t* file, uint32_t line)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(file);
  UNUSED(line);

  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  printf("FAIL on file %s on line %d\r\n", file, line);
  __BKPT(0);
  /* Infinite loop */
  while (1)
  {
  }
}

#endif  /* USE_FULL_ASSERT */
/* Main function -------------------------------------------------------------*/
uint32_t duration_us;
uint32_t duration_dwt;
/* Flag signaling Epoch Controller has finished */

int main(void)
{
  float_t clock_Hz;
  uint8_t *buffer_in;
  uint8_t *buffer_out;
  uint32_t cpuclk;

  set_vector_table_addr();
  
  HAL_Init();

  system_init_post();

  set_mcu_cache_state(USE_MCU_ICACHE, USE_MCU_DCACHE);
     
  /* Configure the system clock */
#if VDDCORE_OVERDRIVE == 1
  upscale_vddcore_level();
  SystemClock_Config_HSI_overdrive();
#else
  SystemClock_Config_HSI_no_overdrive();
#endif
  
  /* Clear SLEEPDEEP bit of Cortex System Control Register */
  CLEAR_BIT(SCB->SCR, SCB_SCR_SLEEPDEEP_Msk);
  
  UART_Config();
  
  NPU_Config();

  init_external_memories();
  
  RISAF_Config();
  set_clk_sleep_mode();
  //uart_selftest_loop(); // Check printf redirection to UART forever
  
  // Get clock frequency to compute inference duration later on 
  // and init time measurement capabilities with PMU
  cpuclk =  HAL_RCC_GetCpuClockFreq();
  clock_Hz = (float_t) cpuclk;
  init_dwt();

  

#if 0
  perplexity();
  while(1);
#endif


  LL_ATON_RT_RetValues_t ll_aton_rt_ret = LL_ATON_RT_DONE;
  const EpochBlock_ItemTypeDef *eb_list = LL_ATON_EpochBlockItems_Default();
  
  /* Retreive the start address of the input and output buffer
  (reserved in the activation buffer) */
  const LL_Buffer_InfoTypeDef * ibuffersInfos = NN_Interface_Default.input_buffers_info();
  const LL_Buffer_InfoTypeDef * obuffersInfos = NN_Interface_Default.output_buffers_info();
  buffer_in = (uint8_t *)LL_Buffer_addr_start(&ibuffersInfos[0]);
  buffer_out = (uint8_t *)LL_Buffer_addr_start(&obuffersInfos[0]);
  

#if 0
  printf("emb 0\r\n");
  for(int i=0; i < 128; i++) printf("%d, ",embedding_table[0][i]);
  printf("\r\n");

  printf("emb 19999\r\n");
  for(int i=0; i < 128; i++) printf("%d, ",embedding_table[19999][i]);
  printf("\r\n");

  printf("emb 5\r\n");
  for(int i=0; i < 128; i++) printf("%d, ",embedding_table[5][i]);
  printf("\r\n");
  while(1);
#endif




  float temperature = 1.0;
  uint32_t new_token;
  while(1)
  {
	  uart_read((uint8_t*)&new_token,4,0);
	  init_text_generation();
	  for(int i=0;i<128;i++)
	  {
		time_in();
		/* ------------- */
		/* - Inference - */
		/* ------------- */


		/* Pre-process and fill the input buffer */
		generate_one_step((int8_t*)buffer_in,new_token);



		/* Perform the inference */
		LL_ATON_RT_RuntimeInit();
		LL_ATON_RT_Init_Network(&NN_Instance_Default);  // Initialize passed network instance object
		do
		{
		  /* Execute first/next step */
		  ll_aton_rt_ret = LL_ATON_RT_RunEpochBlock(&NN_Instance_Default);
		  /* Wait for next event */
		  if (ll_aton_rt_ret == LL_ATON_RT_WFE)
		  LL_ATON_OSAL_WFE();
		}
		while (ll_aton_rt_ret != LL_ATON_RT_DONE);
		LL_ATON_RT_DeInit_Network(&NN_Instance_Default); // De-initialize the network instance object
		LL_ATON_RT_RuntimeDeInit();                  // De-initialize runtime
		/* Post-process the output buffer */
		/* Invalidate the associated CPU cache region if requested */

		new_token = softmax_with_temperature((int8_t*)buffer_out,temperature);

		//_post_process(buffer_out);
		/* -------------------- */
		/* - End of Inference - */
		/* -------------------- */
		duration_dwt = time_out();
		duration_us = (uint32_t)(((float_t)duration_dwt * 1000000.0)/clock_Hz);
		//printf("Token %d, Inference: %d us (%d) cycles)\r\n",new_token,duration_us, duration_dwt);
		uart_write((uint8_t*)&new_token,4);

		// new change to print inference time
		uart_write((uint8_t*)&duration_us, 4);

	  }
 }
  
}







#define CTX_LEN 30
#define EMB_LEN 128
#define VOCAB_LEN 20000
static int8_t g_input_tensor[CTX_LEN][EMB_LEN];
static float g_logits[VOCAB_LEN];

void init_text_generation(void)
{
    memset(g_input_tensor, 0, CTX_LEN * EMB_LEN);
}

void generate_one_step(int8_t *buffer_in, uint32_t new_token)
{
    memcpy(g_input_tensor[0], g_input_tensor[1], (CTX_LEN-1) * EMB_LEN);
    memcpy(g_input_tensor[CTX_LEN-1], embedding_table[new_token], EMB_LEN);
    memcpy(buffer_in, g_input_tensor, CTX_LEN * EMB_LEN);
}

// Optimized softmax with temperature and random choice, no g_probs buffer
uint32_t softmax_with_temperature(int8_t *logits_q, float temperature)
{
    const float out_scale = 0.07027540F;
    const float out_zp = 14.0F;
    const int32_t pivot = (CTX_LEN-1) * VOCAB_LEN;
    float logit;
    float max_logit = -INFINITY;

    // Step 1: dequantize + apply temperature + max detection
    for (int i = 0; i < VOCAB_LEN; i++)
    {
        logit = (float) logits_q[pivot + i];
        logit -= out_zp;
        logit *= out_scale;
        logit /= temperature;
        if (logit > max_logit) max_logit = logit;
        g_logits[i] = logit;
    }

    // Step 2: Compute exp and sum for normalization
    float sum = 0.0f;
    for (int i = 0; i < VOCAB_LEN; ++i)
    {
        sum += expf(g_logits[i] - max_logit);
    }

    // Step 3: Sample directly without allocating g_probs
    float r = (float)rand() / (RAND_MAX + 1.0f);
    float cumulative = 0.0f;
    for (int i = 0; i < VOCAB_LEN; ++i)
    {
        float prob = expf(g_logits[i] - max_logit) / sum;
        cumulative += prob;
        if (r < cumulative) return i;
    }
    // In case of rounding errors, return last index
    return VOCAB_LEN - 1;
}



float perplexity_softmax_with_temperature(int8_t *logits_q, float temperature,uint32_t token_id)
{
    const float out_scale = 0.07027540F;
    const float out_zp = 14.0F;
    const int32_t pivot = (CTX_LEN-1) * VOCAB_LEN;
    float logit;
    float max_logit = -256.0F;
    float token_id_prob;

    // Step 1: dequantize + apply temperature + max detection
    for (int i = 0; i < VOCAB_LEN; i++) 
    {
        logit = (float) logits_q[pivot + i];
        logit -= out_zp;
        logit *= out_scale;
        logit /= temperature;
        if (logit > max_logit) max_logit = logit;
        g_logits[i] = logit;
    }

    // Step 2: Compute exp and sum for normalization
    double sum = 0.0f;
    for (int i = 0; i < VOCAB_LEN; ++i) 
    {
        sum += exp(g_logits[i] - max_logit);
    }

    // Step 3: calculate prob for token_id
    token_id_prob = exp(g_logits[token_id] - max_logit) / sum;
    return log( token_id_prob ) * -1.0;
}



// Perplexity
#define PERPLEXITY_TXT_LEN 149
const uint16_t g_perplexity_txt[PERPLEXITY_TXT_LEN] = {1, 130, 275, 238, 1, 5, 12612, 312, 52, 11524, 891, 8, 46, 1, 3, 77, 6497, 2341, 71, 81, 64, 55, 4710, 51, 130, 1076, 26, 3896, 1429, 2460, 46, 56, 13142, 11821, 3, 419, 13531, 116, 11202, 4, 46, 5881, 46, 1, 143, 15, 1, 21, 75, 14005, 1, 14, 9967, 4, 13497, 3737, 767, 1, 10, 47, 21, 290, 5, 16, 1449, 143, 8382, 1876, 1165, 20, 188, 7, 10942, 3, 1, 20, 18, 1251, 36, 2, 2489, 4, 5822, 1, 66, 1105, 2170, 46, 1, 358, 6, 1225, 1063, 10, 1410, 1, 8, 4422, 26, 7, 140, 11415, 370, 23, 1420, 3858, 1, 3697, 26, 898, 1, 2, 1656, 8, 986, 3416, 1, 358, 6, 1083, 10, 162, 20, 5950, 11296, 1, 3, 7459, 168, 28, 2, 5951, 1, 1, 8, 46, 2456, 15192, 34, 2766, 1, 123, 562, 46, 570, 490, 87, 4, 1};

void perplexity(void)
{
  uint8_t *buffer_in;
  uint8_t *buffer_out;
  double nll,pll;
  float temperature=1.0;


	LL_ATON_RT_RetValues_t ll_aton_rt_ret = LL_ATON_RT_DONE;
	const LL_Buffer_InfoTypeDef * ibuffersInfos = NN_Interface_Default.input_buffers_info();
	const LL_Buffer_InfoTypeDef * obuffersInfos = NN_Interface_Default.output_buffers_info();
	buffer_in = (uint8_t *)LL_Buffer_addr_start(&ibuffersInfos[0]);
	buffer_out = (uint8_t *)LL_Buffer_addr_start(&obuffersInfos[0]);

	init_text_generation();
	// add first 30 tokens -> embeddings
	for(int i=0;i<CTX_LEN;i++)
	{
		memcpy(g_input_tensor[i], embedding_table[g_perplexity_txt[i]], EMB_LEN);
	}
	memcpy(buffer_in, g_input_tensor, CTX_LEN * EMB_LEN);

#if 0
	int32_t checksum=0;
	for(int ii=0; ii<CTX_LEN * EMB_LEN; ii++) checksum += (int8_t)buffer_in[ii];
	printf("checksum %d\r\n",checksum);
#endif

	// calculate perplexity
	int count = 0;
	nll = 0.0;
	pll = 0.0;
	for(int i=CTX_LEN;i<PERPLEXITY_TXT_LEN;i++)
	{
		// forward
		LL_ATON_RT_RuntimeInit();
		LL_ATON_RT_Init_Network(&NN_Instance_Default);  // Initialize passed network instance object
		do
		{
		  /* Execute first/next step */
		  ll_aton_rt_ret = LL_ATON_RT_RunEpochBlock(&NN_Instance_Default);
		  /* Wait for next event */
		  if (ll_aton_rt_ret == LL_ATON_RT_WFE)
		  LL_ATON_OSAL_WFE();
		}
		while (ll_aton_rt_ret != LL_ATON_RT_DONE);
		LL_ATON_RT_DeInit_Network(&NN_Instance_Default); // De-initialize the network instance object
		LL_ATON_RT_RuntimeDeInit();                  // De-initialize runtime

		nll += perplexity_softmax_with_temperature((int8_t*)buffer_out,temperature,(uint32_t)g_perplexity_txt[i]);
		//printf("Step %f\r\n",nll);
		count++;
		generate_one_step((int8_t*)buffer_in,(uint32_t)g_perplexity_txt[i]);
#if 0
		checksum=0;
		for(int ii=0; ii<CTX_LEN * EMB_LEN; ii++) checksum += (int8_t)buffer_in[ii];
		printf("checksum %d\r\n",checksum);
#endif
	}
	pll = exp(nll / count);
	printf("Perplexity %f\r\n", pll);

}
