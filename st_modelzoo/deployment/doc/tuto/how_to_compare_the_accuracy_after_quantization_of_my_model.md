# How to compare the accuracy after quantization of my model?

The quantization process optimizes the model for efficient deployment on embedded devices by reducing its memory usage (Flash/RAM) and accelerating its inference time, with minimal degradation in model accuracy. With ST Model Zoo, you can easily check the accuracy of your model, quantize your model and compare this accuracy after quantization. You can also simply do one of these actions alone.

## Operation modes:

Depending on what you want to do, you can use the operation modes below:

- Evaluate:
    - To evaluate a model, quantized or not (.h5, .tflite or QDQ onnx)
- Chain_eqe:
    - To evaluate a model, quantize it and evaluate it again after quantization for comparison.
- Chain_eqeb:
    - To also add a benchmark of the quantized model.

For any details regarding the parameters of the config file, you can look here:
- [Evaluation documentation](../../../src/evaluation/README.md)
- [Quantization documentation](../../../src/quantization/README.md)
- [Benchmark documentation](../../../src/benchmarking/README.md)


## User_config.yaml:

The way ST Model Zoo works is that you edit the user_config.yaml available for each use case and run the stm32ai_main.py python script. 

Here is an example where we evaluate a .h5 model before quantizing it, evaluate it again for comparison:

The most important parts to define are:
- The model path
- The operation mode
- The dataset for quantization and test
- The preprocessing (usually the same used in training)

```yaml
# user_config.yaml

general:
   model_path: ../../stm32ai-modelzoo/image_classification/mobilenetv1/ST_pretrainedmodel_public_dataset/food-101/mobilenet_v1_0.5_224_fft/mobilenet_v1_0.5_224_fft.h5
   
operation_mode: chain_eqe

dataset:
   quantization_path: <quantization-set-root-directory> # you can use your training dataset
   quantization_split: 0.05
   test_path: <test-set-root-directory>  

preprocessing:
   rescaling:
      scale: 1/127.5
      offset: -1
   resizing:
      aspect_ratio: fit
      interpolation: bilinear
   color_mode: rgb

quantization:
   quantizer: TFlite_converter
   quantization_type: PTQ
   quantization_input_type: uint8
   quantization_output_type: float
   export_dir: quantized_models

mlflow:
   uri: ./src/experiments_outputs/mlruns

hydra:
   run:
      dir: ./src/experiments_outputs/${now:%Y_%m_%d_%H_%M_%S}
```
When evaluating the model, it is highly recommended to use real data for the quantization.

## Run the script:

Edit the user_config.yaml then open a CMD (make sure to be in the UC folder). Finally, run the command:

```powershell
python stm32ai_main.py
```
You can also use any .yaml file using command below:
```powershell
python stm32ai_main.py --config-path=path_to_the_folder_of_the_yaml --config-name=name_of_your_yaml_file
```

## Local benchmarking:

To make the benchmark locally instead of using the ST Edge AI Development Cloud you need to add the path for path_to_stedgeai and to set on_cloud to false in the yaml.

To download the tools:
- [ST Edge AI](https://www.st.com/en/embedded-software/x-cube-ai.html)
- [STM32CubeIDE](https://www.st.com/en/development-tools/stm32cubeide.html)


