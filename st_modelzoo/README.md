# Image classification STM32 model zoo


## Directory components:
* [datasets](datasets/README.md) placeholder for the image classification datasets.
* deployment ([STM32H7](deployment/README_STM32H7.md), [STM32N6](deployment/README_STM32N6.md)) contains the necessary files for the deployment service.
* [pretrained_models](pretrained_models/README.md) points on a collection of optimized pretrained models on different image classification datasets.
* [src](src/README.md) contains tools to train, evaluate, benchmark and quantize your model on your STM32 target.

## Quick & easy examples:
The `operation_mode` top-level attribute specifies the operations or the service you want to execute. This may be single operation or a set of chained operations.

You can refer to readme links below that provide typical examples of operation modes, and tutorials on specific services:

- [training, chain_tqe (train + quantize + evaluate), chain_tqeb](./src/training/README.md)
- [quantization, chain_eqe, chain_qb](./src/quantization/README.md)
- [evaluation, chain_eqeb](./src/evaluation/README.md)
- [benchmarking](./src/benchmarking/README.md)
- [prediction](./src/prediction/README.md)
- deployment, chain_qd ([STM32H7](./deployment/README_STM32H7.md), [STM32N6](./deployment/README_STM32N6.md))

All .yaml configuration examples are located in [config_file_examples](./src/config_file_examples/) folder.

The different values of the `operation_mode` attribute and the corresponding operations are described in the table below. In the names of the chain modes, 't' stands for training, 'e' for evaluation, 'q' for quantization, 'b' for benchmark and 'd' for deployment on an STM32 board.

| operation_mode attribute | Operations |
|:-------------------------|:-----------|
| `training`               | Train a model from the variety of classification models in the model zoo **(BYOD)** or your own model **(BYOM)** |
| `evaluation`             | Evaluate the accuracy of a float or quantized model on a test or validation dataset|
| `quantization`           | Quantize a float model |
| `prediction`             | Predict the classes some images belong to using a float or quantized model |
| `benchmarking`           | Benchmark a float or quantized model on an STM32 board |
| `deployment`             | Deploy a model on an STM32 board |
| `chain_tqeb`             | Sequentially: training, quantization of trained model, evaluation of quantized model, benchmarking of quantized model |
| `chain_tqe`              | Sequentially: training, quantization of trained model, evaluation of quantized model |
| `chain_eqe`              | Sequentially: evaluation of a float model,  quantization, evaluation of the quantized model |
| `chain_qb`               | Sequentially: quantization of a float model, benchmarking of quantized model |
| `chain_eqeb`             | Sequentially: evaluation of a float model,  quantization, evaluation of quantized model, benchmarking of quantized model |
| `chain_qd`               | Sequentially: quantization of a float model, deployment of quantized model |

There is no model_type on the image classification use case, but [here](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/master/image_classification/README.md) can be found the list of the available models.

## You don't know where to start? You feel lost?
Don't forget to follow our tuto below for a quick ramp up : 
* [How can I define and train my own model?](../image_classification/deployment/doc/tuto/how_to_define_and_train_my_own_model.md)
* [How can I fine tune a pretrained model on my own dataset?](../image_classification/deployment/doc/tuto/how_to_finetune_a_model_zoo_model_on_my_own_dataset.md)
* [How can I check the accuracy after quantization of my model?](../image_classification/deployment/doc/tuto/how_to_compare_the_accuracy_after_quantization_of_my_model.md)
* [How can I evaluate my model on STM32N6 target?](../image_classification/deployment/doc/tuto/how_to_evaluate_my_model_on_stm32n6_target.md)

Remember that minimalistic yaml files are available [here](./src/config_file_examples/) to play with specific services, and that all pre-trained models in the [STM32 model zoo](https://github.com/STMicroelectronics/stm32ai-modelzoo/) are provided with their configuration .yaml file used to generate them. These are very good starting points to start playing with!
