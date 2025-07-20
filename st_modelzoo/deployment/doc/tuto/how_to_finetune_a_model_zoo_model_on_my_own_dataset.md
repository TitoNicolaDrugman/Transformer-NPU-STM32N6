# How can I finetune a ST Model Zoo model on my own dataset?

With ST Model Zoo, you can easily pick an already pretrained model and finetune it on your own dataset.

## Pick a pretrained model

A choice of model architectures pretrained on multiple datasets can be found [here](https://github.com/STMicroelectronics/stm32ai-modelzoo-services/tree/main/image_classification/pretrained_models).
Find the model you would like to based on it's input size and performances on various benchmarks. 

## Operation modes:

Depending on what you want to do, you can use the operation modes below:
- Training:
    - To simply train the model on my data and get as output the trained tensorflow model (.h5).
- Chain_tqe:
    - To train, quantize and evaluate the model in one go. You get as ouput both the train and quantized trained models (.h5 and .tflite)
- Chain_tqeb:
    - To train, quantize, evaluate and benchmark the quantized model in one go.

For any details regarding the parameters of the config file or the data augmentation, you can look here:
- [Training documentation](../../../src/training/README.md)
- [Quantization documentation](../../../src/quantization/README.md)
- [Benchmark documentation](../../../src/benchmarking/README.md)
- [Evaluation documentation](../../../src/evaluation/README.md)


## Finetune the model on my dataset

As usual, to retrain the model we edit the user_config.yaml and the stm32ai_main.py python script (both found in /src).
In this example, we retrain the mobilenet_v2 model with an input size of (224x224x3) pretrained on a large public dataset imagenet, with our data.
In our case, our dataset contrains butterflies species images: [Dataset](https://www.kaggle.com/datasets/gpiosenka/butterfly-images40-species)

The most important parts here are to define:
- The operation mode to training
- The data paths for training, validation and test
- Define which and how many classes you want to detect (ie the model output size)
- Choose a model, its pretrained weights and input size
- The other training parameters

```yaml
# user_config.yaml

general:
  project_name: my_project
  logs_dir: logs
  saved_models_dir: saved_models
  display_figures: True
  global_seed: 127
  gpu_memory_limit: 3

operation_mode: training

dataset:
  name: butterflies
  # Define the classes you want to detect, in this example, just the first 5
  # So my model output is of size 5
  class_names: ['ADONIS', 'AFRICAN GIANT SWALLOWTAIL', 'AMERICAN SNOOT', 'AN 88', 'APPOLLO'] 
  # define the paths for your training, validation and test data
  training_path: ./datasets/butterflies/train 
  validation_path: ./datasets/butterflies/valid
  test_path: ./datasets/butterflies/test      

# preprocessing to rescale and resize the data to the model input size define below
preprocessing:
  rescaling:
    scale: 1/127.5
    offset: -1
  resizing:
    aspect_ratio: fit
    interpolation: nearest
  color_mode: rgb

# optional
data_augmentation:
  random_contrast:
    factor: 0.4
  random_brightness:
    factor: 0.2
  random_flip:
    mode: horizontal_and_vertical
  random_translation:
    width_factor: 0.2
    height_factor: 0.2
  random_rotation:
    factor: 0.15
  random_zoom:
    width_factor: 0.25
    height_factor: 0.25

# training parameters
training:
  model:
    name: mobilenet # Select our model architecture
    version: v2 # we also have mobilenetv1
    alpha: 0.35
    pretrained_weights: imagenet # pretrain on imagenet dataset
    input_shape: (224, 224, 3) # input size
  # all the parameters below are standard in machine learning, you can look for them in google
  # they mostly depends on the topology of your model and will need a lot of testing
  batch_size: 64
  epochs: 200
  dropout: 0.3
  optimizer:
    Adam:
        learning_rate: 0.001
  callbacks:
    ReduceLROnPlateau:
        monitor: val_accuracy
        factor: 0.5
        patience: 10
    EarlyStopping:
        monitor: val_accuracy
        patience: 40

mlflow:
  uri: ./src/experiments_outputs/mlruns

hydra:
  run:
    dir: ./src/experiments_outputs/${now:%Y_%m_%d_%H_%M_%S}
  
```

For the Chain_tqe and Chain_tqeb operation modes, you need to edit the config file to add part related to the quantization and benchmark.
Look at the documentation linked above for more details.

You can also find examples of user_config.yaml [here](https://github.com/STMicroelectronics/stm32ai-modelzoo-services/tree/main/image_classification/src/config_file_examples)

## Run the script:

Edit the user_config.yaml then open a CMD (make sure to be in the UC folder). Finally, run the command:

```powershell
python stm32ai_main.py
```
You can also use any .yaml file using command below:
```powershell
python stm32ai_main.py --config-path=path_to_the_folder_of_the_yaml --config-name=name_of_your_yaml_file
```