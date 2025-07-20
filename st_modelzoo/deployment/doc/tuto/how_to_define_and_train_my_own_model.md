# How can I define and train my own model with ST Model Zoo?

With ST Model Zoo, you can easily define and train your own TensorFlow neural network model.

## Define my model

First, create your own model in /src/models/custom_model.py for it to be automatically used with the model zoo training script.
Open the python file and copy your model topology inside, here is the default example model:

```python
from typing import Tuple, Optional
import tensorflow as tf
from tensorflow.keras import layers

def get_custom_model(num_classes: int = None, input_shape: Tuple[int, int, int] = None,
                     dropout: Optional[float] = None) -> tf.keras.Model:
    """
    Creates a custom image classification model with the given number of classes and input shape.

    Args:
        num_classes (int): Number of classes in the classification task.
        input_shape (Tuple[int, int, int]): Shape of the input image.
        dropout (Optional[float]): Dropout rate to be applied to the model.

    Returns:
        keras.Model: Custom image classification model.
    """
    # Define the input layer
    inputs = tf.keras.Input(shape=input_shape)

    # Define the feature extraction layers
    x = layers.Conv2D(16, (3, 3), strides=(1, 1), padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D()(x)

    # Define the classification layers
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    if dropout:
        x = tf.keras.layers.Dropout(dropout)(x)
    if num_classes > 2:
        outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    else:
        outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    # Define and return the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="custom_model")
    return model
```

The model must be created inside the function get_custom_model. The input size and number of classes are then define in the user_config.yaml as below.


## Training my model

To train the model, we then edit the user_config.yaml and run the training using the python script stm32ai_main.py.
For this example, we used this dataset which contains butterflies images for classification [Dataset](https://www.kaggle.com/datasets/gpiosenka/butterfly-images40-species)

### Operation modes:

Depending on what you want to do, you can use the operation modes below:
- Training:
    - To simply train the model and get as output the trained tensorflow model (.h5).
- Chain_tqe:
    - To train, quantize and evaluate the model in one go. You get as ouput both the train and quantized trained models (.h5 and .tflite)
- Chain_tqeb:
    - To train, quantize, evaluate and benchmark the quantized model in one go.

For any details regarding the parameters of the config file or the data augmentation, you can look here:

- [Training documentation](../../../src/training/README.md)
- [Quantization documentation](../../../src/quantization/README.md)
- [Benchmark documentation](../../../src/benchmarking/README.md)
- [Evaluation documentation](../../../src/evaluation/README.md)


### Benchmarking Configuration example:

The most important parts here are to define:
- The operation mode to training
- The data paths for training, validation and test
- Define which and how many classes you want to detect (ie the model output size)
- The model name to custom for model zoo to load the model in custom_model.py
- The input_shape and other training parameters

```yaml
# user_config.yaml

general:
  project_name: Butterflies_example
  logs_dir: logs
  saved_models_dir: saved_models
  display_figures: True

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
   rescaling: {scale : 1/127.5, offset : -1}
   resizing: {interpolation: nearest, aspect_ratio: "fit"}
   color_mode: rgb # images in color, 3 channels.

training:
  model:
    # put custom for model zoo to look for your model define earlier
    name: custom
    input_shape: (224, 224, 3) # images of size 224x224 in color (3 channels)
  # all the parameters below are standard in machine learning, you can look for them in google
  # they mostly depends on the topology of your model and will need a lot of testing
  batch_size: 64
  epochs: 400
  dropout: 0.3
  optimizer: 
      Adam: {learning_rate: 0.001}
  # all tensorflow callbacks are compatible with ST Model Zoo
  callbacks:
      ReduceLROnPlateau:
        monitor: val_accuracy
        factor: 0.5
        patience: 10
      EarlyStopping:
        monitor: val_accuracy
        patience: 60

mlflow:
  uri: ./src/experiments_outputs/mlruns

hydra:
  run:
    dir: ./src/experiments_outputs/${now:%Y_%m_%d_%H_%M_%S}
  
```
For the Chain_tqe and Chain_tqeb operation modes, you need to edit the config file to add part related to the quantization and benchmark. Look at the documentation linked above for more details.

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