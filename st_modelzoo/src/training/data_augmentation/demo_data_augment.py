# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022-2023 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import os
import sys
import argparse
from munch import DefaultMunch
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from src.utils import get_config
from src.preprocessing import preprocess
from common.data_augmentation import random_color, random_affine, random_erasing, random_misc



def display_images_side_by_side(image, image_aug, grayscale=None, legend=None):
    """
    This function displays the original and augmented images side by side.

    Args:
        images (Tuple): original images.
        images_aug (Tuple): corresponding augmented images.
    
    Returns:
        None
    """

    # Calculate the dimensions of the displayed images
    image_width, image_height = np.shape(image)[:2]
    display_size = 9
    if image_width >= image_height:
        x_size = display_size
        y_size = round((image_width / image_height) * display_size)
    else:
        y_size = display_size
        x_size = round((image_height / image_width) * display_size)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(x_size, y_size))

    if grayscale:
        ax1.imshow(image, cmap='gray')
    else:
        ax1.imshow(image)
    ax1.title.set_text("original")

    if grayscale:
        ax2.imshow(image_aug, cmap='gray')
    else:
        ax2.imshow(image_aug)
    ax2.title.set_text(legend)

    plt.show()
    plt.close()
    

def augment_images(images, fn_name=None):

    if fn_name == "random_contrast":
        return random_color.random_contrast(images, factor=0.7)
        
    elif fn_name == "random_brightness":
        return random_color.random_brightness(images, factor=0.4)

    elif fn_name == "random_gamma":
        return random_color.random_gamma(images, gamma=(0.2, 2.0))

    elif fn_name == "random_hue":
        return random_color.random_hue(images, delta=0.1)

    elif fn_name == "random_saturation":
        return random_color.random_saturation(images, delta=0.2)

    elif fn_name == "random_value":           
        return random_color.random_value(images, delta=0.2)        
        
    elif fn_name == "random_hsv":
        return random_color.random_hsv(
                        images, hue_delta=0.1, saturation_delta=0.2, value_delta=0.2)

    elif fn_name == "random_rgb_to_hsv":
        return random_color.random_rgb_to_hsv(images, change_rate=1.0)

    elif fn_name == "random_rgb_to_grayscale":        
        return random_color.random_rgb_to_grayscale(images, change_rate=1.0)

    elif fn_name == "random_sharpness":
        return random_color.random_sharpness(images, factor=(1.0, 4.0))

    elif fn_name == "random_posterize":
        return random_color.random_posterize(images, bits=(1, 8))

    elif fn_name == "random_invert":
        return random_color.random_invert(images, change_rate=1.0)

    elif fn_name == "random_solarize":
        return random_color.random_solarize(images, change_rate=1.0)

    elif fn_name == "random_autocontrast":
        return random_color.random_autocontrast(images, change_rate=1.0)

    elif fn_name == "random_blur":
        return random_misc.random_blur(images, filter_size=(2, 4))

    elif fn_name == "random_gaussian_noise":
        return random_misc.random_gaussian_noise(images, stddev=(0.02, 0.1))

    elif fn_name == "random_jpeg_quality":
        return random_misc.random_jpeg_quality(images, jpeg_quality=(20, 100))

    elif fn_name == "random_crop":
        return random_misc.random_crop(images, change_rate=1.0)

    elif fn_name == "random_flip":
        return random_affine.random_flip(images, mode="horizontal_and_vertical", change_rate=1.0)

    elif fn_name == "random_translation":
        return random_affine.random_translation(images, width_factor=0.2, height_factor=0.2)

    elif fn_name == "random_rotation":
        return random_affine.random_rotation(images, factor=0.075)

    elif fn_name == "random_shear":
        return random_affine.random_shear(images, factor=0.075, axis='xy')

    elif fn_name == "random_shear_x":
        return random_affine.random_shear(images, factor=0.075, axis='x')

    elif fn_name == "random_shear_y":
        return random_affine.random_shear(images, factor=0.075, axis='y')

    elif fn_name == "random_zoom":
        return random_affine.random_zoom(images, width_factor=0.3)

    elif fn_name == "random_rectangle_erasing":
        return random_erasing.random_rectangle_erasing(images, nrec=(0, 4))


def demo_data_augmentation(dataset_path, grayscale=None, num_images=None):
    """
    Samples a batch of images, applies to them the data augmentation 
    functions specified in the YAML configuration file, and displays
    side by side the original images and augmented images.
    """
    
    function_names = [
        "random_contrast", "random_brightness", "random_gamma", "random_hue",
        "random_saturation", "random_value", "random_hsv", "random_rgb_to_hsv",
        "random_rgb_to_grayscale", "random_sharpness", "random_posterize", 
        "random_invert", "random_solarize", "random_autocontrast", "random_blur",
        "random_gaussian_noise", "random_jpeg_quality", "random_crop", "random_flip", 
        "random_translation", "random_rotation", "random_shear", "random_shear_x",
        "random_shear_y", "random_zoom", "random_rectangle_erasing"
    ]

    color_only_functions = [
        "random_hue", "random_saturation", "random_value", "random_hsv",
        "random_rgb_to_hsv", "random_rgb_to_grayscale", "random_autocontrast"
    ]
    
    # If grayscale was requested, remove the functions
    # that are only applicable to color images from
    # the list of function names.
    if grayscale:
        for fn in color_only_functions:
            function_names.remove(fn)

    # Get the class names
    class_names = [p for p in os.listdir(dataset_path) 
                      if os.path.isdir(os.path.join(dataset_path, p))]

    # Create a configuration dictionary with the
    # information needed to create the data loader
    scale = 1./255
    offset = 0
    cfg = DefaultMunch.fromDict({
                "general": { "model_path": None },
                "operation_mode": "training",
                "dataset": {
                    "training_path": dataset_path,
                    "class_names": class_names,
                    "seed": None
                },
                "preprocessing": {
                    "rescaling": { "scale": scale, "offset": offset  },
                    "resizing": { "interpolation": "bilinear", "aspect_ratio": "fit" },
                    "color_mode": "grayscale" if grayscale else "rgb",
                },
                "training": {
                    "model": { "input_shape": (224, 224, 3) },
                    "batch_size": num_images,
                    "resizing": { "interpolation": "bilinear", "aspect_ratio": "fit" },
                    "color_mode": "grayscale" if grayscale else "rgb",
                }
          })
 
    # Create a data loader to get examples from the training set
    print("Dataset:", cfg.dataset.training_path)    
    data_loader, _, _, _ = preprocess(cfg)

    print("Demonstrating data augmentation functions:")
    for fn in function_names:
        print("  " + fn)
    
    for i, data in enumerate(data_loader):
        images, _ = data
        batch_size = tf.shape(images)[0]
        
        # Rescale the images
        images = scale * tf.cast(images, dtype=tf.float32) + offset
        
        images_aug = augment_images(images, fn_name=function_names[i])

        # Plot the images and their groundtruth labels
        for k in range(batch_size):
            display_images_side_by_side(images[k], images_aug[k], grayscale=grayscale, legend=function_names[i])

        # Stop when all the data augmentation functions have been demo'ed
        if i == len(function_names) - 1:
            exit()


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='', required=True,
                        help='path to the dataset to sample')
    parser.add_argument('--grayscale', action='store_true',
                        help='demo data augmentation functions on grayscale images')
    parser.add_argument('--num_images', type=int, default=4,
                        help='number of images to display for each data augmentation function (default: 4)')
    
    args = parser.parse_args()

    if not os.path.isdir(args.dataset_path):
        raise ValueError(f"\nCould not find dataset directory: {args.dataset_path}")
    
    demo_data_augmentation(args.dataset_path, grayscale=args.grayscale, num_images=args.num_images)

if __name__ == '__main__':
    main()
