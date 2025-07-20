# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022 STMicroelectronics.
#  * All rights reserved.
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

#from .evaluate import evaluate, evaluate_h5_model
from .evaluate import (
    evaluate,
    generate_text_h5 as evaluate_h5_model,
)

__all__ = [
    "evaluate",
    "evaluate_h5_model",
]
