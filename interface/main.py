import INTERFACE
import time
import array
import numpy as np
import sys


def createAndTrain(training_parameters):
    print(f"training_parameters={training_parameters}")
    # INTERFACE.createAndTrain(training_parameters)


if (len(sys.argv) < 3):
    print(f"Usage: {sys.argv[0]} dataset_path output_path")
else:
    training_parameters =  {
        "dataset_path": sys.argv[1],
        "output_path" : sys.argv[2],
        "rescale_size": 32,
        "preprocess_transformations": [], 
        "postprocess_transformations": []
    }
    createAndTrain(training_parameters)