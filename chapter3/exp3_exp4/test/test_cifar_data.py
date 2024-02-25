import sys
import secretflow as sf
sys.path.append("/home/yejj/GradExperiments/chapter3/")
from secretflow.utils.simulation.datasets import load_mnist
from utils.model import create_model
import numpy as np
import tensorflow as tf
from utils.data import load_cifar_data


if __name__ == "__main__":
    
    (x_train, y_train), (x_test, y_test) = load_cifar_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # input_shape = (32, 32, 3)
    # model = create_model(input_shape=input_shape, num_classes=10, name='lenet', dataset='cifar')

