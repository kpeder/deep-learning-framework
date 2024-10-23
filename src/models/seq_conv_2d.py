from contextlib2 import AbstractContextManager
from typing import Self

import keras


class SequentialConv2D(keras.Sequential, AbstractContextManager):
    '''
    Sequential model using regularized convolutional neural network.
    '''
    def __init__(self, input_shape: tuple, num_classes: int) -> Self:
        '''
        Implement the default model and return the instance.
        '''
        super().__init__([
            keras.layers.Input(shape=input_shape),
            keras.layers.Conv2D(64, kernel_size=(3,3), activation="relu"),
            keras.layers.Conv2D(64, kernel_size=(3,3), activation="relu"),
            keras.layers.MaxPooling2D(pool_size=(2,2)),
            keras.layers.Conv2D(128, kernel_size=(3,3), activation="relu"),
            keras.layers.Conv2D(128, kernel_size=(3,3), activation="relu"),
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(num_classes, activation="softmax")
            ])

    def __enter__(self) -> Self:
        '''
        Context Manager entry method.
        '''
        return self

    def __exit__(self, *args) -> bool:
        '''
        Context Manager exit method.
        '''
        return False
