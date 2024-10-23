from config import Config
from models.seq_conv_2d import SequentialConv2D
import os


conf = Config()
conf.configure()

os.environ["KERAS_BACKEND"] = conf.configuration["keras"]["backend"]


def test_class_instantiation():
    with SequentialConv2D(input_shape=(28, 28, 1), num_classes=10) as model:
        print(model.input_shape)


def test_model_summarization():
    with SequentialConv2D(input_shape=(28, 28, 1), num_classes=10) as model:
        model.summary()
