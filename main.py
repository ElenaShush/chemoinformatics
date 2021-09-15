# This is a script with examples of using libraries:
# numpy, pandas, keras, pytorch and tensorflow.


import numpy as np
import pandas as pd
import keras
import tensorflow as tf
from tensorflow_example import test_tensorflow


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


#
def test_numpy():
    A = np.array([[1, 2, 3], [4, 5, 6]])
    B = np.ones_like(A)
    # print(B)
    return "numpy is ok"


def test_pandas():
    series = pd.Series([5, 6, -5, 8, 2, -1], index=['a', 'b', 'c', 'd', 'e', 'f'])
    series.index.name = "Letters"
    # print(series[series < 0]*2)
    # print(series)
    return "pandas is ok"


def test_keras():
    return 0


def test_pytorch():
    return 0


if __name__ == '__main__':
    test_numpy()
    test_pandas()
    test_tensorflow()
    test_keras()
    print_hi('All libraries is ok')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
