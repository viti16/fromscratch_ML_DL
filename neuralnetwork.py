import numpy as np
import pandas as pd

import sklearn
from sklearn.datasets import fetch_openml


x = fetch_openml('mnist_784', version=1)["data"]
y = fetch_openml('mnist_784', version=1)["target"]



print(234)