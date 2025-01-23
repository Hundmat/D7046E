# Importing packages
import torch
print('torch imported')
import torch.nn as nn
print('torch.nn imported')
import torch.nn.functional as F
print('torch.nn.functional imported')
from torch.nn.functional import one_hot as to_onehot
print('one_hot imported as to_onehot')
import torch.optim as optim
print('torch.optim imported')
from torch.utils.data import DataLoader
print('torch.utils.data.DataLoader imported')
import torchvision
print('torchvision imported')
from torchvision import datasets, transforms
print('torchvision.datasets and torchvision.transforms imported')
import matplotlib.pyplot as plt
print('matplotlib.pyplot imported')
import numpy
print('numpy imported')
import copy
print('copy imported')

from sklearn.linear_model import LinearRegression, SGDClassifier
print('sklearn.linear_model imported')
from sklearn.tree import DecisionTreeClassifier
print('sklearn.tree imported')
from sklearn.metrics import accuracy_score
print('sklearn.metrics imported')
from sklearn.preprocessing import StandardScaler
print('sklearn.preprocessing imported')

print('The installation seems to be working!')