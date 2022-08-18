import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import numpy as np
import numpy
import matplotlib
import matplotlib.pyplot as plt
import random
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec
import torch

from torch.autograd import Variable
import math
import pdb

import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression

# from adapt.utils import make_regression_da
import random
import torch
import torch
import wandb
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import confusion_matrix


import sklearn.datasets as dt
from torch.utils.data import Dataset


from PIL import Image
import xml.etree.ElementTree as ET
from glob import glob
from torchvision import transforms


from sklearn.preprocessing import StandardScaler


import torchvision.transforms as T
from typing import Optional, Callable, Tuple, Any, List
import torchvision.datasets as torchdatasets
from torch.utils.data import ConcatDataset
import argparse
import timm


sys.path.append("C:\OneDrive - Technion\Phd\Sandbox\SDA\Code\TLIB\\")
from TLIB.common.modules.regressor import Regressor as RegressorBase
from TLIB.common.modules.classifier import Classifier as ClassifierBase


# import common.vision.datasets as datasets
# from common.modules.regressor import Regressor as RegressorBase


