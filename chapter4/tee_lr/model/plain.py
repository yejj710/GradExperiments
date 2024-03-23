import secretflow as sf
import time
import logging
import math
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import spu
from heu import phe
from numpy.random import RandomState
from secretflow.data.ndarray import FedNdarray
from secretflow.data.vertical.dataframe import VDataFrame
from secretflow.device import proxy
from secretflow.device.device.heu import HEU, HEUMoveConfig
from secretflow.device.device.pyu import PYU, PYUObject
from secretflow.device.device.type_traits import spu_fxp_precision
from secretflow.device.driver import reveal
from secretflow.security.aggregation.aggregator import Aggregator
from GradExperiments.chapter4.tee_lr.utils.data_loader import _load_breast_cancer
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score


def fit():
    pass

if __name__ == "__main__":
    sf.shutdown()
    sf.init(['alice','bob'], address="local")