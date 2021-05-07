# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 13:34:06 2021

@author: AMD_PC
"""
from scipy.stats import stats
import pandas as pd
import numpy as np
df = pd.read_excel('data.xlsx', index_col=False)
# 種類跟數量


def value_normal(value):
    return value


def value_ln_normal(value):
    value = np.log(value)
    return value


def value_plus(n1, n2):
    value = n1+n2
    return value


def value_minus(n1, n2):
    value = n1-n2
    return value


def value_ln_plus(n1, n2):
    value = np.log(n1+n2)
    return value


def value_ln_minus(n1, n2):
    value = np.log(n1-n2+2)  # +2的原因是怕相減為負數無法進行對數運算
    return value


def value_divide(n1, n2):
    value = n1/n2
    return value


def value_ln_divide(n1, n2):
    value = np.log(n1/n2)
    return value



