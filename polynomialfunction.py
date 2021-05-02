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
substance = input("代測物質:")
substance = df[substance]


def normal(value):
    p = stats.pearsonr(value, substance)
    return abs(p[0])


def ln_normal(value):
    value = np.log(value)
    p = stats.pearsonr(value, substance)
    return abs(p[0])


def plus(n1, n2):
    value = n1+n2
    p = stats.pearsonr(value, substance)
    return abs(p[0])


def minus(n1, n2):
    value = n1-n2
    p = stats.pearsonr(value, substance)
    return abs(p[0])


def ln_plus(n1, n2):
    value = np.log(n1+n2)
    p = stats.pearsonr(value, substance)
    return abs(p[0])


def ln_minus(n1, n2):
    value = np.log(n1-n2+2)  # +2的原因是怕相減為負數無法進行對數運算
    p = stats.pearsonr(value, substance)
    return abs(p[0])


def divide(n1, n2):
    value = n1/n2
    p = stats.pearsonr(value, substance)
    return abs(p[0])


def ln_divide(n1, n2):
    value = np.log(n1/n2)
    p = stats.pearsonr(value, substance)
    return abs(p[0])

