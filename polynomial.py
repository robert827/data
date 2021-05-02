# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 13:49:07 2021

@author: AMD_PC
"""
from polynomialfunction import normal, ln_normal, plus, minus, ln_plus, ln_minus, divide, ln_divide
import numpy as np
import pandas as pd
from scipy.stats import stats
number = int(input("特徵數量(整數):"))
df = pd.read_excel('data.xlsx', index_col=False)
b = df['blue']
g = df['green']
r = df['red']
edge = df['rededge']
n = df['nir']


data = pd.Series({
    'highest': 1,
    'blue': normal(b),
    'green': normal(g),
    'red': normal(r),
    'rededge': normal(edge),
    'nir': normal(n),
    'ln_blue': ln_normal(b),
    'ln_green': ln_normal(g),
    'ln_red': ln_normal(r),
    'ln_rededge': ln_normal(edge),
    'ln_nir': ln_normal(n),
    'plus_b_g': plus(b, g),
    'plus_b_r': plus(b, r),
    'plus_b_edge': plus(b, edge),
    'plus_b_nir': plus(b, n),
    'plus_g_r': plus(g, r),
    'plus_g_edge': plus(g, edge),
    'plus_g_nir': plus(g, n),
    'plus_r_edge': plus(r, edge),
    'plus_r_nir': plus(r, n),
    'plus_edge_nir': plus(edge, n),
    'minus_b_g': minus(b, g),
    'minus_b_r': minus(b, r),
    'minus_b_edge': minus(b, edge),
    'minus_b_nir': minus(b, n),
    'minus_g_r': minus(g, r),
    'minus_g_edge': minus(g, edge),
    'minus_g_nir': minus(g, n),
    'minus_r_edge': minus(r, edge),
    'minus_r_nir': minus(r, n),
    'minus_edge_nir': minus(edge, n),
    'ln_plus_b_g': ln_plus(b, g),
    'ln_plus_b_r': ln_plus(b, r),
    'ln_plus_b_edge': ln_plus(b, edge),
    'ln_plus_b_nir': ln_plus(b, n),
    'ln_plus_g_r': ln_plus(g, r),
    'ln_plus_g_edge': ln_plus(g, edge),
    'ln_plus_g_nir': ln_plus(g, n),
    'ln_plus_r_edge': ln_plus(r, edge),
    'ln_plus_r_nir': ln_plus(r, n),
    'ln_plus_edge_nir': ln_plus(edge, n),
    'ln_minus_b_g': ln_minus(b, g),
    'ln_minus_b_r': ln_minus(b, r),
    'ln_minus_b_edge': ln_minus(b, edge),
    'ln_minus_b_nir': ln_minus(b, n),
    'ln_minus_g_r': ln_minus(g, r),
    'ln_minus_g_edge': ln_minus(g, edge),
    'ln_minus_g_nir': ln_minus(g, n),
    'ln_minus_r_edge': ln_minus(r, edge),
    'ln_minus_r_nir': ln_minus(r, n),
    'ln_minus_edge_nir': ln_minus(edge, n),
    'divide_b_g': divide(b, g),
    'divide_b_r': divide(b, r),
    'divide_b_edge': divide(b, edge),
    'divide_b_nir': divide(b, n),
    'divide_g_b': divide(g, b),
    'divide_g_r': divide(g, r),
    'divide_g_edge': divide(g, edge),
    'divide_g_nir': divide(g, n),
    'divide_r_b': divide(r, b),
    'divide_r_g': divide(r, g),
    'divide_r_edge': divide(r, edge),
    'divide_r_nir': divide(r, n),
    'divide_edge_b': divide(edge, b),
    'divide_edge_g': divide(edge, g),
    'divide_edge_r': divide(edge, r),
    'divide_edge_nir': divide(edge, n),
    'divide_nir_b': divide(n, b),
    'divide_nir_g': divide(n, g),
    'divide_nir_r': divide(n, r),
    'divide_nir_edge': divide(n, edge),
    'ln_divide_b_g': ln_divide(b, g),
    'ln_divide_b_r': ln_divide(b, r),
    'ln_divide_b_edge': ln_divide(b, edge),
    'ln_divide_b_nir': ln_divide(b, n),
    'ln_divide_g_r': ln_divide(g, r),
    'ln_divide_g_edge': ln_divide(g, edge),
    'ln_divide_g_nir': ln_divide(g, n),
    'ln_divide_r_edge': ln_divide(r, edge),
    'ln_divide_r_nir': ln_divide(r, n),
    'ln_divide_edge_nir': ln_divide(edge, n)})

# 將重複的數值清除只留一個(keep='last')保留重複最後的名稱 (keep=False)將所有重複的數值刪去
drop_series = data.drop_duplicates()
biggest = drop_series.nlargest(number+1)
#print("前幾高pearson", biggest)
polynomial = biggest.index[1:]
print(data[polynomial])
print(polynomial)
