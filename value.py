# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 13:49:07 2021

@author: AMD_PC
"""
from polynomialfunction import substance
from value_testfunction import value_normal, value_ln_normal, value_plus, value_minus, value_ln_plus, value_ln_minus, value_divide, value_ln_divide
import numpy as np
import pandas as pd
from scipy.stats import stats
df = pd.read_excel('data.xlsx', index_col=False)
b = df['blue']
g = df['green']
r = df['red']
edge = df['rededge']
n = df['nir']
data = dict({
    'highest': 1,
    'blue': value_normal(b),
    'green': value_normal(g),
    'red': value_normal(r),
    'rededge': value_normal(edge),
    'nir': value_normal(n),
    'ln_blue': value_ln_normal(b),
    'ln_green': value_ln_normal(g),
    'ln_red': value_ln_normal(r),
    'ln_rededge': value_ln_normal(edge),
    'ln_nir': value_ln_normal(n),
    'plus_b_g': value_plus(b, g),
    'plus_b_r': value_plus(b, r),
    'plus_b_edge': value_plus(b, edge),
    'plus_b_nir': value_plus(b, n),
    'plus_g_r': value_plus(g, r),
    'plus_g_edge': value_plus(g, edge),
    'plus_g_nir': value_plus(g, n),
    'plus_r_edge': value_plus(r, edge),
    'plus_r_nir': value_plus(r, n),
    'plus_edge_nir': value_plus(edge, n),
    'minus_b_g': value_minus(b, g),
    'minus_b_r': value_minus(b, r),
    'minus_b_edge': value_minus(b, edge),
    'minus_b_nir': value_minus(b, n),
    'minus_g_r': value_minus(g, r),
    'minus_g_edge': value_minus(g, edge),
    'minus_g_nir': value_minus(g, n),
    'minus_r_edge': value_minus(r, edge),
    'minus_r_nir': value_minus(r, n),
    'minus_edge_nir': value_minus(edge, n),
    'ln_plus_b_g': value_ln_plus(b, g),
    'ln_plus_b_r': value_ln_plus(b, r),
    'ln_plus_b_edge': value_ln_plus(b, edge),
    'ln_plus_b_nir': value_ln_plus(b, n),
    'ln_plus_g_r': value_ln_plus(g, r),
    'ln_plus_g_edge': value_ln_plus(g, edge),
    'ln_plus_g_nir': value_ln_plus(g, n),
    'ln_plus_r_edge': value_ln_plus(r, edge),
    'ln_plus_r_nir': value_ln_plus(r, n),
    'ln_plus_edge_nir': value_ln_plus(edge, n),
    'ln_minus_b_g': value_ln_minus(b, g),
    'ln_minus_b_r': value_ln_minus(b, r),
    'ln_minus_b_edge': value_ln_minus(b, edge),
    'ln_minus_b_nir': value_ln_minus(b, n),
    'ln_minus_g_r': value_ln_minus(g, r),
    'ln_minus_g_edge': value_ln_minus(g, edge),
    'ln_minus_g_nir': value_ln_minus(g, n),
    'ln_minus_r_edge': value_ln_minus(r, edge),
    'ln_minus_r_nir': value_ln_minus(r, n),
    'ln_minus_edge_nir': value_ln_minus(edge, n),
    'divide_b_g': value_divide(b, g),
    'divide_b_r': value_divide(b, r),
    'divide_b_edge': value_divide(b, edge),
    'divide_b_nir': value_divide(b, n),
    'divide_g_b': value_divide(g, r),
    'divide_g_r': value_divide(g, r),
    'divide_g_edge': value_divide(g, edge),
    'divide_g_nir': value_divide(g, n),
    'divide_r_b': value_divide(r, b),
    'divide_r_g': value_divide(r, g),
    'divide_r_edge': value_divide(r, edge),
    'divide_r_nir': value_divide(r, n),
    'divide_edge_b': value_divide(edge, b),
    'divide_edge_g': value_divide(edge, g),
    'divide_edge_r': value_divide(edge, r),
    'divide_edge_nir': value_divide(edge, n),
    'divide_nir_b': value_divide(n, b),
    'divide_nir_g': value_divide(n, g),
    'divide_nir_r': value_divide(n, r),
    'divide_nir_edge': value_divide(n, edge),
    'ln_divide_b_g': value_ln_divide(b, g),
    'ln_divide_b_r': value_ln_divide(b, r),
    'ln_divide_b_edge': value_ln_divide(b, edge),
    'ln_divide_b_nir': value_ln_divide(b, n),
    'ln_divide_g_r': value_ln_divide(g, r),
    'ln_divide_g_edge': value_ln_divide(g, edge),
    'ln_divide_g_nir': value_ln_divide(g, n),
    'ln_divide_r_edge': value_ln_divide(r, edge),
    'ln_divide_r_nir': value_ln_divide(r, n),
    'ln_divide_edge_nir': value_ln_divide(edge, n)})

#qwe = data[polynomial[2]]
value_list = [substance, ]

for i in range(0, number, 1):   # 第二層迴圈執行num次，每次輸入成績
    value = data[polynomial[i]]
    value_list += [value]

finaldata = np.transpose(value_list)
