# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 18:07:31 2021

@author: AMD_PC
"""
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 13:49:07 2021

@author: AMD_PC
"""
import numpy as np
import pandas as pd
from scipy.stats import stats
from itertools import product as prod
import triplefunction
number = int(input("特徵數量(整數):"))
df = pd.read_excel('data.xlsx', index_col=False)
data = {'g': df['green'], 'b': df['blue'], 'r': df['red'], 'e': df['rededge'], 'n': df['nir'],
        'a': np.log(df['blue']+2), 's': np.log(df['green']+2), 'd': np.log(df['red']+2),
        'f': np.log(df['rededge']+2), 'h': np.log(df['nir']+2)}
# create a datadict, which can access the value by key name(like 'e')
sequence = list(prod('bgrneasdfh', repeat=3))
times = len(sequence)
#number = int(input("特徵數量(整數):"))
df = pd.read_excel('data.xlsx', index_col=False)

value_list=[]
key = 'ln(n1+n2+n3)'
name12 = [key]*times
for i in range(0, times):
    zxc = sequence[i]
    zxcv_value = format(triplefunction.triple_lnplus(
        data[zxc[0]], data[zxc[1]], data[zxc[2]]), '.10f')
    value_list += [zxcv_value]
    
key = 'ln(n1-n2-n3)'
name13 = [key]*times
for i in range(0, times):
    zxc = sequence[i]
    zxcv_value = format(triplefunction.triple_lnminus(
        data[zxc[0]], data[zxc[1]], data[zxc[2]]), '.10f')
    value_list += [zxcv_value]

key = 'ln(n1*n2*n3)'
name14 = [key]*times
for i in range(0, times):
    zxc = sequence[i]
    zxcv_value = format(triplefunction.triple_lnmultiply(
        data[zxc[0]], data[zxc[1]], data[zxc[2]]), '.10f')
    value_list += [zxcv_value]

key = 'ln(n1/(n2+n3))'
name15 = [key]*times
for i in range(0, times):
    zxc = sequence[i]
    zxcv_value = format(triplefunction.triple_lnplus1(
        data[zxc[0]], data[zxc[1]], data[zxc[2]]), '.10f')
    value_list += [zxcv_value]

key = 'ln(n1/(n2-n3))'
name16 = [key]*times
for i in range(0, times):
    zxc = sequence[i]
    zxcv_value = format(triplefunction.triple_lnminus1(
        data[zxc[0]], data[zxc[1]], data[zxc[2]]), '.10f')
    value_list += [zxcv_value]
    
key = 'ln(n1/(n2*n3))'
name17 = [key]*times
for i in range(0, times):
    zxc = sequence[i]
    zxcv_value = format(triplefunction.triple_lndivide(
        data[zxc[0]], data[zxc[1]], data[zxc[2]]), '.10f')
    value_list += [zxcv_value]
    
key = 'ln((n1-n2)*n3)'
name18 = [key]*times
for i in range(0, times):
    zxc = sequence[i]
    zxcv_value = format(triplefunction.triple_lnminusmultiply(
        data[zxc[0]], data[zxc[1]], data[zxc[2]]), '.10f')
    value_list += [zxcv_value]

key = 'ln(n1+n2)*n3)'
name19 = [key]*times
for i in range(0, times):
    zxc = sequence[i]
    zxcv_value = format(triplefunction.triple_lnplusmultiply(
        data[zxc[0]], data[zxc[1]], data[zxc[2]]), '.10f')
    value_list += [zxcv_value]

key = 'ln((n1+n2)/n3)'
name20 = [key]*times
for i in range(0, times):
    zxc = sequence[i]
    zxcv_value = format(triplefunction.triple_lnplusdivide(
        data[zxc[0]], data[zxc[1]], data[zxc[2]]), '.10f')
    value_list += [zxcv_value]
  
key = 'ln((n1*n2)/n3)'
name21 = [key]*times
for i in range(0, times):
    zxc = sequence[i]
    zxcv_value = format(triplefunction.triple_lnmultiplydivide(
        data[zxc[0]], data[zxc[1]], data[zxc[2]]), '.10f')
    value_list += [zxcv_value]
    
  

total_name = list(zip(name12+name13+name14+name15+name16+name17+name18+name19+name20+name21,
                      sequence*int(len(value_list)/1000)))
z = pd.Series(value_list, index=total_name, dtype='float64')  # float64 是重
z = z.drop_duplicates()
biggest = z.nlargest(number)
polynomial = biggest.index
