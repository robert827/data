# -*- coding: utf-8 -*-
"""
Created on Thu May  6 20:53:04 2021

@author: AMD_PC
"""
from test_function import correction, data_point,regression
import pandas as pd
print('./data')
image_path = input("相片存放位置:")
print("IMG_0134")
image_name = input("相片名稱:")
print("./data")
panel_path = input("校正板相片存放位置:")
print("IMG_0003")
panel_name = input("校正板相片名稱:")
correction(image_path, image_name, panel_path, panel_name)

###抓指定座標反射率##


image='IMG_0134'
x=800
y=1200
data_point(image, x, y)

import pandas as pd
df = pd.read_excel('data.xlsx', index_col=False)
b=df['blue']
g = df['green']
r = df['red']
re = df['rededge']
n = df['nir']

regression(b, g, r, re, n)
#%%


