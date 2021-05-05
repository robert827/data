# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 16:07:51 2021

@author: AMD_PC
"""

from numpy import log as ln
from micasense import plotutils
import matplotlib.pyplot as plt
import numpy as np
import cv2

b = np.load('blue.npy')
g = np.load('green.npy')
r = np.load('red.npy')
re = np.load('rededge.npy')
n = np.load('nir.npy')
# 將0項 更改為0.1
b[b <= 0.0] = 0.1
g[g <= 0.0] = 0.1
r[r <= 0.0] = 0.1
re[re <= 0.0] = 0.1
n[n <= 0.0] = 0.1

# thermal = np.load('thermal.npy')

# 分離陸地和水
g = cv2.GaussianBlur(g, (75, 75), 0)
n = cv2.GaussianBlur(n, (75, 75), 0)

ndwi = (g-n)/(g+n)  # ndwi


water = ndwi
threshold = 0.35  # 閥值0.23  #####關鍵

water[water > threshold] = 1  # 二值化
water[water < threshold] = 0
water2 = (water+1) % 2
mask = water2 == 1
mx = np.ma.array(water2, mask=mask)
mx = (mx+1) % 2


# 迴歸模型
#do = -4.766568597764355+8.65157*(re/b)+3.71332*(re/r)
#sd = -148.15972753386228+786.509*(b/re)-578.341*(re/b)-12.2402*(re/r)-1418.79*(ln(b/re))
#tp = 74.64894665023942+26.8326*(b/re)-29.9983*(ln(re/b))
#chla = 147.7982562902869-22.5735*(re/r)+4.60035*(r-re)


tp = 321.5543527260936+ln(r-re+2)*-18.383172673518466 + \
    (r-re)*-58.69946889092934+ln(g-re+2)*-66.78355137654445
do = -66.51197238904545+ln(g-re+2)*107.4000413061391 + \
    ln(r-re+2)*-5.354033204154389+(g-re)*-40.09561193703867
nh3n = -3.292821754723806+(g-n)*-2.5550290792942345 + \
    ln(g-n+2)*6.767628927423466+(g-r)*-0.8268918126619064
bod = -152.57148568490678+ln(g-r+2)*218.90513526343148 + \
    (g-r)*-93.9315743799634+(b-n)*1.317117160145991
ss = 140.26668682994864-127.435*(b/r)


# ctis =  82.21412460847752-5.02451*(g/b)


# 濃度分布可視化
size = 3
plt.figure(figsize=(5*size, 5*size))


scale = 255
blurry = 101

rgb = cv2.imread('rgb.png')
plt.subplot(3, 2, 1)
plt.imshow(rgb, cmap=plt.cm.get_cmap('jet', scale), vmin=60, vmax=80)
plt.title("RGB")
bar = plt.colorbar()
bar.remove()
plt.axis('off')

plt.subplot(3, 2, 2)
do = cv2.GaussianBlur(do, (blurry, blurry), 0)
do = mx*do
plt.imshow(do, cmap=plt.cm.get_cmap('jet', scale),
           vmin=np.min(do), vmax=np.max(do))
plt.title("DO(mg/L)")
plt.colorbar()
plt.axis('off')

plt.subplot(3, 2, 3)
bod = cv2.GaussianBlur(bod, (blurry, blurry), 0)
bod = mx*bod
plt.imshow(bod, cmap=plt.cm.get_cmap('jet', scale),
           vmin=np.min(bod), vmax=np.max(bod))
plt.title("BOD(mg/L)")
plt.colorbar()
plt.axis('off')

plt.subplot(3, 2, 4)
ss = cv2.GaussianBlur(ss, (blurry, blurry), 0)
ss = mx*ss
plt.imshow(ss, cmap=plt.cm.get_cmap('jet', scale),
           vmin=np.min(ss), vmax=np.max(ss))
plt.title("SS(mg/L)")
plt.colorbar()
plt.axis('off')

plt.subplot(3, 2, 5)
nh3n = cv2.GaussianBlur(nh3n, (blurry, blurry), 0)
nh3n = mx*nh3n
plt.imshow(nh3n, cmap=plt.cm.get_cmap('jet', scale),
           vmin=np.min(nh3n), vmax=np.max(nh3n))
plt.title("NH3-N(mg/L)")
plt.colorbar()
plt.axis('off')

plt.subplot(3, 2, 6)
tp = cv2.GaussianBlur(tp, (blurry, blurry), 0)
tp = mx*tp
plt.imshow(tp, cmap=plt.cm.get_cmap('jet', scale),
           vmin=np.min(tp), vmax=np.max(tp))
plt.title("TP(mg/L)")
plt.colorbar()
plt.axis('off')
# plt.subplot(2, 3, 6)
# ctis = cv2.GaussianBlur(ctis,(blurry,blurry),0)
# ctis = mx*ctis
# plt.imshow(ctis, cmap=plt.cm.get_cmap('jet', scale), vmin=60, vmax=80)
# plt.title("CTIS")
# plt.colorbar()
# plt.axis('off')
