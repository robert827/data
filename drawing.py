# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 16:07:51 2021

@author: AMD_PC
"""

from micasense import plotutils
import matplotlib.pyplot as plt
import numpy as np
import cv2

blue = np.load('blue.npy')
green = np.load('green.npy')
red = np.load('red.npy')
rededge = np.load('rededge.npy')
nir = np.load('nir.npy')
# thermal = np.load('thermal.npy')
b=blue*100
g=green*100
r=red*100
re=rededge*100
n=nir*100


#分離陸地和水
g = cv2.GaussianBlur(g,(75,75),0)
n = cv2.GaussianBlur(n,(75,75),0)

ndwi = (g-n)/(g+n)    #ndwi



import numpy as np
water = ndwi
threshold = 0.23  #閥值0.23

water[water > threshold] = 1    #二值化
water[water < threshold] = 0
water2 = (water+1)%2
mask = water2 == 1
mx = np.ma.array(water2, mask=mask)
mx = (mx+1)%2



#迴歸模型
from numpy import log as ln
#do = -4.766568597764355+8.65157*(re/b)+3.71332*(re/r)
#sd = -148.15972753386228+786.509*(b/re)-578.341*(re/b)-12.2402*(re/r)-1418.79*(ln(b/re))
#tp = 74.64894665023942+26.8326*(b/re)-29.9983*(ln(re/b))
#chla = 147.7982562902869-22.5735*(re/r)+4.60035*(r-re)


tp = 209.71+12.22*(b-re)-1.48*(b/re)
do = 7.416-4.443*(1/g)+0.166*(g-r)
nh3n = 0.81+6.237*(1/b)-0.093*(b-n)
bod = 2.5+1.98267*(ln(b-n))+0.297802*(ln(g-r))
ss = 140.26668682994864-127.435*(b/r)







# ctis =  82.21412460847752-5.02451*(g/b)


#濃度分布可視化
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
do = cv2.GaussianBlur(do,(blurry,blurry),0)
do = mx*do
plt.imshow(do, cmap=plt.cm.get_cmap('jet', scale), vmin=3, vmax=10)
plt.title("DO(mg/L)")
plt.colorbar()
plt.axis('off')

plt.subplot(3, 2, 3)
bod = cv2.GaussianBlur(bod,(blurry,blurry),0)
bod = mx*bod
plt.imshow(bod, cmap=plt.cm.get_cmap('jet', scale), vmin=1, vmax=3)
plt.title("BOD(mg/L)")
plt.colorbar()
plt.axis('off')

plt.subplot(3, 2, 4)
ss = cv2.GaussianBlur(ss,(blurry,blurry),0)
ss = mx*ss
plt.imshow(ss, cmap=plt.cm.get_cmap('jet', scale), vmin=np.min(ss)-5, vmax=np.max(ss)+5)
plt.title("SS(mg/L)")
plt.colorbar()
plt.axis('off')

plt.subplot(3, 2, 5)
nh3n = cv2.GaussianBlur(nh3n,(blurry,blurry),0)
nh3n = mx*nh3n
plt.imshow(nh3n, cmap=plt.cm.get_cmap('jet', scale), vmin=0, vmax=5)
plt.title("NH3-N(mg/L)")
plt.colorbar()
plt.axis('off')

plt.subplot(3, 2, 6)
tp = cv2.GaussianBlur(tp,(blurry,blurry),0)
tp = mx*tp
plt.imshow(tp, cmap=plt.cm.get_cmap('jet', scale), vmin=190, vmax=220)
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
