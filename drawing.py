# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 16:07:51 2021

@author: AMD_PC
"""
import math
from numpy import log as ln
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import cv2

b = np.load('blue.npy')
g = np.load('green.npy')
r = np.load('red.npy')
re = np.load('rededge.npy')
n = np.load('nir.npy')

a=ln(b+2)
s=ln(g+2)
d=ln(r+2)
f=ln(re+2)
h=ln(n+2)

# 將0項 更改為0.1
b[b <= 0.0] = 0.01
g[g <= 0.0] = 0.01
r[r <= 0.0] = 0.01
re[re <= 0.0] = 0.01
n[n<= 0.0] = 0.01

# thermal = np.load('thermal.npy')

# 分離陸地和水
g = cv2.GaussianBlur(g, (37, 37), 0)
n = cv2.GaussianBlur(n, (37, 37), 0)

ndwi = (g-n)/(g+n)  # ndwi


water = ndwi
threshold = 0.3  # 閥值0.23  #####關鍵

water[water > threshold] = 1  # 二值化
water[water < threshold] = 0
water2 = (water+1) % 2
mask = water2 == 1
mx = np.ma.array(water2, mask=mask)
mx = (mx+1) % 2


# 總數據的模型
#do=ln(r)*(-39.85260581105394)+(b/r)*(-24.973572642827786)+ln(g+r)*(-6.737021412074297)+ln(b+r)*(45.298129784209365)+ln(r-re+2)*(19302.273077817787)+(r-re)*(-9438.393162572196)+(-13380.486074101436)

#ss=(r/b)*(2120.749764287625)+ln(b/r)*(4617.674459391813)+(b/r)*(-2496.8118396774566)+ln(r-re+2)*(92792.90465005224)+(r-re)*(-45731.85920577422)+(re/r)*(153.78910887549492)+ln(r/re)*(133.29371687331113)+(-64080.382077084374)

#nh3n=(b/r)*(24.867239267389873)+ln(r)*(-0.9511472450933971)+ln(r+n)*(0.4312158593446043)+ln(b/r)*(-19.76642003162551)+(-24.961847126641874)

#bod=(n/re)*(73.94655120766447)+ln(re/n)*(-6.417819646591307)+ln(b/n)*(-14.200682327541099)+ln(r/n)*(-86.6536289470813)+(n/b)*(-4.524302790766111)+ln(g/n)*(25.705652659358034)+(r/n)*(5.112827926958432)+(re/n)*(5.116906516302272)+(n/r)*(-215.1018562396907)+(105.9755678060438)


#單筆數據模型
ss=(b/re)*(2836.243835485653)+ln(b/re)*(-4161.393126042226)+(b-re)*(-2066819.5606279066)+ln(b-re+2)*(4175366.2337553073)+(-2896887.3721405575)

bod=(g-n)*(46.455703093379476)+(-1.8614023030904727)

nh3n=ln(g-n+2)*(-23850.736118849763)+(g-n)*(11300.824227849698)+(b/n)*(0.11894486603548918)+(b/re)*(0.8840820776654675)+ln(g)*(6.712900493222946)+(16579.320668008593)

do=(b/n)*(-0.8048065524866753)+(re/n)*(-1.30393219216478)+ln(re/n)*(3.507069162646055)+(b/g)*(4.097660423787862)+(g/n)*(0.6746224872786607)+(4.018866123307145)
do[do<0]=0
bod[bod<0]=0
ss[ss<0]=0
nh3n[nh3n<0]=0
#擷取顯示範圍之數值上下限
def upper_lower_limit(substance):
    substance=substance*water
    
    substance=[i for item in substance for i in item]

    substance_set = set(substance)
    substance= list(substance_set)

    substance.sort()
   
    substance_min=math.floor(substance[int(len(substance)*0.1)])
    if substance_min <0:
        substance_min=0
    substance_max=(math.ceil(substance[int(len(substance)*0.8)]))
    return substance_min,substance_max
    
ss_limit=(upper_lower_limit(ss))
do_limit=(upper_lower_limit(do))
bod_limit=(upper_lower_limit(bod))
nh3n_limit=(upper_lower_limit(nh3n))
    

# ctis =  82.21412460847752-5.02451*(g/b)


# 濃度分布可視化
size = 2
plt.figure(figsize=(5*size, 5*size))


scale = 255
blurry = 37

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
           vmin=do_limit[0], vmax=do_limit[1])
plt.title("DO(mg/L)")
plt.colorbar()
plt.axis('off')

plt.subplot(3, 2, 3)
ss = cv2.GaussianBlur(ss, (blurry, blurry), 0)
ss = mx*ss
plt.imshow(ss, cmap=plt.cm.get_cmap('jet', scale),
           vmin=ss_limit[0], vmax=ss_limit[1])
plt.title("SS(mg/L)")
plt.colorbar()
plt.axis('off')

plt.subplot(3, 2, 4)
bod = cv2.GaussianBlur(bod, (blurry, blurry), 0)
bod = mx*bod
plt.imshow(bod, cmap=plt.cm.get_cmap('jet', scale),
           vmin=bod_limit[0], vmax=bod_limit[1])

plt.title("BOD(mg/L)")
plt.colorbar()
plt.axis('off')

plt.subplot(3, 2, 5)
nh3n = cv2.GaussianBlur(nh3n, (blurry, blurry), 0)
nh3n = mx*nh3n
plt.imshow(nh3n, cmap=plt.cm.get_cmap('jet', scale),
           vmin=nh3n_limit[0], vmax=nh3n_limit[1])

plt.title("NH3-N(mg/L)")
plt.colorbar()
plt.axis('off')
#

#河川汙染指數的規定
do[(do >= 6.5) & (do != 3) & (do != 6) & (do != 10) & (do != 0)] = 1.0
do[(6.5 > do) & (4.6 <= do) & (do != 1) & (
    do != 6) & (do != 10) & (do != 0)] = 3.0
do[(4.6 > do) & (2 <= do) & (do != 1) & (
    do != 3) & (do != 10) & (do != 0)] = 6.0
do[(do < 2) & (do >= 0) & (do != 1) & (do != 3) & (do != 6) & (do != 0)] = 10.0

bod[bod < 0] = 0.0
bod[(bod <= 3.0) & (bod > 0) & (bod != 3.0) & (
    bod != 6) & (bod != 10) & (bod != 0)] = 1.0
bod[(5.0 >= bod) & (3.0 < bod) & (bod != 1) & (
    bod != 6) & (bod != 10) & (bod != 0)] = 3.0
bod[(15.0 >= bod) & (5.0 < bod) & (bod != 1) & (
    bod != 3) & (bod != 10) & (bod != 0)] = 6.0
bod[(bod > 15.0) & (bod != 1) & (bod != 3) & (bod != 6) & (bod != 0)] = 10.0


ss[ss < 0] = 0.0
ss[(ss <= 20.0) & (ss > 0) & (ss != 3.0) & (
    ss != 6) & (ss != 10) & (ss != 0)] = 1.0
ss[(50.0 >= ss) & (3.0 < ss) & (ss != 1) & (
    ss != 6) & (ss != 10) & (ss != 0)] = 3.0
ss[(100 >= ss) & (50.0 < ss) & (ss != 1) & (
    ss != 3) & (ss != 10) & (ss != 0)] = 6.0
ss[(ss > 100.0)] = 10.0


nh3n[nh3n < 0] = 0.0
nh3n[(nh3n <= 0.5) & (nh3n > 0) & (nh3n != 3.0) & (
    nh3n != 6) & (nh3n != 10) & (nh3n != 0)] = 1.0
nh3n[(1.0 >= nh3n) & (0.5 < nh3n) & (nh3n != 1) & (
    nh3n != 6) & (nh3n != 10) & (nh3n != 0)] = 3.0
nh3n[(3 >= nh3n) & (1.0 < nh3n) & (nh3n != 1) & (
    nh3n != 3) & (nh3n != 10) & (nh3n != 0)] = 6.0
nh3n[(nh3n > 3.0) & (nh3n != 1) & (nh3n != 3)
     & (nh3n != 6) & (nh3n != 0)] = 10.0





pollution = (do+bod+ss+nh3n)/4
p = pollution
plt.subplot(3, 2, 6)
p = mx*p
bounds=[1,2,3,6,10]
cmap=mpl.colors.ListedColormap(['red','green','blue','yellow'])
norm=mpl.colors.BoundaryNorm(bounds,cmap.N)
plt.imshow(p, cmap=cmap,vmin=1,vmax=10)
plt.title("RPI")
plt.colorbar(
    mpl.cm.ScalarMappable(cmap=cmap,norm=norm),
    spacing='proportional',
    orientation='vertical')
plt.axis('off')
