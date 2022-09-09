# coding:utf-8
from functools import reduce
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

dis_vol = {
    1: 4.1757707595825195,
    0.9: 4.050173759460449,
    0.8: 3.9399726390838623,
    0.7: 3.8399813175201416,
    0.6: 3.753603219985962,
    0.5: 3.6647939682006836,
    0.4: 3.62606143951416,
    0.3: 3.5994834899902344,
    0.2: 3.555565118789673,
    0.1: 3.46772837638855}

ch_vol = {0.1: 3.2641804218292236,
          0.2: 3.482800006866455,
          0.3: 3.5769572257995605,
          0.4: 3.610665798187256,
          0.5: 3.6369194984436035,
          0.6: 3.6737072467803955,
          0.7: 3.763002634048462,
          0.8: 3.8485705852508545,
          0.9: 3.980163812637329,
          1: 4.1757707595825195}


# dis_vol1 = list(dis_vol.values())[:-3]
dis_vol2 = list(reversed(list(dis_vol.values())))
# # print(dis_vol, list(ch_vol.values()))
emf = list(map(lambda a, b: (a+b)/2, dis_vol2, list(ch_vol.values())))
diff = list(map(lambda a, b: (a-b)/2, dis_vol2, list(ch_vol.values())))
print(emf)
print(diff)


# draw multi-pictures
'''
plt.figure(figsize=(8,6), dpi=100)
plt.style.use('ggplot')

plt.subplot(2,2,1)
sns.lineplot(list(dis_vol.keys()), list(dis_vol.values()), color='r')
plt.title("Relationship between voltage and soc (discharging phase)", fontsize=12)
plt.xlabel("soc")
plt.ylabel("voltage")
plt.legend(['voltage-discharge'])


plt.subplot(2,2,2)
sns.lineplot(list(ch_vol.keys()), list(ch_vol.values()), color = 'g')
plt.title("Relationship between voltage and soc (charging phase)", fontsize=12)
plt.xlabel("soc")
plt.ylabel("voltage")
plt.legend(['voltage-charge'])


plt.subplot(2,2,3)
sns.lineplot(list(ch_vol.keys()), emf, color = 'b')
plt.title("Relationship between voltage and soc (average)", fontsize=12)
plt.xlabel("soc")
plt.ylabel("voltage")
plt.legend(['voltage-average'])


plt.subplot(2,2,4)
sns.lineplot(list(ch_vol.keys()), diff, color = 'y')
plt.title("Relationship between voltage and soc (difference)", fontsize=12)
plt.xlabel("soc")
plt.ylabel("voltage")
plt.legend(['voltage-difference'])
plt.show()


# 拟合一下SOC和电压的关系，用emf
line = np.polyfit(list(ch_vol.keys()), emf, 7)
parameter = np.poly1d(line)
# print(parameter)
'''


# 欧姆内阻的计算
R_0d=[0.230,	0.224,	0.222,	0.221,	0.221,	0.222,	0.221,	0.220,	0.220,	0.219]
R_0c=[0.237,	0.221,	0.217,	0.236,	0.239,	0.237,	0.234,	0.232,	0.230,	0.220]
r0_v = list(map(lambda a, b: a+b, R_0d, R_0c))

'''
plt.figure(figsize = (8, 6))
plt.style.use('ggplot')
plt.title("Relationship between ohmic resistance and soc")

plt.subplot(3,1,1)
sns.lineplot(x = list(ch_vol.keys()), y = R_0d, color='r')
plt.xlabel("soc")
plt.ylabel("ohmic resistance")
plt.title("Relationship between ohmic resistance and soc")
plt.legend(['discharging stage'])

plt.subplot(3,1,2)
sns.lineplot(x = list(ch_vol.keys()), y = R_0c, color = 'g')
plt.xlabel("soc")
plt.ylabel("ohmic resistance")
plt.legend(['charging stage'])

plt.subplot(3,1,3)
sns.lineplot(x = list(ch_vol.keys()), y = r0_v, color = 'b')
plt.xlabel("soc")
plt.ylabel("ohmic resistance")
plt.legend(['average'])


plt.show()
'''
line2 = np.polyfit(list(ch_vol.keys()), r0_v , 7)
para = np.poly1d(line2)
print(para)



