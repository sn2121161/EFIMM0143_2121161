# coding:utf-8

# 构建SOC和各阶段的对应关系
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt



# 构建表，用append的形式

df = pd.DataFrame([])
for i in range(1, 4):
    da = pd.read_excel('D:\\python_project\\application-research-project\\12_2_2015_Incremental OCV test_SP20-1.xlsx', sheet_name = 'Channel_1-005_%d'%(i))
    df = df.append(da)
df = df.reset_index()


# check一下放电区间
# print(df['Current(A)'].iloc[772:1485])  # 100
# print(df['Current(A)'].iloc[8611:9324])  #90
# print(df['Current(A)'].iloc[16450:17163])  # 80
# print(df['Current(A)'].iloc[24289:25002])  # 70
# print(df['Current(A)'].iloc[32129:32842])  # 60
# print(df['Current(A)'].iloc[39969:40682])  # 50
# print(df['Current(A)'].iloc[47808:48521])  # 40
# print(df['Current(A)'].iloc[55647:56360])  # 30
# print(df['Current(A)'].iloc[63485:64199])  # 20
# print(df['Current(A)'].iloc[71325:71994])  # 10

# 充电区间的标记位置，可以用来看充电阶段的极化内阻的情况
# print(df['Current(A)'].iloc[79121:79834]) # 10
# print(df['Current(A)'].iloc[86962:87674]) # 20
# print(df['Current(A)'].iloc[94804:95517]) # 30
# print(df['Current(A)'].iloc[102644:103357]) # 40
# print(df['Current(A)'].iloc[110484:111197]) # 50
# print(df['Current(A)'].iloc[118324:119037]) # 60
# print(df['Current(A)'].iloc[126165:126878]) # 70
# print(df['Current(A)'].iloc[134005:134718]) # 80
# print(df['Current(A)'].iloc[134720:134718]) # 90
# print(df['Current(A)'].iloc[142121]) # 100  4.176



'''
# 计算放电时的欧姆内阻和极化内阻
time_constant = 12*60
# vol = {
#     "soc=100": df['Voltage(V)'].iloc[772],
#     "soc=90": df['Voltage(V)'].iloc[8611],
#     "soc=80": df['Voltage(V)'].iloc[16450],
#     "soc=70": df['Voltage(V)'].iloc[24289],
#     "soc=60": df['Voltage(V)'].iloc[32129],
#     "soc=50": df['Voltage(V)'].iloc[39969],
#     "soc=40": df['Voltage(V)'].iloc[47808],
#     "soc=30": df['Voltage(V)'].iloc[55647],
#     "soc=20": df['Voltage(V)'].iloc[63485],
#     "soc=10": df['Voltage(V)'].iloc[71325]
# }
vol2 = {
    "soc=100": df['Voltage(V)'].iloc[771],
    "soc=90": df['Voltage(V)'].iloc[8610],
    "soc=80": df['Voltage(V)'].iloc[16449],
    "soc=70": df['Voltage(V)'].iloc[24289-1],
    "soc=60": df['Voltage(V)'].iloc[32129-1],
    "soc=50": df['Voltage(V)'].iloc[39969-1],
    "soc=40": df['Voltage(V)'].iloc[47808-1],
    "soc=30": df['Voltage(V)'].iloc[55647-1],
    "soc=20": df['Voltage(V)'].iloc[63485-1],
    "soc=10": df['Voltage(V)'].iloc[71325-1]
}

r0d = {
    "soc=100": abs((df['Voltage(V)'].iloc[771] - df['Voltage(V)'].iloc[772])/df['Current(A)'].iloc[772]),
    "soc=90": abs((df['Voltage(V)'].iloc[8610] - df['Voltage(V)'].iloc[8611])/df['Current(A)'].iloc[8611]),
    "soc=80": abs((df['Voltage(V)'].iloc[16450-1] - df['Voltage(V)'].iloc[16450])/df['Current(A)'].iloc[16450]),
    "soc=70": abs((df['Voltage(V)'].iloc[24289-1] - df['Voltage(V)'].iloc[24289])/df['Current(A)'].iloc[24289]),
    "soc=60": abs((df['Voltage(V)'].iloc[32129-1] - df['Voltage(V)'].iloc[32129])/df['Current(A)'].iloc[32129]),
    "soc=50": abs((df['Voltage(V)'].iloc[39969-1] - df['Voltage(V)'].iloc[39969])/df['Current(A)'].iloc[39969]),
    "soc=40": abs((df['Voltage(V)'].iloc[47808-1] - df['Voltage(V)'].iloc[47808])/df['Current(A)'].iloc[47808]),
    "soc=30": abs((df['Voltage(V)'].iloc[55647-1] - df['Voltage(V)'].iloc[55647])/df['Current(A)'].iloc[55647]),
    "soc=20": abs((df['Voltage(V)'].iloc[63485-1] - df['Voltage(V)'].iloc[63485])/df['Current(A)'].iloc[63485]),
    "soc=10": abs((df['Voltage(V)'].iloc[71325-1] - df['Voltage(V)'].iloc[71325])/df['Current(A)'].iloc[71325])
}

print(vol2)
print('--------------------------')
print(r0d)

'''



# 计算充电时的欧姆内阻和极化内阻
'''
volc = {
    "soc=10": df['Voltage(V)'].iloc[79121-1],
    "soc=20": df['Voltage(V)'].iloc[86962-1],
    "soc=30": df['Voltage(V)'].iloc[94804-1],
    "soc=40": df['Voltage(V)'].iloc[102644-1],
    "soc=50": df['Voltage(V)'].iloc[110484-1],
    "soc=60": df['Voltage(V)'].iloc[118324-1],
    "soc=70": df['Voltage(V)'].iloc[126165-1],
    "soc=80": df['Voltage(V)'].iloc[134005-1],
    "soc=90": df['Voltage(V)'].iloc[134720-1],
    "soc=100":df['Voltage(V)'].iloc[142121-1]
}

r0c = {
    "soc=10": abs((df['Voltage(V)'].iloc[79120] - df['Voltage(V)'].iloc[79121]) / df['Current(A)'].iloc[79121]),
    "soc=20": abs((df['Voltage(V)'].iloc[86962-1] - df['Voltage(V)'].iloc[86962]) / df['Current(A)'].iloc[86962]),
    "soc=30": abs((df['Voltage(V)'].iloc[94804-1] - df['Voltage(V)'].iloc[94804])/df['Current(A)'].iloc[94804]),
    "soc=40": abs((df['Voltage(V)'].iloc[102644-1] - df['Voltage(V)'].iloc[102644])/df['Current(A)'].iloc[102644]),
    "soc=50": abs((df['Voltage(V)'].iloc[110484-1] - df['Voltage(V)'].iloc[110484])/df['Current(A)'].iloc[110484]),
    "soc=60": abs((df['Voltage(V)'].iloc[118324-1] - df['Voltage(V)'].iloc[118324])/df['Current(A)'].iloc[118324]),
    "soc=70": abs((df['Voltage(V)'].iloc[126165-1] - df['Voltage(V)'].iloc[126165])/df['Current(A)'].iloc[126165]),
    "soc=80": abs((df['Voltage(V)'].iloc[134006-1] - df['Voltage(V)'].iloc[134006])/df['Current(A)'].iloc[134006]),
    "soc=90": abs((df['Voltage(V)'].iloc[134718] - df['Voltage(V)'].iloc[134719]) / df['Current(A)'].iloc[134718]),
    "soc=100": abs((df['Voltage(V)'].iloc[142120] - 4.176) / df['Current(A)'].iloc[142120])

}

volct = {
    df["Test_Time(s)"].iloc[79121-1]: df['Voltage(V)'].iloc[79121-1],
    df["Test_Time(s)"].iloc[86962-1]: df['Voltage(V)'].iloc[86962-1],
    df["Test_Time(s)"].iloc[94804-1]: df['Voltage(V)'].iloc[94804-1],
    df["Test_Time(s)"].iloc[102644-1]: df['Voltage(V)'].iloc[102644-1],
    df["Test_Time(s)"].iloc[110484-1]: df['Voltage(V)'].iloc[110484-1],
    df["Test_Time(s)"].iloc[118324-1]: df['Voltage(V)'].iloc[118324-1],
    df["Test_Time(s)"].iloc[126165-1]: df['Voltage(V)'].iloc[126165-1],
    df["Test_Time(s)"].iloc[134005-1]: df['Voltage(V)'].iloc[134005-1],
    df["Test_Time(s)"].iloc[134720-1]: df['Voltage(V)'].iloc[134720-1],
    df["Test_Time(s)"].iloc[142121-1]:df['Voltage(V)'].iloc[142121-1]
}




print(volct)
print('------------------')
'''

# parameter identify
'''
y = k0-k1e^(-b1t) -k2e^(-b2t) there are four parameters, k1, k2, k3, k4
k1e^(-b1t)+k2e^(-bt2)=k0-y
y-k0, t is known
each SoC state has different parameters
so for each step, it should has 4 equations, two for charging, two for discharging
'''

def func(t, k0, k1, b1, k2, b2):
    return k0 - k1 * np.exp(-1 * b1 * t) - k2 * np.exp(-1 * b2 * t)


# xd1 = list(map(lambda x: x - df['Test_Time(s)'].iloc[1483], df['Test_Time(s)'].iloc[1484: 1862]))
yd2 = list(map(lambda x: x - df['Voltage(V)'].iloc[1483], df['Voltage(V)'].iloc[1484: 5000]))

xd1 = list(map(lambda x: x - df['Test_Time(s)'].iloc[1483], df['Test_Time(s)'].iloc[1484: 5000]))
yd1 = list(df['Voltage(V)'].iloc[1484: 5000])


popc, pcov = curve_fit(func, xd1, yd2, maxfev=50000)
print(popc)

r1 = popc[1] / (-1 * df['Current(A)'].iloc[1483])
r2 = popc[3] / (-1 * df['Current(A)'].iloc[1483])
c1 = 1 / (r1*popc[2])
c2 = 1 / (r2*popc[4])
print(r1, r2, c1, c2)