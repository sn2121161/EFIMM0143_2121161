# coding:utf-8

# import pybamm as pb

from scipy import io
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

pd.set_option('display.max_rows', None)
# path = 'D:\download_apps\SOC_40-60_HalfC\SOC_40%-60%_HalfC\PL04.mat'
# file = io.loadmat(path)
# print(file['PL04'][1], file['PL04'][2])

# 把三个sheet合到一起
df = pd.DataFrame([])
for i in range(1, 4):
    da = pd.read_excel('12_2_2015_Incremental OCV test_SP20-1.xlsx', sheet_name = 'Channel_1-005_%d'%(i))
    df = df.append(da)
# concat two sheets to get the whole process
# df = pd.read_excel('D:\\download_apps\\SP1_25C_IC_OCV_12_2_2015\\12_2_2015_Incremental OCV test_SP20-1.xlsx', sheet_name = 'Channel_1-005_1')
# df2 = pd.read_excel('D:\\download_apps\\SP1_25C_IC_OCV_12_2_2015\\12_2_2015_Incremental OCV test_SP20-1.xlsx', sheet_name = 'Channel_1-005_2')
# df = pd.concat([df1.iloc[63000: 63990], df2.iloc[: 3000]])
# print(df.columns)
# print(df.shape)

# 整理出那些起始点，看看这个实验是怎么做的
def test(ls):
    res = []
    for i in range(1, len(ls)):
        if ls[i-1] == 0 and ls[i] != 0:
            res.append(i+1)
        if ls[i-1] != 0 and ls[i] == 0:
            res.append(i+1)
    res.sort()
    return res

# print(test(list(df['Current(A)'])))

# 放电阶段的数据开始到结束 (excel)
# df[773:1486]  # 1  713   7126
# df[8612:9325]  # 2  713  7126
# df[16451:17864]  # 3
# df[773:1486]  # 4
# df[773:1486]  # 5
# df[773:1486]  # 6
# df[773:1486]  # 7
# df[773:1486]  # 8
# df[773:1486]  # 9
# df[773:1486]  # 10




# fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
# ax1 = ax.plot(df['soc'], df['voltage'])
# ax2 = ax.plot(df2['soc'], df2['vol'])
# plt.xlabel("soc")
# plt.ylabel("voltage")
# plt.title("Voltage decrease of discharge stage")
# plt.legend("voltage")
# plt.show()


# soc and voltage
#
#
#
# # 拟合测试
# line = np.polyfit(df['soc'], df['voltage'], 7)
# para = np.poly1d(line)
# print(para)

# print(df_discharge)


# 放电，充电过程过程的图像
# df = df[df['Current(A)'].apply(lambda x: x < 0)]
# df = df[df['Discharge_Capacity(Ah)'].apply(lambda x: x > 0)]
# print(df2.shape)

fig, ax = plt.subplots(figsize=(8,6), dpi = 100)
plt.style.use('ggplot')
#
# ax.plot([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0],
# [3.808, 3.691, 3.581, 3.482, 3.42, 3.379, 3.344, 3.297, 3.208, 2.5])

# sns.lineplot(df['Test_Time(s)'], df['Voltage(V)'])

ax1 = sns.lineplot(df['Test_Time(s)'].iloc[78500:80500], df['Voltage(V)'].iloc[78500:80500])
# ax2 = sns.lineplot(df['Discharge_Capacity(Ah)'], df['Voltage(V)'])
# ax2 = sns.lineplot(df['Test_Time(s)'].iloc[86000:88000], df['Voltage(V)'].iloc[86000:88000])
# ax3 = sns.lineplot(df['Test_Time(s)'].iloc[94000:96000], df['Voltage(V)'].iloc[94000:96000])


# 全数据的电压电流曲线呈现状态
# ax1 = ax.plot(df['Test_Time(s)'].iloc[54:141848], df['Voltage(V)'].iloc[54:141848], label='U')
# ax2 = ax.plot(df['Test_Time(s)'].iloc[54:141848], df['Current(A)'].iloc[54:141848], label='A')
#
# plt.xlabel("Test_Time(s)")
# plt.legend("Voltage(V)")
# plt.legend("Current(A)")
# plt.ylabel("value V-A")
# plt.title("Charge and discharge graph of the whole process")
# # ax2 = ax.plot(df['Test_Time(s)'].iloc[7000: 11000], df['Current(A)'].iloc[7000: 11000], label  ='current')
# plt.show()

# 全电压和SOC的图像
# ax1 = ax.plot(df_discharge['Discharge_Capacity(Ah)'], df_discharge['Voltage(V)'])
plt.xlabel('Test_Time')
plt.ylabel('voltage(V)')
plt.legend('voltage')
plt.title("Voltage changes on the charging stage")
plt.show()


