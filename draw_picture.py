# coding:utf-8

from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

x = np.linspace(0, 10, 100)
y1 = list(map(lambda x: np.exp(x/13)-1, x))
y2 = list(map(lambda x: np.sqrt(x/13), x))
y3 = [2 for i in range(1, 101)]
yt = list(map(lambda a, b, c: a+b+c, y1,y2,y3))

# print(x)
# print(y2)
plt.style.use("ggplot")
fig, ax = plt.subplots(figsize = (10, 6), dpi =100)


ax1 = sns.lineplot(x = x, y = y1)
ax2 = sns.lineplot(x = x, y = y2)
ax3 = sns.lineplot(x = x, y = y3)
ax4 = sns.lineplot(x = x, y = yt)
plt.legend(['Differential concentration polarisation', 'Electrochemical polarisation', "ohmic polarisation", "battery total polarisation"])
plt.title("Different contributions of three types of polarisation")
plt.xlabel("Time(s)")
plt.ylabel("Voltage(v)")

plt.show()