# coding:utf-8
# parameter identify
'''
y = k0-k1e^(-b1t) -k2e^(-b2t) there are four parameters, k1, k2, k3, k4
k1e^(-b1t)+k2e^(-bt2)=k0-y
y-k0, t is known
each SoC state has different parameters
so for each step, it should has 4 equations, two for charging, two for discharging
'''
from matplotlib import pyplot as plt
import numpy as np

# 恒流放电，I = 1
def calculate(k1, b1, k2, b2):
    i = 1
    r1 = i/k1
    r2 = i/k2
    c1 = 1/(r1*b1)
    c2 = 1/(r2*b2)
    return r1, r2, c1, c2

# 90%
k1 = [0.2197, 0.2201, 0.22, 0.2198, 0.2196, 0.2206, 0.2213, 0.2234, 0.2296]
b1 = [5.276, 5.152, 5.196, 5.322, 5.429, 5.815, 5.396, 5.279, 4.999]
k2 = [0.01468, 0.01938,0.02627, 0.01679, 0.0133, 0.01469, 0.01729, 0.01718, 0.01736]
b2 = [0.05621, 0.053, 0.04066, 0.04209, 0.05043, 0.04571, 0.03759, 0.03891, 0.05818]
soc = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
r1 = []
r2 = []
c1 = []
c2 = []
for i in range(len(k1)):
    r1.append(calculate(k1[i], b1[i], k2[i], b2[i])[0])
    r2.append(calculate(k1[i], b1[i], k2[i], b2[i])[1])
    c1.append(calculate(k1[i], b1[i], k2[i], b2[i])[2])
    c2.append(calculate(k1[i], b1[i], k2[i], b2[i])[3])

print(r1)
print(r2)
print(c1)
print(c2)

def fitting(ls):
    line = np.polyfit(soc, ls, 7)
    para = np.poly1d(line)
    # print(para[0], para[1])
    res = list(map(lambda x:
               para[7]*x**7+para[6]*x**6+para[5]*x**5+para[4]*x**4+para[3]*x**3+para[2]*x**2+para[1]*x+para[0],
               soc))
    return para

'''
plt.figure(1)
plt.style.use("ggplot")
plt.subplot(2,2,1)
plt.title('Relationship between soc and C1')
plt.scatter(soc, r1, edgecolors='b')
plt.plot(soc, fitting(r1), 'r')
plt.legend(['Real result', 'fitting curve'])

plt.subplot(2,2,2)
plt.title('Relationship between soc and C2')
plt.scatter(soc, r2, edgecolors='r')
plt.plot(soc, fitting(r2), 'b')
plt.legend(['Real result', 'fitting curve'])

plt.subplot(2,2,3)
plt.title('Relationship between soc and R1')
plt.scatter(soc, c1, edgecolors='gray')
plt.plot(soc, fitting(c1), 'g')
plt.legend(['Real result', 'fitting curve'])

plt.subplot(2,2,4)
plt.title('Relationship between soc and R2')
plt.scatter(soc, c2, edgecolors='g')
plt.plot(soc, fitting(c2), 'gray')
plt.legend(['Real result', 'fitting curve'])
plt.show()
'''

print("C1:", fitting(r1))
print("C2:", fitting(r2))
print("R1:", fitting(c1))
print("R2:", fitting(c2))