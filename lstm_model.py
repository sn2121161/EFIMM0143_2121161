# coding:utf-8
# 先做数据处理，把所有放电阶段的数据合并起来，做归一化
import ssl

import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import numpy as np
import torchvision
from torch.utils.data import DataLoader
from datetime import datetime  # 用于计算时间
from matplotlib import pyplot as plt



'''
# concatenate data
df = pd.DataFrame([])
for i in range(1, 4):
    da = pd.read_excel('D:\\python_project\\application-research-project\\12_2_2015_Incremental OCV test_SP20-1.xlsx', sheet_name = 'Channel_1-005_%d'%(i))
    df = df.append(da)
df = df.reset_index()


data = pd.concat([
    df[['Test_Time(s)', 'Current(A)', 'Voltage(V)']].iloc[772:1485],
    df[['Test_Time(s)', 'Current(A)', 'Voltage(V)']].iloc[8611:9324],
    df[['Test_Time(s)', 'Current(A)', 'Voltage(V)']].iloc[16450:17163],
    df[['Test_Time(s)', 'Current(A)', 'Voltage(V)']].iloc[24289:25002],
    df[['Test_Time(s)', 'Current(A)', 'Voltage(V)']].iloc[32129:32842],
    df[['Test_Time(s)', 'Current(A)', 'Voltage(V)']].iloc[39969:40682],
    df[['Test_Time(s)', 'Current(A)', 'Voltage(V)']].iloc[47808:48521],
    df[['Test_Time(s)', 'Current(A)', 'Voltage(V)']].iloc[55647:56360],
    df[['Test_Time(s)', 'Current(A)', 'Voltage(V)']].iloc[63485:64199],
    df[['Test_Time(s)', 'Current(A)', 'Voltage(V)']].iloc[71325:71994]
])
data.reset_index()
data.columns = ['time', 'current', 'voltage']
data.to_excel('D:\\python_project\\application-research-project\\discharge_data.xlsx')
print("write in successfully")
'''
'''

data = pd.read_excel('D:\\python_project\\application-research-project\\discharge_data.xlsx', sheet_name="Sheet1")
data['current'].apply(lambda x: -1*x)

data['current_norm'] = data['current'].apply(lambda x: (x - min(data['current']))/(max(data['current']) - min(data['current'])))
data['voltage_norm'] = data['voltage'].apply(lambda x: (x - min(data['voltage']))/(max(data['voltage']) - min(data['voltage'])))

data.to_excel('D:\\python_project\\application-research-project\\discharge_norm.xlsx')
print('success')
'''


# data preprocessing
data = pd.read_excel('D:\\pythonProject\\dessertation\\discharge_norm.xlsx', sheet_name='Sheet1')

# four datasets settings
train, test = train_test_split(data[['current_norm', 'voltage_norm', 'time', 'soc']], train_size=0.7)
train_set = np.array([train[['current_norm', 'voltage_norm', 'time']]])
train_label = np.array([train['soc']])
test_set = np.array([test[['current_norm', 'voltage_norm', 'time']]])
test_label = np.array([test['soc']])

def trainProcess(dataset):
    train_data = torch.FloatTensor(np.array([dataset])).view(-1)

    train_window = 3

    def create_inout_sequences(input_data, tw):
        inout_seq = []
        L = len(input_data)
        for i in range(0,L-tw,4):
            train_seq = input_data[i:i+tw]
            train_label = input_data[i+tw:i+tw+1]
            inout_seq.append((train_seq ,train_label))
        return inout_seq

    inout_seq = create_inout_sequences(train_data, train_window)
    return inout_seq

train_inout_seq = trainProcess(train)
# 调用函数让test也跑一遍


class LSTM(nn.Module):
    def __init__(self,input_size=1,hidden_layer_size=30,output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size,hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size,output_size)
        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),torch.zeros(1,1,self.hidden_layer_size))

    def forward(self,input_seq):
        lstm_out,self.hidden_cell = self.lstm(input_seq.view(len(input_seq),1,-1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

model = LSTM()  # change the configuration
loss_function = nn.MSELoss()  # 交叉熵损失函数，一种均方误差

optimizer = torch.optim.Adam(model.parameters(), lr = 0.001) # 优化器选adam，优化器的学习率选择
# optimizer = torch.optim.SGD(model.parameters(), lr = 0.001)
# 等会论文里再加上优化器的部分

losses = []  # result
# train model
epochs = 1

for i in range(epochs):
    for seq, labels in train_inout_seq:
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))

        y_pred = model(seq)

        single_loss = loss_function(y_pred, labels)
        single_loss.backward()
        optimizer.step()

    # print(f'epoch: {i:3} loss: {single_loss.item():10.12f}')
    # losses.append(single_loss.item())

# print(losses)

# 训练集误差画图
'''
i = [i for i in range(0, 150-5)]
j = [309.2146301269531, 351.13018798828125, 209.4216766357422, 81.26155090332031, 22.02980613708496, 6.606820106506348, 5.446699619293213, 1.0811258554458618, 0.0001400244073010981, 1.509325623512268, 1.674432635307312, 1.8968614339828491, 2.055628538131714, 2.172830581665039, 2.8665359020233154, 2.0899648666381836, 1.748810052871704, 3.4852781295776367, 2.874032735824585, 0.08229166269302368, 1.393823504447937, 2.0137739181518555, 0.6778533458709717, 2.7970285415649414, 3.0883493423461914, 2.06805157661438, 3.352426052093506, 2.4456281661987305, 2.2877392768859863, 2.731046676635742, 3.18158221244812, 0.3854447603225708, 1.9300203323364258, 2.2480320930480957, 1.1782828569412231, 1.667588233947754, 4.5003981590271, 2.3929402828216553, 0.3669620454311371, 0.6355205774307251, 3.2049500942230225, 1.6070057153701782, 0.7809078097343445, 1.4149620532989502, 1.209342122077942, 1.0078277587890625, 1.0188871622085571, 0.9926131963729858, 0.9605262279510498, 0.847253143787384, 0.8333219289779663, 1.409286618232727, 0.6654233932495117, 0.47951018810272217, 0.4807789623737335, 0.4560927152633667, 1.0755325555801392, 1.2806036472320557, 1.2083523273468018, 1.3932651281356812, 0.8246111869812012, 0.820722222328186, 0.9121013879776001, 1.1099852323532104, 1.0071384906768799, 0.6821566224098206, 0.6536141037940979, 0.47412604093551636, 0.48510536551475525, 0.46959802508354187, 0.4289734363555908, 0.4508318305015564, 0.42038214206695557, 0.4229682683944702, 0.44133371114730835, 0.4948462247848511, 0.3965024948120117, 0.3594765067100525, 0.3619784116744995, 0.33731430768966675, 0.32869407534599304, 0.29775527119636536, 0.29535382986068726, 0.2684876620769501, 0.27124616503715515, 0.2688594162464142, 0.2465861439704895, 0.23337173461914062, 0.2261529564857483, 0.22087962925434113, 0.2115710824728012, 0.20622208714485168, 0.995442807674408, 0.922741174697876, 0.1836099475622177, 0.15040823817253113, 0.12316205352544785, 0.1241816058754921, 0.2295178323984146, 0.042427100241184235, 0.1498820185661316, 0.13056904077529907, 0.11336205154657364, 0.12547005712985992, 0.04102145507931709, 0.03489352762699127, 0.032304052263498306, 0.049582600593566895, 0.04097819700837135, 0.08067572861909866, 0.3674890995025635, 0.26363155245780945, 0.2227553129196167, 0.12548087537288666, 0.0844726487994194, 0.07030557096004486, 0.08632313460111618, 0.09769439697265625, 0.06336893886327744, 0.06148562580347061, 0.09623570740222931, 0.12544843554496765, 0.0597686693072319, 0.07035413384437561, 0.005534500814974308, 0.031546130776405334, 0.0392092689871788, 0.06346884369850159, 0.042392533272504807, 0.042969413101673126, 0.025547487661242485, 0.5678570866584778, 0.002291232580319047, 0.3097213804721832, 0.35353636741638184, 0.10618910193443298, 0.258245587348938, 0.03796836733818054, 0.02199067734181881, 0.022026896476745605, 0.014547519385814667, 0.0007401893381029367, 0.00017582462169229984, 5.820766091346741e-11, 0.008107582107186317, 0.0024035966489464045, 0.0012918328866362572, 0.0005536163225769997, 0.0002330635325051844, 7.940828800201416e-05]
plt.figure(1)
plt.style.use("ggplot")
plt.plot(i, j[5:], 'g')
plt.title("MSE change in each epoch")
plt.xlabel("epochs")
plt.ylabel("mean square error")
plt.show()
'''
# test set
# 首先还是把testset变成train的形式

test_input = trainProcess(test)

model.eval()

test_res = []
test_l = []
for seq, label in test_input:
    with torch.no_grad():
        model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))
        # print(seq, '-----', label, '------', model(seq).item())
        test_l.append(label.item())
        test_res.append(model(seq).item())

i = [i for i in range(0, len(test_res))]
plt.style.use("ggplot")
plt.subplots(figsize=(8,6), dpi=100)
plt.scatter(i, test_l)
plt.scatter(i, test_res)
plt.title("MSE change in each epoch")
plt.legend(['real_res', 'predicted_res'])
plt.xlabel("epochs")
plt.ylabel("mean square error")
plt.show()