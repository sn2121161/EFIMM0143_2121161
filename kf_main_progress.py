# coding:utf-8

import numpy as np
from kf_detail import EKF_UKF as sim

def main(Work_mode, SoC_est_init):
    if nargin == 0:  #  Set parameter by default
        Work_mode = 1
        SoC_est_init = 1
    elif nargin == 1:
        SoC_est_init = 1

    if Work_mode == 1:
        # sim BBDST_workingcondition
        I = -np.linalg.inv(current.data) * 1.5 / 50
    elif Work_mode == 2:
        N = 60001
        I = 1.5 * np.ones(1, N)
        I[np.ceil(N / 5) : np.ceil[N * 3 / 9]] = 0
        I[np.ceil(N * 5 / 9) : np.ceil(N * 4 / 5)] = 0
    else:
        print("Input error!")
        print("Work_mode: Mode of working condition")
        print("           1 --> BBDST, 2 --> constant current ")
        print("SOC_est_init : The initial value of estimated SOC")
        return

SoC_est_init = 1
I = np.array([1,1,1,1,1])
Work_mode = 1
ss = sim()
avr_err_EKF, std_err_EKF, SoC_real, SoC_AH = ss.EKF_UKF_Thev(SoC_est_init, I)
print('Initial SOC : # f\nWorking Mode: # d\n', SoC_est_init, Work_mode)
print("avr_err_EKF --> # f\n", )
print("standard_err_EKF --> # f\n", std_err_EKF)
# print("avr_err_UKF --> # f\n", avr_err_UKF)
# print("standard_err_UKF --> # f\n", std_err_UKF)
