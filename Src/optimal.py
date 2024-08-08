import numpy as np
import pandas as pd
from cvxopt import matrix, solvers, blas
from matplotlib import pyplot as plt



# Define IC
h_init ,v_init = 0 , 0
# Final state
h_final,v_final =10, 3
# Boundary conditions
h_min , h_max = 0, 10
N_h = 100
v_min, v_max =0, 3
N_v = 100
interval = (h_max - h_min) / N_h
# Create state array
Hd = np.linspace(h_min, h_max, N_h+1)   # Discretized state space np.linspace(start, stop, num)
Vd = np.linspace(v_min, v_max, N_v+1)
# Input constraints
u_min = -2
u_max = 2
# Define cost to go matrix
J_costtogo = np.zeros((N_h+1, N_v+1))
# Define control matrix
u = np.zeros((N_h+1, N_v+1))

## 10m ##
# final state # h=10m v=0

## From 10m to 8m   ## 8m
v_avg = 0.5*(v_final + Vd)     # Average velocity   from 10m v10 to 8m v8 v10 = v_final
T_delta = interval / v_avg   # Time step
acc = (v_final - Vd) / T_delta        # Control input
J_temp = T_delta                    # Cost to go
acc_idx = np.where((acc > u_max) | (acc < u_min))   # Find the index of the acceleration that is within the constraints
J_temp[acc_idx] = np.inf        # Set the cost to go of the acceleration that is outside the constraints to inf
J_costtogo[1][:] = J_temp      # Update the cost to go matrix
u[1][:] = acc    # Update the control matrix

## From 8m to 2m   ## 6m -- 2m
for k in range(2,N_h):
    vd_x, vd_y = np.meshgrid(Vd,Vd)    # Create a meshgrid of the velocity
    v_avg = 0.5*(vd_x + vd_y)        # Average velocity # 第一排 就是6m到8m的平均速度
    T_delta = interval / v_avg    # Time step
    acc = (vd_x - vd_y) / T_delta       # Control input
    J_temp = T_delta                  # Cost to go
    x, y = np.where((acc > u_max) | (acc < u_min))    # Find the index 
    for ix, iy in zip(x, y):
        J_temp[ix][iy] = np.inf        # Set the cost outside the constraints to inf
    J_temp = J_temp + np.meshgrid(J_costtogo[k-1][:])   # Add the cost to go of the previous state
    J_costtogo[k][:] = np.min(J_temp, axis=1)    # Update the cost to go matrix
    u[k][:] = acc[np.arange(acc.shape[0]),np.argmin(J_temp, axis=1)]      # Update the control matrix

## From 2m to 0m   ## 0 m
v_avg = 0.5*(Vd + v_init)     # Average velocity
T_delta = interval /v_avg    # Time step
acc = (Vd - v_init) / T_delta       # Control input
J_temp = T_delta                  # Cost to go
acc_idx = np.where((acc > u_max) | (acc < u_min))   # Find the index of the acceleration that is within the constraints
J_temp[acc_idx] = np.inf        # Set the cost to go of the acceleration that is outside the constraints to inf
J_temp =  J_temp + J_costtogo[N_h-1][:]   # Add the cost to go of the previous state
J_costtogo[N_h][0] = min(J_temp)    # Update the cost to go matrix
u[N_h][0] = acc[np.argmin(J_temp)]    # Update the control matrix



def findidx(cost,control):
    n = len(cost)       # 离散的高度层
    ls = [0]*n          # 存储每层最低cost的下标
    ls1 = [0]           # 存储每层最低的cost
    for i in range(1,n-1):  # 最低层 最高层 不用存 最高 速度为0 最低 速度为0
        ls1.append(min(cost[i]))
        ls[i] = cost[i].index(ls1[-1])
    ls1.append(cost[-1][0])         # 最后一层的cost 速度为0
    ls2 = []
    # 每层的加速度控制
    j = 0
    for arr in control:
        ls2.append(arr[ls[j]])
        j += 1
    return ls,ls1,ls2


l0,l1,l2 = findidx(J_costtogo.tolist(),u.tolist())
# Plot the cost to go matrix

fig, ax = plt.subplots()


# 绘制起始位置和终点位置 写大一点
ax.text(0, 0, 'Start', fontsize=12)
ax.text(3, 10, 'End', fontsize=12)

# 绘制最优点的连线
optimal_speeds = [Vd[idx] for idx in l0]
ax.plot(optimal_speeds, Hd, color='red', marker='o', label='Optimal Path')

# 标记最优点和对应的控制点
for i in range(len(Hd)):
    ax.annotate(f'Cost: {l1[i]:.2f}\nControl: {l2[i]:.2f}', 
                (optimal_speeds[i], Hd[i]), 
                textcoords="offset points", 
                xytext=(0,10), 
                ha='center')

# 显示最优加速度的变化
l2 = np.array(l2)
l2 = l2.reshape(-1,1)
ax.plot(Vd,l2, color='blue', marker='o', label='Optimal Control')


# 设置标签和标题
ax.set_xlabel('速度 (m/s)')

plt.show()


        
        


