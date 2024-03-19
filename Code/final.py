import opyf
import matplotlib.pyplot as plt
import sys
import os
from WhatIfAnalysis import GoalSeek
import numpy as np
import pandas as pd

video=opyf.videoAnalyzer('IMG_1139.MOV')
video.set_vecTime(Ntot=10,starting_frame=100)

video.set_interpolationParams(Sharpness=2)
video.set_goodFeaturesToTrackParams(qualityLevel=0.01)

video.set_vlim([0, 20])
video.extractGoodFeaturesDisplacementsAccumulateAndInterpolate(display1='quiver',display2='field',displayColor=True)
video.set_filtersParams(maxDevInRadius=1.5, RadiusF=0.15,range_Vx=[0.01,10])
video.filterAndInterpolate()
video.writeVelocityField(fileFormat='csv')

df = pd.read_csv("frame_100_to_110_with_step_1_and_shift_1.csv")
df_transformed = df[(df['X']>250) & (df['X']<1500) &(df['Y']==933)]

Velocity = df_transformed['Uy_[px.deltaT^{-1}]']
print(type(Velocity.values))

def generate_D(x):
    something = np.abs(np.random.normal(0.8,0.5,int((x+1)/2)))
    something = np.concatenate((np.sort(something),np.sort(something)[::-1]),axis = None)
    if(len(something)>x):
        something = something[:-1]
    return something

U_surface = np.absolute(Velocity.values)
data_len = len(U_surface)
x = np.linspace(0.1,5,len(U_surface))
D = np.linspace(0.1,5,len(U_surface))
# D = generate_D(data_len)
y = np.linspace(0.1,1,100)

area_of_one_point = 5 / (data_len * 100)

print(x[0:5])
print(area_of_one_point)
print(D[0:5])

phi_Mobs = np.mean(U_surface) / np.max(U_surface)
U_max = np.max(U_surface)
U_mean = np.mean(U_surface)

def find_M():
    def fun(x):
        y = np.exp(x) / (np.exp(x)-1)
        y = y - 1/x
        return y

    goal=U_mean / U_max

    # (iii) Define a starting point
    x0=0.001

    ## Here is the result
    ans = GoalSeek(fun,goal,x0)
#     print('M is = ', ans)
    return ans

def implement_flow_chart():
    p = 1
    a = 0.05
    u_xy = []
    
    delta_dash = a + 1 + 1.3 * np.exp(-x/D)
    M = find_M()

    # h missing in flow-chart
    h = D - D/delta_dash

    u_maxv = U_surface / (1/M * np.log(1 + (np.exp(M)-1)* delta_dash * np.exp(1-delta_dash)))

    for j in range(len(y)):
        temp = u_maxv/M * np.log(1+ (np.exp(M) -1)*y[j]*np.exp(1- y[j]/(D-h)))
        u_xy.append(temp)


    phi_Mp = np.mean(u_maxv) / np.max(u_maxv)

    p = p+1
    a = a + 1
    
    while not(phi_Mp-phi_Mobs <0.03):
        delta_dash = a + 1 + 1.3 * np.exp(-x/D)
        M = find_M()

        # h missing in flow-chart
        h = D - D/delta_dash

        u_maxv = U_surface / (1/M * np.log(1 + (np.exp(M)-1)* delta_dash * np.exp(1-delta_dash)))

        for j in range(len(y)):
            temp = u_maxv/M * np.log(1+ (np.exp(M) -1)*y[j]*np.exp(1- y[j]/(D-h)))
            u_xy.append(temp)

        phi_Mp = np.mean(u_maxv) / np.max(u_maxv)

        p = p+1
        a = a + 1
    
    u_xy = np.array(u_xy)
    return u_xy


ans = implement_flow_chart()
ans = ans * area_of_one_point
print( np.sum(ans))