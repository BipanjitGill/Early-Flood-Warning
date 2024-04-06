import sys
sys.path.append("opyflow/src/")
sys.path.append("./")
import bson
import opyf
import matplotlib.pyplot as plt
import sys
import os
from WhatIfAnalysis import GoalSeek
import numpy as np
import pandas as pd
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from datetime import datetime
import time
import subprocess
from datetime import datetime
import base64
import cv2

def capture_photo(video_path, output_path):
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        print("Error: Unable to open video file.")
        return
    ret, frame = video_capture.read()
    if not ret:
        print("Error: Unable to read frame from video.")
        return
    cv2.imwrite(output_path, frame)
    video_capture.release()

    print("Photo captured successfully!")


def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read())
        return encoded_string.decode('utf-8')
uri = "mongodb+srv://dp42:1234@cluster0.fcqtdbe.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(uri, server_api=ServerApi('1'))

try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

db=client.cluster0
collection=db.outputs


data = {
    "IMG": "image",
    "VEL": 2.2,
    "TIME": datetime.now() 
}
#     collection.insert_one(data)

def delete_file(filename):
    if os.path.exists(filename):
        os.remove(filename)
        print(f"{filename} has been deleted.")
    else:
        print(f"{filename} does not exist.")
def record_video(filename, duration_ms=10000, width=1280, height=720):
    subprocess.run("rpicam-vid -t 5000 -o test.h264", shell=True)
    time.sleep(1)
    subprocess.run("ffmpeg -r 30 -i test.h264 -c:v copy output.mp4", shell=True)
	


def find_M(U_mean , U_max):
    def fun(x):
        y = np.exp(x) / (np.exp(x)-1)
        y = y - 1/x
        return y
    goal=U_mean / U_max
    x0=0.001

    ans = GoalSeek(fun,goal,x0)
    return ans

def implement_flow_chart(U_surface):
    x = np.linspace(0.1,5,len(U_surface))
    D = np.linspace(0.1,5,len(U_surface))
    y = np.linspace(0.1,1,100)
    phi_Mobs = np.mean(U_surface) / np.max(U_surface)
    U_max = np.max(U_surface)
    U_mean = np.mean(U_surface)

    p = 1
    a = 0.05
    u_xy = []
    
    delta_dash = a + 1 + 1.3 * np.exp(-x/D)
    M = find_M(U_mean , U_max)

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


record_video("output.MOV", duration_ms=10000, width=1280, height=720)

video=opyf.videoAnalyzer("output.mp4")
video.set_vecTime(Ntot=5,starting_frame=10)

video.set_interpolationParams(Sharpness=2)
video.set_goodFeaturesToTrackParams(qualityLevel=0.01)

video.set_vlim([0, 20])
video.extractGoodFeaturesDisplacementsAccumulateAndInterpolate(display1='quiver',display2='field',displayColor=True)
video.set_filtersParams(maxDevInRadius=1.5, RadiusF=0.15,range_Vx=[0.01,10])
video.filterAndInterpolate()
video.writeVelocityField(fileFormat='csv')

df = pd.read_csv(video.filename+".csv")
df_transformed = df[(df['X']>50) & (df['X']<500) &(df['Y']==433)]

Velocity = df_transformed['Uy_[px.deltaT^{-1}]']
U_surface = np.absolute(Velocity.values)
ans = implement_flow_chart(U_surface)
area_of_one_point = 5 / (len(U_surface) * 100)
ans = ans * area_of_one_point
# capture_photo("output.mp4","output.jpg")

print( np.sum(ans))
        
