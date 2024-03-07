#!/usr/bin/env python
# coding: utf-8

# In[1]:


import opyf
import matplotlib.pyplot as plt
import sys
import os


# In[9]:


video=opyf.videoAnalyzer('IMG_1139.MOV')
video.set_vecTime(Ntot=10,starting_frame=100)

video.set_interpolationParams(Sharpness=2)
video.set_goodFeaturesToTrackParams(qualityLevel=0.01)

video.set_vlim([0, 20])

video.extractGoodFeaturesDisplacementsAccumulateAndInterpolate(display1='quiver',display2='field',displayColor=True)
video.set_filtersParams(maxDevInRadius=1.5, RadiusF=0.15,range_Vx=[0.01,10])

video.filterAndInterpolate()

# video.writeVelocityField(fileFormat='csv')


# In[ ]:




