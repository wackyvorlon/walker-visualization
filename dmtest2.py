# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 13:46:30 2018

@author: wacky
"""

from dm_control import suite
import matplotlib.pyplot as plt
import numpy as np
import skvideo
import skvideo.io


skvideo.setFFmpegPath("c:\\Users\\wacky\\Downloads\\ffmpeg\\bin")

max_frame = 320

width = 480
height = 480
video = np.zeros((max_frame, height, 2 * width, 3), dtype=np.uint8)

# Load one task:
env = suite.load(domain_name="humanoid", task_name="walk")

# Step through an episode and print out reward, discount and observation.
action_spec = env.action_spec()
time_step = env.reset()
j=0
while not time_step.last():
  for i in range(max_frame):
    action = np.random.uniform(action_spec.minimum,
                             action_spec.maximum,
                             size=action_spec.shape)
    time_step = env.step(action)
    video[i] = np.hstack([env.physics.render(height, width, camera_id=0),
                          env.physics.render(height, width, camera_id=1)])
    print(time_step.reward, time_step.discount, time_step.observation)
  
  if (j%10)==0:
    fname="output-vids/outputvideo"+str(j)+".mp4"
    #j+=1
    skvideo.io.vwrite(fname, video)
  j+=1
  print(j)
  #for i in range(max_frame):
   # img = plt.imshow(video[i])
    #plt.pause(0.01)  # Need min display time > 0.0.
    #lt.draw() 
   
