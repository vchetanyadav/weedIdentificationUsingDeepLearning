# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 17:43:29 2024

@author: RAVITEJA S
"""

#pip install ultralytics
import pandas as pd
import numpy as np
from ultralytics import YOLO
import torch

import shutil
import os

import random

from PIL import Image
import cv2
import matplotlib.pyplot as plt
import seaborn as sns


model = YOLO('C:/Code/Centrenet/Weed/weights/best.pt ')
device= 'cpu'

metrics = model.val(split='test', conf=0.25, device=device) 

print(metrics)

ax = sns.barplot(x=['mAP50-95', 'mAP50', 'mAP75'], y=[metrics.box.map, metrics.box.map50, metrics.box.map75])


ax.set_title('Centrenet Evaluation Metrics')
ax.set_xlabel('Metric')
ax.set_ylabel('Value')


fig = plt.gcf()
fig.set_size_inches(8, 6)

for p in ax.patches:
    ax.annotate('{:.3f}'.format(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()), ha='center', va='bottom')
    

plt.show()

precision = metrics.results_dict['metrics/precision(B)']
recall = metrics.results_dict['metrics/recall(B)']
f1 = (2 * precision * recall) / (precision + recall)  # Вычисление F1


metrics = ['Precision', 'Recall', 'F1']
values = [precision, recall, f1]


ax = sns.barplot(x=metrics, y=values, palette='viridis')

ax.set_title('Precision, Recall, and F1 Scores')
ax.set_xlabel('Metric')
ax.set_ylabel('Value')
for p in ax.patches:
    ax.annotate('{:.3f}'.format(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()), ha='center', va='bottom')

plt.show()
images_folder = "C:/Code/testimages"
image_paths = random.sample(os.listdir(images_folder), 10)

fig, axes = plt.subplots(2, 5, figsize=(20, 12))
fig.tight_layout()


for i, ax in enumerate(axes.flat):
    image_path = os.path.join(images_folder, image_paths[i])
    image = Image.open(image_path) 
    res = model(image, verbose=False)
    detect_img = res[0].plot()
    ax.imshow(detect_img)
    ax.axis('off')
    detect_img = cv2.cvtColor(detect_img, cv2.COLOR_BGR2RGB)
    
plt.show()
'''
#image_path = os.path.join(images_folder, image_paths[i])
image = Image.open(image_path) 
res = model(image, verbose=False)
detect_img = res[0].plot()
detect_img = cv2.cvtColor(detect_img, cv2.COLOR_BGR2RGB)
ax.imshow(detect_img)
ax.axis('off')


plt.show()
'''
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
  