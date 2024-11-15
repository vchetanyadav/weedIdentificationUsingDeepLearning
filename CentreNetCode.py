# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 07:17:06 2024

@author: user
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

source_folder = "WeedCrop"


test_source_images = os.path.join(source_folder, "test/images")
test_source_labels = os.path.join(source_folder, "test/labels")

test_destination_folder = "working/test"
#os.makedirs(test_destination_folder, exist_ok=True)


#shutil.copytree(test_source_images, os.path.join(test_destination_folder, "images"))


#shutil.copytree(test_source_labels, os.path.join(test_destination_folder, "labels"))


train_source_images = os.path.join(source_folder, "train/images")
train_source_labels = os.path.join(source_folder, "train/labels")

train_destination_folder = "working/train"
#os.makedirs(train_destination_folder, exist_ok=True)


#shutil.copytree(train_source_images, os.path.join(train_destination_folder, "images"))


#shutil.copytree(train_source_labels, os.path.join(train_destination_folder, "labels"))




valid_source_images = os.path.join(source_folder, "valid/images")
valid_source_labels = os.path.join(source_folder, "valid/labels")

valid_destination_folder = "working/valid"
#os.makedirs(valid_destination_folder, exist_ok=True)


#shutil.copytree(valid_source_images, os.path.join(valid_destination_folder, "images"))


#shutil.copytree(valid_source_labels, os.path.join(valid_destination_folder, "labels"))




def move_files(file_path, source_folder, destination_folder):
    
    with open(file_path, "r") as file:
        file_names = [os.path.basename(line.strip()) for line in file.readlines()]

    
    images_folder = os.path.join(destination_folder, "images")
    labels_folder = os.path.join(destination_folder, "labels")
    
    
    for file_name in file_names:
        image_file = os.path.join(source_folder, file_name)
        label_file = os.path.join(source_folder, file_name.replace(".png", ".txt"))
        if os.path.isfile(image_file):
            shutil.copy(image_file, images_folder)
        if os.path.isfile(label_file):
            shutil.copy(label_file, labels_folder)


train_file = "C:/Users/RAVITEJA S/Downloads/all_fields_lincolnbeet/all_fields_lincolnbeet/all_fields_lincolnbeet_train_.txt"
valid_file = "C:/Users/RAVITEJA S/Downloads/all_fields_lincolnbeet/all_fields_lincolnbeet/all_fields_lincolnbeet_val_.txt"
test_file = "C:/Users/RAVITEJA S/Downloads/all_fields_lincolnbeet/all_fields_lincolnbeet/all_fields_lincolnbeet_test_.txt"

source_folder = "C:/Users/RAVITEJA S/Downloads/all_fields_lincolnbeet/all_fields_lincolnbeet/all"
train_destination = "working/train"
valid_destination = "working/valid"
test_destination = "working/test"


#move_files(train_file, source_folder, train_destination)
#print("Moving of Train Files to Destination")
#move_files(valid_file, source_folder, valid_destination)
#print("Moving of Valid Files to Destination")
#move_files(test_file, source_folder, test_destination)
#print("Moving of Test Files to Destination")


import  yaml

# Data structure
dataset = {
'train': 'working/train',
'val': 'working/valid',
'test': 'working/test',
'nc': 2,
'names': ['crop', 'weed']
}

# save to YAML-file
with open('working/dataset.yaml', 'w') as file:
    yaml.dump(dataset, file)
    
print("load yaml")  
    
images_folder = "working/train/images"
labels_folder = "working/train/labels"


image_files = os.listdir(images_folder)


random.shuffle(image_files)
random_image_files = image_files[:6]


num_images = len(random_image_files)
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i in range(num_images):
    
    image_file = os.path.join(images_folder, random_image_files[i])
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    
    label_file = os.path.join(labels_folder, os.path.splitext(random_image_files[i])[0] + ".txt")
    with open(label_file, "r") as file:
        labels = file.readlines()

    
    for label in labels:
        class_id, x, y, width, height = map(float, label.strip().split())
        x = int(x * image.shape[1])
        y = int(y * image.shape[0])
        width = int(width * image.shape[1])
        height = int(height * image.shape[0])
        cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)

    
    axes[i].imshow(image)
    axes[i].axis("off")

plt.tight_layout()
plt.show()


#device = 'cuda' if torch.cuda.is_available() else 'cpu'
device ='cpu'
print("Device Name=======================================")
print(device)


centrenetmodel = YOLO('yolov8x.pt')
print("Loaded Initial Yolo Weights")

centrenetmodel.train(data='working/dataset.yaml ', epochs=1, imgsz=640,
            optimizer = 'AdamW', lr0 = 1e-3, 
            project = 'Centrenet', name='Weed',
            batch=16, device=device, seed=69)

print("Build Centrenet Model")


df = pd.read_csv('C:/Code/Centrenet/Weed/results.csv')

print(df.columns)

fig, axes = plt.subplots(3, 2, figsize=(15, 15))
fig.tight_layout()

# train/box_loss
axes[0, 0].plot(df['                  epoch'], df['         train/box_loss'], label='         train/box_loss')
axes[0, 0].set_title('Train Box Loss')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()

# val/box_loss
axes[0, 1].plot(df['                  epoch'], df['           val/box_loss'], label='           val/box_loss')
axes[0, 1].set_title('Validation Box Loss')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()



# train/cls_loss
axes[1, 0].plot(df['                  epoch'], df['         train/cls_loss'], label='         train/cls_loss')
axes[1, 0].set_title('Train Class Loss')
axes[1, 0].set_ylabel('Loss')
axes[1, 0].legend()

# val/cls_loss
axes[1, 1].plot(df['                  epoch'], df['           val/cls_loss'], label='           val/cls_loss')
axes[1, 1].set_title('Validation Class Loss')
axes[1, 1].set_ylabel('Loss')
axes[1, 1].legend()

# train/dfl_loss
axes[2, 0].plot(df['                  epoch'], df['         train/dfl_loss'], label='         train/dfl_loss')
axes[2, 0].set_title('Train Distribution Focal loss')
axes[2, 0].set_xlabel('Epoch')
axes[2, 0].set_ylabel('Loss')
axes[2, 0].legend()

# val/dfl_loss
axes[2, 1].plot(df['                  epoch'], df['           val/dfl_loss'], label='           val/dfl_loss')
axes[2, 1].set_title('Validation Distribution Focal loss')
axes[2, 1].set_xlabel('Epoch')
axes[2, 1].set_ylabel('Loss')
axes[2, 1].legend()

plt.show()

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.tight_layout()

# metrics/precision(B)
axes[0, 0].plot(df['                  epoch'], df['   metrics/precision(B)'], label='   metrics/precision(B)')
axes[0, 0].set_title('Precision')
axes[0, 0].set_ylabel('Precision')
axes[0, 0].legend()

# metrics/recall(B)
axes[0, 1].plot(df['                  epoch'], df['      metrics/recall(B)'], label='      metrics/recall(B)')
axes[0, 1].set_title('Recall')
axes[0, 1].set_ylabel('Recall')
axes[0, 1].legend()

axes[1, 0].plot(df['                  epoch'], df['       metrics/mAP50(B)'], label='       metrics/mAP50(B)')
axes[1, 0].set_title('mAP@0.5')
axes[1, 0].set_ylabel('mAP@0.5')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].legend()

# metrics/mAP50-95(B)
axes[1, 1].plot(df['                  epoch'], df['    metrics/mAP50-95(B)'], label='    metrics/mAP50-95(B)')
axes[1, 1].set_title('mAP@0.5:0.95')
axes[1, 1].set_ylabel('mAP@0.5:0.95')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].legend()

plt.show()



# F1_curve.png
f1_curve = Image.open("C:/Code/Centrenet/Weed/F1_curve.png")
plt.figure(figsize=(10, 10))
plt.imshow(f1_curve)
plt.title("F1 Curve")
plt.axis("off")
plt.show()

# PR_curve.png
pr_curve = Image.open("C:/Code/Centrenet/Weed/PR_curve.png")
plt.figure(figsize=(10, 10))
plt.imshow(pr_curve)
plt.title("Precision-Recall Curve")
plt.axis("off")
plt.show()



# P_curve.png
p_curve = Image.open("C:/Code/Centrenet/Weed/P_curve.png")
plt.figure(figsize=(10, 10))
plt.imshow(p_curve)
plt.title("Precision Curve")
plt.axis("off")
plt.show()

# R_curve.png
r_curve = Image.open("C:/Code/Centrenet/Weed/R_curve.png")
plt.figure(figsize=(10, 10))
plt.imshow(r_curve)
plt.title("Recall Curve")
plt.axis("off")
plt.show()


# confusion matrix
confusion_matrix = Image.open("C:/Code/Centrenet/Weed/confusion_matrix.png")
plt.figure(figsize=(8, 6))
plt.imshow(confusion_matrix)
plt.title("Confusion Matrix")
plt.axis("off")
plt.show()


res = centrenetmodel('working/test/images/bbro_bbro_14_05_2021_v_0_18.png')
detect_img = res[0].plot()
detect_img = cv2.cvtColor(detect_img, cv2.COLOR_BGR2RGB)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))


axes[0].imshow(plt.imread('working/test/images/bbro_bbro_14_05_2021_v_0_18.png'))
axes[0].axis('off')


axes[1].imshow(detect_img)
axes[1].axis('off')

plt.show();


model = YOLO('C:/Code/Centrenet/Weed/weights/best.pt ')


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

image_paths = random.sample(os.listdir(images_folder), 10)


fig, axes = plt.subplots(2, 5, figsize=(20, 12))
fig.tight_layout()


for i, ax in enumerate(axes.flat):
    image_path = os.path.join(images_folder, image_paths[i])
    image = Image.open(image_path) 
    res = model(image, verbose=False)
    detect_img = res[0].plot()
    detect_img = cv2.cvtColor(detect_img, cv2.COLOR_BGR2RGB)
    
    ax.imshow(detect_img)
    ax.axis('off')
plt.show()
