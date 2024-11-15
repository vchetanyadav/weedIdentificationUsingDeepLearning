from tkinter import *
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk  # pip install pillow
from tkinter.filedialog import askopenfilename
import cv2 # pip install opencv-python
from tkinter import filedialog
import cv2
import os
import numpy as np
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




window = tk.Tk()

window.title("Weed Detection")

window.configure(background= "white")

window.geometry('1600x800')

window.grid_rowconfigure(0, weight = 1)
window.grid_columnconfigure(0, weight = 1)

load = Image.open("wallpaper.jpg")
photo = ImageTk.PhotoImage(load)
label = tk.Label(window, image=photo)
label.image=photo
label.place(x=0,y=0)
lb = tk.Label(window, text = "Weed Identification Using Deep Learning In Vegetable Farming", bg = 'black', fg = 'white', font = ('times', 30, 'bold'))
lb.place(x=230,y=10)

def showImgg():
    global load
    load = askopenfilename(filetypes=[("Image File",'.jpeg .jpg .png .HEIC')])
    
    
    im = Image.open(load)
    
    im = im.resize((400, 200))

    render = ImageTk.PhotoImage(im)
    
    

    # labels can be text or images
    img = tk.Label(window, image=render,width=400,height=200)
    img.image = render
    img.place(x=100, y=300)

k=tk.Button(window,text="Browse Image", command=showImgg, bg="white"  ,fg="black"  ,width=20  ,height=1,font=('times', 20, 'italic bold underline'))
k.place(x=150,y=200)


def predict():
    print("predicts")
    
    image = Image.open(load) 
    res = model(image, verbose=False)
    detect_img = res[0].plot()
    detect_img = cv2.cvtColor(detect_img, cv2.COLOR_BGR2RGB)
    
    cv2.imwrite('saved_image.jpg', detect_img)
    print("Image saved successfully.")
    
    im = Image.open("saved_image.jpg")
        
    im = im.resize((400, 200))
    
    render = ImageTk.PhotoImage(im)       
        
    # labels can be text or images
    img = tk.Label(window, image=render,width=400,height=200)
    img.image = render        
    img.place(x=900,y=300)

        
    
but  = tk.Button(window, text = "Predict", command = predict, bg = 'red', fg = 'white', width=20, height =1, font=('times', 20, 'italic bold underline') )
but.place(x=950, y=200)


window.mainloop()