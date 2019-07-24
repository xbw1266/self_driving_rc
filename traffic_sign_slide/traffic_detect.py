from keras.models import load_model
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from PIL import Image
import numpy as np
import os
import matplotlib.image as mpimg
import glob
from skimage.feature import hog
import random
from scipy import ndimage
from scipy.ndimage.measurements import label

INPUT_SIZE = 30

class traffic_sign_detect:

    def __init__(self,filename,showimage=True):
        self.showimage = showimage
        self.filename = filename
        self.model = load_model(self.filename)


    def draw_boxes(self,img, bboxes, color=(0, 0, 255), thick=1):
        # Make a copy of the image
        imcopy = np.copy(img)
        # Iterate through the bounding boxes
        for bbox in bboxes:
            r=random.randint(0,255)
            g=random.randint(0,255)
            b=random.randint(0,255)
            color=(r, g, b)
            # Draw a rectangle given bbox coordinates
            cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
        # Return the image copy with boxes drawn
        return imcopy

    def slide_window(self,img, x_start_stop=[None, None], y_start_stop=[None, None], 
                        xy_window=(60, 60), xy_overlap=(0.9, 0.9)):
        if x_start_stop[0] == None:
            x_start_stop[0]=0
        if x_start_stop[1] == None:
            x_start_stop[1]=img.shape[1]
        if y_start_stop[0] ==  None:
            y_start_stop[0]= 0
        if y_start_stop[1] ==  None:
            y_start_stop[1]=img.shape[0]
        window_list = []
        image_width_x= x_start_stop[1] - x_start_stop[0]
        image_width_y= y_start_stop[1] - y_start_stop[0]
        windows_x = np.int( 1 + (image_width_x - xy_window[0])/(xy_window[0] * xy_overlap[0]))
        windows_y = np.int( 1 + (image_width_y - xy_window[1])/(xy_window[1] * xy_overlap[1]))
        modified_window_size= xy_window
        for i in range(0,windows_y):
            y_start = y_start_stop[0] + np.int( i * modified_window_size[1] * xy_overlap[1])
            for j in range(0,windows_x):
                x_start = x_start_stop[0] + np.int( j * modified_window_size[0] * xy_overlap[0])
                x1 = np.int( x_start +  modified_window_size[0])
                y1= np.int( y_start + modified_window_size[1])
                window_list.append(((x_start,y_start),(x1,y1)))
        return window_list

    def DrawCars(self,image,windows, converColorspace=False):
        refinedWindows=[]
        for window in windows:
            start= window[0]
            end= window[1]
            clippedImage=image[start[1]:end[1], start[0]:end[0]]
            if(clippedImage.shape[1] == clippedImage.shape[0] and clippedImage.shape[1]!=0):
                image_from_array = Image.fromarray(clippedImage, 'RGB')
                clippedImage = image_from_array.resize((INPUT_SIZE, INPUT_SIZE))
                clippedImage = np.array(clippedImage)
                clippedImage = clippedImage.astype('float32')/255 
                clippedImage = np.expand_dims(clippedImage, axis=0)
                predictedOutput=self.model.predict(clippedImage)
                predictedOutputIndex = np.argmax(predictedOutput)
                if(predictedOutputIndex==14 and max(max(predictedOutput)) > 0.9):
                    refinedWindows.append(window)
        return refinedWindows
    
    def loadimage(self,image_name):
        #image = cv2.imread(image_name)i
        image = image_name
<<<<<<< HEAD
        self.image = cv2.resize(image,(int((image.shape[1])/5),int(image.shape[0]/5)),interpolation = cv2.INTER_AREA )

    def find_box(self):
        #windows1 = self.slide_window(self.image,   
        #                    xy_window=(50,50), xy_overlap=(0.20, 0.20))
        #windows4 = self.slide_window(self.image, 
        #                    xy_window=(55,55), xy_overlap=(0.25, 0.25))
        #windows2 = self.slide_window(self.image,  
        #                    xy_window=(60,60), xy_overlap=(0.30, 0.30))
        #windows3 = self.slide_window(self.image, 
        #                    xy_window=(65,65), xy_overlap=(0.40, 0.40))
        #windows = windows1 + windows2 +  windows3 + windows4
        windows4 = self.slide_window(self.image, 
                            xy_window=(40,40), xy_overlap=(0.20, 0.20))
        windows1 = self.slide_window(self.image,   
                            xy_window=(50,50), xy_overlap=(0.3, 0.3))
        windows3 = self.slide_window(self.image, 
                            xy_window=(55,55), xy_overlap=(0.40, 0.40))
        windows = windows1 +  windows4+windows3
=======
        self.image = cv2.resize(image,(int((image.shape[1])/3),int(image.shape[0]/3)),interpolation = cv2.INTER_AREA )

    def find_box(self):
        windows1 = self.slide_window(self.image,   
                            xy_window=(50,50), xy_overlap=(0.20, 0.20))
        windows4 = self.slide_window(self.image, 
                            xy_window=(55,55), xy_overlap=(0.25, 0.25))
        windows2 = self.slide_window(self.image,  
                            xy_window=(60,60), xy_overlap=(0.30, 0.30))
        windows3 = self.slide_window(self.image, 
                            xy_window=(65,65), xy_overlap=(0.40, 0.40))
        windows = windows1 + windows2 +  windows3 + windows4
>>>>>>> master
        print("Total No of windows are ",len(windows))
        self.refinedWindows=self.DrawCars(self.image,windows, True)
        window_img = self.draw_boxes(self.image, windows) 
        window_img_refined = self.draw_boxes(self.image,self.refinedWindows) 
        self.window_img = window_img_refined
        if(self.showimage):
<<<<<<< HEAD
=======
            #plt_img = cv2.hconcat([window_img,window_img_refined])
            #cv2.imshow('box image',plt_img)
>>>>>>> master
            cv2.imshow('box image',window_img_refined)
            cv2.waitKey(1)
        #if(self.showimage):
        #    f,axes= plt.subplots(2,1, figsize=(30,15))
        #    axes[0].imshow(window_img)
        #    axes[0].set_title("Window Coverage")
        #window_img = self.draw_boxes(self.image,self.refinedWindows) 
        #if(self.showimage):
        #    axes[1].set_title("Test Image with Refined Sliding Windows")
        #    axes[1].imshow(window_img_refined)
        #    plt.show()

    def add_heat(self,heatmap, bbox_list):
        # Iterate through list of bboxes
        for box in bbox_list:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
        return heatmap
    # applying a threshold value to the image to filter out low pixel cells
    def apply_threshold(self,heatmap, threshold):
        # Zero out pixels below the threshold
        heatmap[heatmap <= threshold] = 0
        # Return thresholded map
        return heatmap
    # find pixels with each car number and draw the final bounding boxes
    from scipy.ndimage.measurements import label
    def draw_labeled_bboxes(self,img, labels):
        # Iterate through all detected cars
        for car_number in range(1, labels[1]+1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
        # Return the image
        return img

    def draw_heatmap(self):
        #testing our heat function
        heat = np.zeros_like(self.window_img[:,:,0]).astype(np.float)
        heat = self.add_heat(heat,self.refinedWindows)
        # Apply threshold to help remove false positives
        heat = self.apply_threshold(heat,3)
        # Visualize the heatmap when displaying    
        heatmap = np.clip(heat, 0, 255)
        heat_image=heatmap
        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        print(" Number of sign found - ",labels[1])
        draw_img = self.draw_labeled_bboxes(np.copy(self.image), labels)
        if(self.showimage):
            cv2.imshow('heat map',heat_image)
            cv2.waitKey(1)
           # f,axes= plt.subplots(2,1, figsize=(30,15))
           # axes[0].imshow(heat_image,cmap='gray')
           # axes[0].set_title("Heat Map Image")
           # axes[1].imshow(draw_img)
           # axes[1].set_title("Final Image after applying Heat Map")
           # plt.show()
