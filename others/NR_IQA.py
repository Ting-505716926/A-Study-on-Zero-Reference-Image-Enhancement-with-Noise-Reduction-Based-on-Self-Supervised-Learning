# Non Reference Image quality Assessment
# Paper Name: <<No-reference color image quality assessment: from entropy to perceptual quality>>

import tensorflow as tf
import numpy as np
import cv2
import collections
import math
import time

def calc_array(img):
    # a = [i for i in range(256)]
    # img = np.array(a).astype(np.uint8).reshape(16,16)
    
    # 統計[0-255]灰階值的出現的次數
    hist_cv = cv2.calcHist([img], [0], None, [256], [0,256])
    
    h,w = np.shape(img)
    P = hist_cv/(h*w)
    E = np.sum([p * np.log2(1/p) for p in P])
    print("E: ",E)

def calc_2D_entropy(img):
    # a = [i for i in range(256)]
    # img = np.array(a).astype(np.uint8).reshape(16,16)
    
    N = 1
    h,w = np.shape(img)
    IJ = []
    
    for row in range(h):
        for col in range(w):
            L_x = np.max([0,col-N])
            R_x = np.min([w,col+N+1])
            U_y = np.max([0,row-N])
            D_y = np.min([h,row+N+1])
            region = img[U_y:D_y,L_x:R_x]
            j = (np.sum(region) - img[row][col]) / 8
            IJ.append([img[row][col],j])
    # print(IJ)
    
    F = []
    arr = [list(i) for i in set(tuple(j) for j in IJ)]

    # print(arr)
    for i in range(len(arr)):
        F.append(IJ.count(arr[i]))
    print(F)
    
    P = np.array(F)/(h*w)
    E = np.sum([p * np.log2(1/p) for p in P])
    print(E)  
    
def entropy_2D(img):
    img = cv2.copyMakeBorder(img, 1,1,1,1, cv2.BORDER_CONSTANT, value=0)
    h,w = np.shape(img)
    
    compare_list = []
    for m in range(1,h-1):
        for n in range(1,w-1):
            sum_region = img[m-1, n-1] + img[m, n-1] + img[m+1, n-1] + img[m-1, n] + img[m+1, n] + img[m-1,n+1] + img[m, n+1] + img[m+1,n+1]
            mean_region = sum_region / 8
            pix = img[m,n]
            temp = (pix, mean_region)
            compare_list.append(temp)
    compare_dict = collections.Counter(compare_list)
    size =(h-2)*(w-2)
    
    F =[]

    for count in compare_dict.values():
        F.append(count)
    
    P = np.array(F) / size
    E = np.sum([p * np.log2(1/p) for p in P])
        
    return E

class image_2D_entropy(tf.Module):
    def __init__(self):
        super(image_2D_entropy, self).__init__()
        self.kernel_up = tf.constant([[1,1,1],[1,0,1],[1,1,1]], dtype=tf.float32)
        self.kernel_up = tf.reshape(self.kernel_up,[3,3,1,1])
    def __call__(self, img):
        
        b,h,w,c = tf.shape(img)
        IJ = [[]*int(b)]
        strat = time.time()
        region = tf.nn.conv2d(img, self.kernel_up, padding='SAME', strides=[1,1,1,1])
        print(time.time()-strat)
        region /= 8
        img_center = tf.cast(tf.reshape(img,(h*w*c)), dtype=tf.float32) 
        j = tf.reshape(region,(h*w*c)) 
        
        # IJ = j**2 + 4* img_center * j
        IJ = -img_center/j
        
        IJ = tf.keras.layers.Concatenate()([img_center,j])
        
        compare_dict = collections.Counter(IJ.ref())
        print(compare_dict)
        
        _, _, count = tf.unique_with_counts(IJ)
        
        
        count = tf.cast(count, dtype=tf.float32)
        size = tf.cast(h*w, dtype=tf.float32)
        P = tf.math.divide(count, size)
        E = P * np.log2(1/P)
        E = tf.reduce_sum(E)
        # E = tf.math.reduce_sum([p * np.log2(1/p) for p in P])
      
        return IJ, count, E