import tensorflow as tf
from tensorflow.keras.layers import AvgPool2D
import numpy as np

class L_spa(tf.Module):
    def __init__(self):
        super(L_spa, self).__init__()
        self.kernel_up = tf.constant([[0,-1,0],
                                      [0,1,0],
                                      [0,0,0]], dtype=tf.float32)
        self.kernel_up = tf.reshape(self.kernel_up,[3,3,1,1])
        self.kernel_down = tf.constant([[0,0,0],
                                        [0,1,0],
                                        [0,-1,0]], dtype=tf.float32)
        self.kernel_down = tf.reshape(self.kernel_down,[3,3,1,1])
        self.kernel_left = tf.constant([[0,0,0],
                                        [-1,1,0],
                                        [0,0,0]], dtype=tf.float32)
        self.kernel_left = tf.reshape(self.kernel_left,[3,3,1,1])
        self.kernel_right = tf.constant([[0,0,0],
                                         [0,1,-1],
                                         [0,0,0]], dtype=tf.float32)
        self.kernel_right = tf.reshape(self.kernel_right,[3,3,1,1])
        
        self.avgpool = AvgPool2D((4,4))
    
    def __call__(self, org, enhance):
        org_mean =tf.math.reduce_mean(org, 3, keepdims=True)
        enhance_mean =tf.math.reduce_mean(enhance, 3, keepdims=True)
        
        org_pool = self.avgpool(org_mean)
        enhance_pool = self.avgpool(enhance_mean)
        # h = tf.shape(org_pool)[1]
        # k = tf.cast(h, dtype=tf.float32)
        
        D_org_up = tf.nn.conv2d(org_pool, self.kernel_up, padding='SAME', strides=[1,1,1,1])
        D_org_down = tf.nn.conv2d(org_pool, self.kernel_down, padding='SAME', strides=[1,1,1,1])
        D_org_left = tf.nn.conv2d(org_pool, self.kernel_left, padding='SAME', strides=[1,1,1,1])
        D_org_right = tf.nn.conv2d(org_pool, self.kernel_right, padding='SAME', strides=[1,1,1,1])
        
        D_enhance_up = tf.nn.conv2d(enhance_pool, self.kernel_up, padding='SAME', strides=[1,1,1,1])
        D_enhance_down = tf.nn.conv2d(enhance_pool, self.kernel_down, padding='SAME', strides=[1,1,1,1])
        D_enhance_left = tf.nn.conv2d(enhance_pool, self.kernel_left, padding='SAME', strides=[1,1,1,1])
        D_enhance_right = tf.nn.conv2d(enhance_pool, self.kernel_right, padding='SAME', strides=[1,1,1,1])
        
        D_up = tf.math.pow(D_org_up-D_enhance_up, 2)
        D_down = tf.math.pow(D_org_down-D_enhance_down, 2)
        D_left = tf.math.pow(D_org_left-D_enhance_left, 2)
        D_right = tf.math.pow(D_org_right-D_enhance_right, 2)
        
        E = (D_up + D_down + D_left + D_right)
        E = tf.math.reduce_mean(E)
        # E = tf.math.reduce_sum(E)
        
        return E
    
class L_exp(tf.Module):
    def __init__(self, M_size):
        super(L_exp, self).__init__()
        self.M_size = M_size
        self.avgpool = AvgPool2D((self.M_size,self.M_size))
        
    def __call__(self, x, E):
        _,h,_,_ = np.shape(x)
        # k = tf.cast(h/self.M_size, dtype=tf.float32)
        
        x_mean = tf.math.reduce_mean(x, 3, keepdims=True)
        
        x_pool = self.avgpool(x_mean)
        
        d = tf.math.pow(x_pool - tf.constant([E]), 2)
        
        d = tf.math.reduce_mean(d)
        # new_x = tf.math.reduce_sum(new_x)
        
        return d
        
class L_col(tf.Module):
    def __init__(self):
        super(L_col, self).__init__()
    def __call__(self, x):
        mean_rgb = tf.math.reduce_mean(x, [1,2], keepdims=True)
        mean_r, mean_g, mean_b = tf.split(mean_rgb, 3, axis=3)
        
        Drg = tf.math.pow(mean_r - mean_g, 2)
        Drb = tf.math.pow(mean_r - mean_b, 2)
        Dgb = tf.math.pow(mean_g - mean_b, 2)
        
        k = tf.math.pow(Drg + Drb + Dgb + 1e-5, 0.5)

        k = tf.math.reduce_mean(k)
        
        return k
      
class L_tv(tf.Module):
    def __init__(self,TVLoss_weight=1.0):
        super(L_tv, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def __call__(self,x):
        batch_size = tf.shape(x)[0]
        batch_size = tf.cast(batch_size, dtype=tf.float32)
        h_x = tf.shape(x)[1]
        w_x = tf.shape(x)[2]
        
        count_h = (h_x-1) * w_x
        count_h = tf.cast(count_h, dtype=tf.float32)
        count_w = h_x * (w_x - 1)
        count_w = tf.cast(count_w, dtype=tf.float32)
        
        h_tv = tf.math.reduce_sum(tf.math.pow((x[:,1:,:,:]-x[:,:h_x-1,:,:]),2))
        w_tv = tf.math.reduce_sum(tf.math.pow((x[:,:,1:,:]-x[:,:,:w_x-1,:]),2))
        
        return self.TVLoss_weight*2*(h_tv/count_h + w_tv/count_w)/batch_size     
        
class L_rec(tf.Module):
    def __init__(self):
        super(L_rec, self).__init__()
    def __call__(self, Lambda, enhance_image_g2, denoising_img_g1, denoising_image_g1, denoising_image_g2):
        diff = denoising_img_g1 - enhance_image_g2
        exp_diff = denoising_image_g1 - denoising_image_g2
        L_rec1 = tf.math.reduce_mean(tf.pow(diff,2))
        L_rec2 = Lambda * tf.math.reduce_mean(tf.pow(diff - exp_diff, 2))
        return L_rec1, L_rec2

# Assembling regularizer
class L_ass(tf.Module):
    def __init__(self,TVLoss_weight=1.0):
        super(L_ass, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def __call__(self,x):
        batch_size = tf.shape(x)[0]
        batch_size = tf.cast(batch_size, dtype=tf.float32)
        h_x = tf.shape(x)[1]
        w_x = tf.shape(x)[2]
    
        L_ass1 = tf.reduce_mean(tf.math.pow((x[:,1:,:,:]-x[:,:h_x-1,:,:]),2)) + \
                 tf.reduce_mean(tf.math.pow((x[:,:,1:,:]-x[:,:,:w_x-1,:]),2))
        
        L_ass2 = tf.reduce_mean(tf.math.pow(x[:,:-2,:,:]- 2*x[:,1:-1,:,:] + x[:,2:,:,:],2)) + \
                 tf.reduce_mean(tf.math.pow(x[:,:,:-2,:]- 2*x[:,:,1:-1,:] + x[:,:,2:,:],2))
        
        return L_ass1, L_ass2