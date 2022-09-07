import tensorflow as tf
import numpy as np


class add_noise(tf.Module):
    def __init__(self):
        super(add_noise, self).__init__()

    def __call__(self,img):
        # noise_random = np.random.randint(0,3)
        noise_random = 2
        if noise_random == 0: noise_img = self.gauss_fix(img)
        if noise_random == 1: noise_img = self.gauss_range(img)
        if noise_random == 2: noise_img = self.poisson_fix(img)
        # if noise_random == 3: noise_img = self.poisson_range(img)
        return noise_img
    
    # 高斯噪聲 sigma = 25
    def gauss_fix(self, img):
        sigma = 25 / 255
        noise_img = tf.random.normal(shape=tf.shape(img))
        noise_img = img + noise_img * sigma
        noise_img = tf.clip_by_value(noise_img, clip_value_min=0, clip_value_max=1)
        return noise_img
    
    # 高斯噪聲 sigma = [5,50]
    def gauss_range(self, img):
        min_std = 5 / 255
        max_std = 50 / 255
        sigma = tf.random.uniform(shape=(1,1,1), minval=min_std, maxval=max_std)
        noise_img = img + tf.random.normal(shape=tf.shape(img)) * sigma
        noise_img = tf.clip_by_value(noise_img, clip_value_min=0, clip_value_max=1)
        return noise_img
    
    # 泊松噪聲 lam = 25
    def poisson_fix(self, img):
        lam = 30 / 255
        poisson = tf.random.poisson(shape=tf.shape(img), lam = lam)
        noise_img = img + poisson
        noise_img = tf.clip_by_value(noise_img, clip_value_min=0, clip_value_max=1)
        return noise_img
    
    def poisson_range(self, img):
        min_lam = 5
        max_lam = 50
        lam = np.random.uniform(low=min_lam, high=max_lam, size=(1,1,1))
        noise_img = np.array(np.random.poisson(lam * img)/lam, dtype=np.float32)
        noise_img = np.clip(noise_img, a_min=0, a_max=1)
        return noise_img