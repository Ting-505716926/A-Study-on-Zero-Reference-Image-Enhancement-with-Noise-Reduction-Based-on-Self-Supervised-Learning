import tensorflow as tf
import numpy as np

class sub_sampler(tf.Module):
    def __init__(self):
        super(sub_sampler, self).__init__()
        i = 0
        j = 0
        i = np.random.randint(1,5)
        while(True):
            j = np.random.randint(1,5)
            if i == j:
                continue
            else:
                break
        
        mask1_1 = tf.constant([[[1,0,0],[0,0,0]],[[0,0,0],[0,0,0]]], dtype=tf.float32)
        mask1_2 = tf.constant([[[0,1,0],[0,0,0]],[[0,0,0],[0,0,0]]], dtype=tf.float32)
        mask1_3 = tf.constant([[[0,0,1],[0,0,0]],[[0,0,0],[0,0,0]]], dtype=tf.float32)
        
        mask2_1 = tf.constant([[[0,0,0],[1,0,0]],[[0,0,0],[0,0,0]]], dtype=tf.float32)
        mask2_2 = tf.constant([[[0,0,0],[0,1,0]],[[0,0,0],[0,0,0]]], dtype=tf.float32)
        mask2_3 = tf.constant([[[0,0,0],[0,0,1]],[[0,0,0],[0,0,0]]], dtype=tf.float32)

        mask3_1 = tf.constant([[[0,0,0],[0,0,0]],[[1,0,0],[0,0,0]]], dtype=tf.float32)
        mask3_2 = tf.constant([[[0,0,0],[0,0,0]],[[0,1,0],[0,0,0]]], dtype=tf.float32)
        mask3_3 = tf.constant([[[0,0,0],[0,0,0]],[[0,0,1],[0,0,0]]], dtype=tf.float32)
        
        mask4_1 = tf.constant([[[0,0,0],[0,0,0]],[[0,0,0],[1,0,0]]], dtype=tf.float32)
        mask4_2 = tf.constant([[[0,0,0],[0,0,0]],[[0,0,0],[0,1,0]]], dtype=tf.float32)
        mask4_3 = tf.constant([[[0,0,0],[0,0,0]],[[0,0,0],[0,0,1]]], dtype=tf.float32)

        if i == 1:
            self.mask1_1 = tf.reshape(mask1_1, [2,2,3,1])
            self.mask1_2 = tf.reshape(mask1_2, [2,2,3,1])
            self.mask1_3 = tf.reshape(mask1_3, [2,2,3,1])
        elif i == 2 :
            self.mask1_1 = tf.reshape(mask2_1, [2,2,3,1])
            self.mask1_2 = tf.reshape(mask2_2, [2,2,3,1])
            self.mask1_3 = tf.reshape(mask2_3, [2,2,3,1])
        elif i == 3 :
            self.mask1_1 = tf.reshape(mask3_1, [2,2,3,1])
            self.mask1_2 = tf.reshape(mask3_2, [2,2,3,1])
            self.mask1_3 = tf.reshape(mask3_3, [2,2,3,1])
        elif i == 4 :
            self.mask1_1 = tf.reshape(mask4_1, [2,2,3,1])
            self.mask1_2 = tf.reshape(mask4_2, [2,2,3,1])
            self.mask1_3 = tf.reshape(mask4_3, [2,2,3,1])
        
        if j == 1:
            self.mask2_1 = tf.reshape(mask1_1, [2,2,3,1])
            self.mask2_2 = tf.reshape(mask1_2, [2,2,3,1])
            self.mask2_3 = tf.reshape(mask1_3, [2,2,3,1])
        elif j == 2 :
            self.mask2_1 = tf.reshape(mask2_1, [2,2,3,1])
            self.mask2_2 = tf.reshape(mask2_2, [2,2,3,1])
            self.mask2_3 = tf.reshape(mask2_3, [2,2,3,1])
        elif j == 3 :
            self.mask2_1 = tf.reshape(mask3_1, [2,2,3,1])
            self.mask2_2 = tf.reshape(mask3_2, [2,2,3,1])
            self.mask2_3 = tf.reshape(mask3_3, [2,2,3,1])
        elif j == 4 :
            self.mask2_1 = tf.reshape(mask4_1, [2,2,3,1])
            self.mask2_2 = tf.reshape(mask4_2, [2,2,3,1])
            self.mask2_3 = tf.reshape(mask4_3, [2,2,3,1])

    def __call__(self, img):
        g1_1 = tf.nn.conv2d(img, self.mask1_1, padding='SAME', strides=[1,2,2,1])
        g1_2 = tf.nn.conv2d(img, self.mask1_2, padding='SAME', strides=[1,2,2,1])
        g1_3 = tf.nn.conv2d(img, self.mask1_3, padding='SAME', strides=[1,2,2,1])
        g1 = tf.keras.layers.Concatenate()([g1_1, g1_2, g1_3])
        
        g2_1 = tf.nn.conv2d(img, self.mask2_1, padding='SAME', strides=[1,2,2,1])
        g2_2 = tf.nn.conv2d(img, self.mask2_2, padding='SAME', strides=[1,2,2,1])
        g2_3 = tf.nn.conv2d(img, self.mask2_3, padding='SAME', strides=[1,2,2,1])
        g2 = tf.keras.layers.Concatenate()([g2_1, g2_2, g2_3])
        
        return g1,g2
            