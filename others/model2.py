# 工具
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Concatenate, Activation, Conv2DTranspose, LeakyReLU, \
    MaxPool2D, BatchNormalization, DepthwiseConv2D, UpSampling2D
from tensorflow.keras import Input, Model
import numpy as np
# 自定義Loss
from utils import Losses
from utils.Activation import HardSwish
from utils.sub_sampler import sub_sampler
from utils.add_noise import add_noise


def Upsamplecat(x1,x2, filter):
    initializer = tf.keras.initializers.HeNormal()
    bias_inint = tf.keras.initializers.Zeros()
    x1 = Conv2DTranspose(filter,
                         2,
                         (2,2),
                         'same',
                         use_bias = True,
                         kernel_initializer=initializer,
                         bias_initializer= bias_inint
                         )(x1)
    # x1 = tf.image.resize(x1,[tf.shape(x2)[1], tf.shape(x2)[2]])
    x2 = Concatenate()([x1,x2])
    return x2

def conv_Leakyrelu_batchN(x, filter, kernel_size):
    initializer = tf.keras.initializers.HeNormal()
    bias_inint = tf.keras.initializers.Zeros()
    x = Conv2D(filter, 
               kernel_size, 
               (1, 1),
               padding='same',
               use_bias = True,
               kernel_initializer=initializer,
               bias_initializer= bias_inint
               )(x)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization()(x)
    return x

def UNet(inputs):
    
    filter = 32
    #------------------Encoder part---------------------
    pool0 = inputs
    x = conv_Leakyrelu_batchN(inputs, filter, 3)
    x = conv_Leakyrelu_batchN(inputs, filter, 3)
    x = MaxPool2D()(x)
    pool1 = x
    
    x = conv_Leakyrelu_batchN(x, filter, 3)
    x = MaxPool2D()(x)
    pool2 = x
    
    x = conv_Leakyrelu_batchN(x, filter, 3)
    x = MaxPool2D()(x)
    pool3 = x
    
    x = conv_Leakyrelu_batchN(x, filter, 3)
    x = MaxPool2D()(x)
    pool4 = x
    
    x = conv_Leakyrelu_batchN(x, filter, 3)
    x = MaxPool2D()(x)
    
    x = conv_Leakyrelu_batchN(x, filter, 3)
    
    #-------------------Decoder part--------------------
    x = Upsamplecat(x, pool4, filter)
    x = conv_Leakyrelu_batchN(x, filter*2, 3)
    
    x = Upsamplecat(x, pool3, filter) 
    x = conv_Leakyrelu_batchN(x, filter*2, 3)
    x = conv_Leakyrelu_batchN(x, filter*2, 3)
    
    x = Upsamplecat(x, pool2, filter)
    x = conv_Leakyrelu_batchN(x, filter*2, 3)
    x = conv_Leakyrelu_batchN(x, filter*2, 3)
    
    x = Upsamplecat(x, pool1, filter)
    x = conv_Leakyrelu_batchN(x, filter*2, 3)
    x = conv_Leakyrelu_batchN(x, filter*2, 3)
    
    x = Upsamplecat(x, pool0, filter)
    
    x = conv_Leakyrelu_batchN(x, filter*2+3, 3)
    x = conv_Leakyrelu_batchN(x, filter*2, 3)
    
    x = conv_Leakyrelu_batchN(x, filter*2, 1)
    x = conv_Leakyrelu_batchN(x, filter*2, 1)
    
    x = Conv2D(3, 1, (1, 1), padding='same')(x)
    
    return Model(inputs=inputs, outputs=x)

def dwconv(x, filter, kernel_size):
    x = DepthwiseConv2D(kernel_size, (1,1), 'same')(x)
    x = Conv2D(filter, 1, (1,1), 'same')(x)
    return x

def enhancement_net(inputs):
    pool = inputs
    x1 = Conv2D(32, 3, (1, 1), padding='same')(inputs)
    x1 = Activation('relu')(x1)
    x2 = Conv2D(32, 3, (1, 1), padding='same')(x1)
    x2 = Activation('relu')(x2)
    x3 = Conv2D(32, 3, (1, 1), padding='same')(x2)
    x3 = Activation('relu')(x3)
    x4 = Conv2D(32, 3, (1, 1), padding='same')(x3)
    x4 = Activation('relu')(x4)
    
    x5 = Concatenate()([x3,x4])
    x5 = Conv2D(32, 3, (1, 1), padding='same')(x5)
    x5 = Activation('relu')(x5)
    
    x6 = Concatenate()([x2,x5])
    x6 = Conv2D(32, 3, (1, 1), padding='same')(x6)
    x6 = Activation('relu')(x6)
    
    parameter_map = Concatenate()([x1,x6])
    parameter_map = Conv2D(3,  3, (1, 1), padding='same')(parameter_map)
    parameter_map = Activation('tanh')(parameter_map)
    
    x7 = Concatenate()([x3,x4])
    x7 = Conv2D(32, 3, (1, 1), padding='same')(x7)
    x7 = Activation('relu')(x7)
    
    x8 = Concatenate()([x2,x7])
    x8 = Conv2D(32, 3, (1, 1), padding='same')(x8)
    x8 = Activation('relu')(x8)
    
    x9 = Concatenate()([x8,pool])
    x9 = Conv2D(16, 1, (1, 1), padding='same')(x9)
    x9 = Activation('relu')(x9)
    x9 = Conv2D(8, 1, (1, 1), padding='same')(x9)
    x9 = Activation('relu')(x9)
    denoised_img = Conv2D(3, 1, (1, 1), padding='same')(x9)
    # denoised_img = Activation('sigmoid')(denoised_img)
    
    return Model(inputs=inputs, outputs=[parameter_map, denoised_img])

def enhancement_net_lw(inputs):
    x1 = dwconv(inputs, 32, 3)
    x1 = HardSwish(name='hardswish1')(x1)
    
    x2 = dwconv(x1, 32, 3)
    x2 = HardSwish(name='hardswish2')(x2)
    
    x3 = dwconv(x2, 32, 3)
    x3 = HardSwish(name='hardswish3')(x3)
    
    x4 = dwconv(x3, 32, 3)
    x4 = HardSwish(name='hardswish4')(x4)
    
    x5 = Concatenate()([x3,x4])
    x5 = dwconv(x5, 32, 3)
    x5 = HardSwish(name='hardswish5')(x5)
    
    x6 = Concatenate()([x2,x5])
    x6 = dwconv(x6, 32, 3)
    x6 = HardSwish(name='hardswish6')(x6)
    
    parameter_map = Concatenate()([x1,x6])
    parameter_map = dwconv(parameter_map, 3, 3)
    parameter_map = Activation('tanh')(parameter_map)
    
    return Model(inputs=inputs, outputs=parameter_map)
    
class enhance_net(tf.keras.Model):
    def __init__(self, input_shape=(None,None,3),model_name='', train=True, add_noise=False, Lambda1=1, Lambda2=2, **kwargs):
        super(enhance_net, self).__init__(**kwargs)
        self.train = train
        self.model_name = model_name
        self.Lambda1 = Lambda1
        self.Lambda2 = Lambda2
        
        self.L_spa = Losses.L_spa()
        self.L_exp = Losses.L_exp(16)
        self.L_col = Losses.L_col()
        self.L_tv = Losses.L_tv()
        self.L_rec = Losses.L_rec()
        
        self.sub_sample = sub_sampler()
        self.add_noise = add_noise
        
        self.enhancement_net = enhancement_net(Input(shape=input_shape))        
        # self.denoising_net = UNet(Input(shape=input_shape))
            
        self.build(input_shape=(None, input_shape[0], input_shape[1], input_shape[2]))
        self.call(Input(shape=input_shape))
         
    def enhance(self, x, x_r):
        for i in range(8):
            x = x + x_r*(x - tf.math.pow(x,2))
        return x
        
    
    def call(self, inputs):
        # Denoise
        inputs = add_noise()(inputs) if self.add_noise == True else inputs
        
        g1,g2 = self.sub_sample(inputs)
        _,denoised_g1 = self.enhancement_net(g1)
        
        _,denoised_img = self.enhancement_net(inputs)
        fully_g1,fully_g2 = self.sub_sample(denoised_img)

        parameter_map,_ = self.enhancement_net(inputs)
        enhance_image = self.enhance(inputs, parameter_map)
        
        if self.train == True:
             return enhance_image, parameter_map, g2, denoised_g1, fully_g1, fully_g2
        else:
            return parameter_map, inputs, denoised_img
        
    @tf.function
    def train_step(self, data, epoch, epochs):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        
        with tf.GradientTape() as tape:
            # Forward pass
            
            enhance_image, parameter_map, g2, enhance_g1, fully_g1, fully_g2 = self(data)
            
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            
            # enhancemnet_net Loss
            l_spa = self.L_spa(enhance_image, data)
            l_exp = self.L_exp(enhance_image, 0.6)
            l_col = self.L_col(enhance_image)
            l_tv  = self.L_tv(parameter_map)
            
            # 8, 1.75, 1, 7
            # 1, 1, 0.5, 20
            # 1, 10, 5, 200
            # 10, 8, 5, 200
            
            w = [10, 8, 5, 200]
            # Total loss
            enhancement_loss = w[0]*l_spa + w[1]*l_exp + w[2]*l_col + w[3]*l_tv
            
            # denoising_net loss
            Lambda = (epoch+1)/epochs*2

            l_rec1, l_rec2 = self.L_rec(Lambda, g2, enhance_g1, fully_g1, fully_g2)
            L_rec = self.Lambda1 * l_rec1 + self.Lambda2 * l_rec2
            
            loss = enhancement_loss + L_rec
           
        # Compute gradients or Update weights
        trainable_variables = self.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))

        
        
        return {'train_loss': loss, 'l_spa': w[0]*l_spa, 'l_exp': w[1]*l_exp, 'l_col': w[2]*l_col, 'l_tv': w[3]*l_tv, 'l_rec': L_rec}
    
    
    def validation_step(self, validation_data, validation_label):
        self.train = False
        _, _, denoised_image = self(validation_data)
        ssim = tf.reduce_mean(tf.image.ssim(denoised_image, validation_label, max_val=1))
        psnr = tf.reduce_mean(tf.image.psnr(denoised_image, validation_label, max_val=1))
        self.train = True
        return ssim, psnr
    
    def model_save(self, epoch, path):
        #self.enhancement_net.save(path + self.model_name +'model/epoch{0}'.format(epoch+1))
        self.enhancement_net.save_weights(path + self.model_name + '/enhancment/weights/epoch{0}/'.format(epoch+1)) 
        #self.denoising_net.save(path + self.model_name +'/model/epoch{0}'.format(epoch+1))
        # self.denoising_net.save_weights(path + self.model_name + '/denoising/weights/epoch{0}/'.format(epoch+1))     

        
