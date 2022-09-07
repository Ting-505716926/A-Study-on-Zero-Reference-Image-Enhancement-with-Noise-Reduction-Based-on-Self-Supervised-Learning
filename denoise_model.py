# 工具
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Concatenate, Activation, Conv2DTranspose, LeakyReLU, Dense,\
    MaxPool2D, BatchNormalization, DepthwiseConv2D, UpSampling2D, GlobalAveragePooling2D, Minimum, Reshape
from tensorflow.keras import Input, Model
# from utils.DyReLU import DyReLU
import numpy as np
# 自定義Loss
from utils import Losses
from utils.Activation import HardSwish
from utils.sub_sampler import sub_sampler
from utils.add_noise import add_noise
from utils.CBAM import CBAM_block



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
    x1 = tf.image.resize(x1, [tf.shape(x2)[1], tf.shape(x2)[2]])
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
    filter = 48
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
    x = conv_Leakyrelu_batchN(x, filter*2, 3)
    
    x = Upsamplecat(x, pool3, filter*2) 
    x = conv_Leakyrelu_batchN(x, filter*2, 3)
    x = conv_Leakyrelu_batchN(x, filter*2, 3)
    
    x = Upsamplecat(x, pool2, filter*2)
    x = conv_Leakyrelu_batchN(x, filter*2, 3)
    x = conv_Leakyrelu_batchN(x, filter*2, 3)
    
    x = Upsamplecat(x, pool1, filter*2)
    x = conv_Leakyrelu_batchN(x, filter*2, 3)
    x = conv_Leakyrelu_batchN(x, filter*2, 3)
    
    x = Upsamplecat(x, pool0, filter*2)
    x = conv_Leakyrelu_batchN(x, filter*2, 3)
    x = conv_Leakyrelu_batchN(x, filter*2, 3)
    
    x = conv_Leakyrelu_batchN(x, filter*2, 1)
    x = conv_Leakyrelu_batchN(x, filter*2, 1)
    
    x = Conv2D(3, 1, (1, 1), padding='same')(x)
    x = Activation('sigmoid')(x)
    
    return Model(inputs=inputs, outputs=x)

def DSC(x, filters):
    initializer = tf.keras.initializers.HeNormal()
    x = DepthwiseConv2D(3, (1,1), 'same', depthwise_initializer=initializer)(x)
    x = Conv2D(filters, 1, (1,1), 'same', kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = DepthwiseConv2D(3, (1,1), 'same', depthwise_initializer=initializer)(x)
    x = Conv2D(filters, 1, (1,1), 'same', kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    return x

def smart_unet(inputs):
    d0 = inputs
    x = Conv2D(48, 1, (1,1), 'same', kernel_initializer=tf.keras.initializers.HeNormal())(inputs)
    x = LeakyReLU(0.2)(x)
    x = DSC(x, 48)
    x = MaxPool2D()(x)
    d1 = x # 128 * 128
    
    x = DSC(x, 48)
    x = MaxPool2D()(x)
    d2 = x # 64 * 64
    
    x = DSC(x, 48)
    x = MaxPool2D()(x)
    d3 = x # 32 * 32
    
    x = DSC(x, 48)
    x = MaxPool2D()(x)
    d4 = x # 16 * 16

    x = DSC(x, 48)
    x = MaxPool2D()(x) # 8 * 8
    
    x = DSC(x, 48)
    
    x = Upsamplecat(x, d4, 48) # 16* 16
    x = DSC(x, 48) 
    
    x = Upsamplecat(x, d3, 96)# 32 * 32
    x = DSC(x, 96) 
    
    x = Upsamplecat(x, d2, 96)# 64 *64
    x = DSC(x, 96)
    
    x = Upsamplecat(x, d1, 96)# 128 *128
    x = DSC(x, 96)

    x = Upsamplecat(x, d0, 96)# 256 * 256
    x = DSC(x, 96)

    x = conv_Leakyrelu_batchN(x, 48*2, 1)
    x = conv_Leakyrelu_batchN(x, 48*2, 1)
    output = Conv2D(3, 1, (1,1), 'same', activation='sigmoid')(x)
    
    return Model(inputs,output)

def smart_unet_dyrelu(inputs):
    d0 = inputs
    x = Conv2D(48, 1, (1,1), 'same', kernel_initializer=tf.keras.initializers.HeNormal())(inputs)
    x = DyReLU(48)(x)
    x = DSC(x, 48)
    x = MaxPool2D()(x)
    d1 = x # 128 * 128
    
    x = DSC(x, 48)
    x = MaxPool2D()(x)
    d2 = x # 64 * 64
    
    x = DSC(x, 48)
    x = MaxPool2D()(x)
    d3 = x # 32 * 32
    
    x = DSC(x, 48)
    x = MaxPool2D()(x)
    d4 = x # 16 * 16

    x = DSC(x, 48)
    x = MaxPool2D()(x) # 8 * 8
    
    x = DSC(x, 48)
    
    x = Upsamplecat(x, d4, 48) # 16* 16
    x = DSC(x, 48) 
    
    x = Upsamplecat(x, d3, 96)# 32 * 32
    x = DSC(x, 96) 
    
    x = Upsamplecat(x, d2, 96)# 64 *64
    x = DSC(x, 96)
    
    x = Upsamplecat(x, d1, 96)# 128 *128
    x = DSC(x, 96)

    x = Upsamplecat(x, d0, 96)# 256 * 256
    x = DSC(x, 96)

    x = conv_Leakyrelu_batchN(x, 48*2, 1)
    x = conv_Leakyrelu_batchN(x, 48*2, 1)
    output = Conv2D(3, 1, (1,1), 'same', activation='sigmoid')(x)
    
    return Model(inputs,output)

class denoising_net(tf.keras.Model):
    def __init__(self, input_shape=(None,None,3), mode = 'Small', model_name='', train=True, add_noise=False, Lambda1=1, Lambda2=2, **kwargs):
        super(denoising_net, self).__init__(**kwargs)
        self.train = train
        self.model_name = model_name
        self.Lambda1 = Lambda1
        self.Lambda2 = Lambda2
        
        self.L_rec = Losses.L_rec()
        self.L_ass = Losses.L_ass()
        self.L_spq = Losses.L_spa()
        

        self.add_noise = add_noise
        if mode == 'Large':            
            self.denoising_net = UNet(Input(shape=input_shape))
        elif mode == 'Small':
            self.denoising_net = smart_unet(Input(shape=input_shape))
        # self.denoising_net = smart_unet_dyrelu(Input(shape=input_shape))

        self.build(input_shape=(None, input_shape[0], input_shape[1], input_shape[2]))
        self.call(Input(shape=input_shape))
        

    def call(self, inputs):
        # inputs = RandomCrop(256,256)(inputs)
        # Denoise
        inputs = add_noise()(inputs) if self.add_noise == True else inputs
        
        g1,g2 = sub_sampler()(inputs)
        denoised_g1 = self.denoising_net(g1)

        denoised_img = self.denoising_net(inputs)
        denoised_img =tf.stop_gradient(denoised_img)
        fully_g1,fully_g2 = sub_sampler()(denoised_img)
        
        if self.train == True:
            return g2, denoised_g1, fully_g1, fully_g2, denoised_img
        else:
            return denoised_img, inputs
        

    def train_step(self, data, epoch, epochs):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        
        with tf.GradientTape() as tape:
            # Forward pass
            g2, denoising_g1, fully_g1, fully_g2, denoised_img = self(data)
            
            # up_fully_g1 = UpSampling2D()(fully_g1)
            # up_fully_g2 = UpSampling2D()(fully_g2)
            
            # Total loss
            Lambda = (epoch+1)/epochs*2

            l_rec1, l_rec2 = self.L_rec(Lambda, g2, denoising_g1, fully_g1, fully_g2)
            L_rec = self.Lambda1 * l_rec1 + self.Lambda2 * l_rec2

            # l_ass1_1, l_ass2_1 = self.L_ass(up_fully_g1)
            # l_ass1_2, l_ass2_2 = self.L_ass(up_fully_g2)
            # l_ass1 = l_ass1_1 + l_ass1_2
            # l_ass2 = l_ass2_1 + l_ass2_2

            # l_ssim_1 = tf.reduce_mean(tf.abs(1 - tf.image.ssim(denoised_img, up_fully_g1, max_val=1)))
            # l_ssim_2 = tf.reduce_mean(tf.abs(1 - tf.image.ssim(denoised_img, up_fully_g2, max_val=1)))
            # L_ssim = l_ssim_1 + l_ssim_2
            

            w = [1,1]
            loss =  L_rec # + L_ssim
        

        # Compute gradients or Update weights
        trainable_vaiables = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vaiables)
        self.optimizer.apply_gradients(zip(gradients, trainable_vaiables))
        
        
        return {'train_loss': loss, 'l_rec1': l_rec1, 'l_rec2': l_rec2, # 'L_ssim': L_ssim
            }
    

    def validation_step(self, validation_data, label):
        self.train = False
        denoised_image, _ = self(validation_data)
        ssim = tf.reduce_mean(tf.image.ssim(denoised_image, label, max_val=1))
        psnr = tf.reduce_mean(tf.image.psnr(denoised_image, label, max_val=1))
        self.train = True
        return ssim, psnr
    
    def model_save(self, epoch, path):
        self.denoising_net.save_weights(path + self.model_name + '/weights/epoch{0}/'.format(epoch+1))     

        
