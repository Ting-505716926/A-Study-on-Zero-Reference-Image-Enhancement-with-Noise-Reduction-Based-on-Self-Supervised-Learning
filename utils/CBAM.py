from tensorflow import split, reduce_mean, reduce_max,reshape
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation,\
    DepthwiseConv2D, Reshape, Concatenate, Multiply, Add, ReLU, GlobalAveragePooling2D,GlobalMaxPooling2D, Permute

from utils.Activation import HardSigmoid, HardSwish


def CBAM_block(x , r=2, name=''):
    # 獲取Channel個數
    C = x.shape[3] 
    # -----------------------------------Channel Attention Module-----------------------------------------------------------
    GMP_x = GlobalMaxPooling2D(name = name + '_CBAM_GMP')(x)
    GAP_x = GlobalAveragePooling2D(name = name + '_CBAM_GAP')(x)
    
    shared_mlp_down = Dense(C//r, name=name + '_CBAM_Shared_MLP_Down')
    GMP_x = shared_mlp_down(GMP_x)
    GMP_x = HardSigmoid(name=name+"_CBAM_GMP_hardsigmoid1")(GMP_x)
    GAP_x = shared_mlp_down(GAP_x)
    GAP_x = HardSigmoid(name=name+"_CBAM_GAP_hardsigmoid2")(GAP_x)
    
    shared_mlp_up = Dense(C, name=name + '_CBAMShared_MLP_Up')
    GMP_x = shared_mlp_up(GMP_x)
    GMP_x = HardSwish(name=name+"_CBAM_GMP_hardswish1")(GMP_x)
    GAP_x = shared_mlp_up(GAP_x)
    GAP_x = HardSwish(name=name+"_CBAM_GAP_hardswish2")(GAP_x)
    
    channel_attention = Add()([GMP_x, GAP_x])
    channel_attention = HardSigmoid(name=name+"_CBAM_ADD_hardsigmoid")(channel_attention)
    
    # -----------------------------------channel_attention與x相乘-----------------------------------------------------------
    channel_attention = Reshape(target_shape=(1,1,C),name=name+"_CBAM_Reshape")(channel_attention)
    x = Multiply(name=name+"_CBAM_Multiply1")([x, channel_attention])
    
    # -----------------------------------Spatial Attention Module-----------------------------------------------------------
    MP_y = reduce_mean(x, axis=3, keepdims=True)
    AP_y = reduce_max(x, axis=3,  keepdims=True)
    y = Concatenate(name=name+"_CBAM_Concatenate")([MP_y, AP_y])
    
    # -----------------------------------channel pooling--------------------------------------------------------------------
    y = Conv2D(1, 7, padding='same',name=name+"_CBAM_Conv1")(y)
    y = HardSigmoid(name=name+"_CBAM_hardsigmoid3")(y)
    
    # x與y相乘
    output = Multiply(name=name+"_CBAM_Multiply2")([x, y])
    return output