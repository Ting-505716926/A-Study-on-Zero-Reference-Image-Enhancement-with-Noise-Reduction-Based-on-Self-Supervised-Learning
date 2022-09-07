import cv2
import numpy as np
from utils.GGD import GGD
from sklearn.svm import SVR

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def pre_processing(img):
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    win = cv2.getGaussianKernel(7, 7/6)
    # img local mean
    mu = cv2.filter2D(gray_img,-1,win)
    mu_sq = mu * mu
    
    # img local variance
    sigma = np.sqrt(np.abs(cv2.filter2D(gray_img * gray_img, -1, win) - mu_sq))   

    return mu, sigma

def luminance_gaussian(enhancement_gray_img, enhancement_mu, enhancement_sigma):
    structdis = (enhancement_gray_img - enhancement_mu) / (enhancement_sigma + 1)
    
    alpha, overallstd = GGD(structdis)
    
    return alpha, overallstd

def color_comparison(low_light_img, enhancement_img):
    low_light_hsv_img = cv2.cvtColor(low_light_img, cv2.COLOR_BGR2HSV)
    enhancemnet_hsv_img = cv2.cvtColor(enhancement_img, cv2.COLOR_BGR2HSV)

    low_h,low_s,_ = cv2.split(low_light_hsv_img)
    low_h = np.asarray(low_h) + 0.000001
    low_s = np.asarray(low_s) + 0.000001
    low_p_h = low_h / np.sum(low_h)
    low_p_s = low_s / np.sum(low_s)
    
    enh_h,enh_s,_ = cv2.split(enhancemnet_hsv_img)
    enh_h = np.asarray(enh_h) + 0.000001
    enh_s = np.asarray(enh_s) + 0.000001
    enh_p_h = enh_h / np.sum(enh_h)
    enh_p_s = enh_s / np.sum(enh_s)
    
    KL_h = np.sum(enh_p_h * np.log(enh_p_h / low_p_h))
    KL_s = np.sum(enh_p_s * np.log(enh_p_s / low_p_s))
    return KL_h, KL_s
   
def noise_measurement(enhancement_gray_img, low_mu, low_sigma):
    # 低對比度的區域
    lowContrastArea = low_sigma < np.mean(low_sigma)
    # 過暗的區域
    lowlightArea = low_mu < np.mean(low_mu)
    # 只計算dark and flat areas中出現noise像素數
    noise_area = lowContrastArea & lowlightArea
    
    win = cv2.getGaussianKernel(5, 1)
    img_gaussian = cv2.filter2D(enhancement_gray_img, -1, win)
    img_median = cv2.medianBlur(enhancement_gray_img, 3)
    
    M_g = enhancement_gray_img - img_gaussian
    M_m = enhancement_gray_img - img_median

    N_g = np.mean(M_g * noise_area)
    N_m = np.mean(M_m * noise_area)
    
    return N_g, N_m , lowContrastArea, lowlightArea

def structure(low_light_gray_img, enhancement_gray_img, low_mu, low_sigma, 
              enhancement_mu, enhancement_sigma, lowContrastArea, lowlightArea):
    # variance similarity
    sigmaSIM = (2 * low_sigma * enhancement_sigma) / (low_sigma**2 + enhancement_sigma**2 + 0.000001)
    sigmaSIM_mean = np.mean(sigmaSIM)
    
    # normalized local variance
    low_nv = low_sigma / (low_mu + 1 + 0.000001)
    enhancement_nv = enhancement_sigma / (enhancement_mu + 1 + 0.000001) 
    # normalized variance similarity
    nvSIM = (2 * low_nv * enhancement_nv) / (low_nv**2 + enhancement_nv**2 + 0.000001)
    nvSIM_mean = np.mean(nvSIM)
    
    # normalize image similarity
    low_nor_img = (low_light_gray_img - low_mu) / (low_sigma + 1) + 3
    # low_nor_img = NormalizeData(low_nor_img)
    enhancement_nor_img = (enhancement_gray_img - enhancement_mu) / (enhancement_sigma + 1) + 3
    # enhancement_nor_img = NormalizeData(enhancement_nor_img)
    imgnorSIM = (2 * low_nor_img * enhancement_nor_img) / (low_nor_img**2 + enhancement_nor_img**2 + 0.000001)
    imgnorSIM_mean = np.mean(imgnorSIM)
    
    # log edge similarity
    blur = cv2.GaussianBlur(low_light_gray_img, (3,3), 0)
    low_log_img = cv2.Laplacian(blur, -1)
    blur = cv2.GaussianBlur(enhancement_gray_img, (3,3), 0)
    enhancement_log_img = cv2.Laplacian(blur, -1)
    logSIM = (2 * low_log_img * enhancement_log_img) / (low_log_img**2 + enhancement_log_img**2 + 0.000001)
    logSIM_mean = np.mean(logSIM)
    
    # structure enhance in low light areas
    E_sigmaSIM = np.mean(sigmaSIM * lowlightArea)
    E_nvSIM = np.mean(nvSIM * lowlightArea)
    E_imgnorSIM = np.mean(imgnorSIM * lowlightArea)
    E_logSIM = np.mean(logSIM * lowlightArea)
    
    # over-enhacnement in low contrast areas
    highEnhancementArea = (enhancement_sigma - low_sigma) > np.mean(enhancement_sigma - low_sigma)
    area = lowContrastArea & highEnhancementArea
    O_sigmaSIM = np.mean(sigmaSIM * area)
    O_nvSIM = np.mean(nvSIM * area)
    O_imgnorSIM = np.mean(imgnorSIM * area)
    O_logSIM = np.mean(logSIM * area)
    
    return sigmaSIM_mean , nvSIM_mean, imgnorSIM_mean, logSIM_mean, E_sigmaSIM, E_nvSIM, E_imgnorSIM, E_logSIM, O_sigmaSIM, O_nvSIM, O_imgnorSIM, O_logSIM
    
def get_NLIEE_features(low_light_img, enhancement_img):
    # 18個特徵
    f = np.zeros((18))
    # low_light_img = low_light_img.astype(np.float32)
    # enhancement_img = enhancement_img.astype(np.float32)
    
    low_light_gray_img = cv2.cvtColor(low_light_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    enhancement_gray_img = cv2.cvtColor(enhancement_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    
    low_mu, low_sigma = pre_processing(low_light_img)
    enhancement_mu, enhancement_sigma = pre_processing(enhancement_img)
    
    f[0], f[1] = luminance_gaussian(enhancement_gray_img, enhancement_mu, enhancement_sigma)
    
    f[2], f[3] = color_comparison(low_light_img, enhancement_img)
    
    f[4], f[5], lowContrastArea, lowlightArea = noise_measurement(enhancement_gray_img, low_mu, low_sigma)
    
    f[6:18] = structure(low_light_gray_img, enhancement_gray_img, low_mu, low_sigma, enhancement_mu, enhancement_sigma, lowContrastArea, lowlightArea)
    
    return f