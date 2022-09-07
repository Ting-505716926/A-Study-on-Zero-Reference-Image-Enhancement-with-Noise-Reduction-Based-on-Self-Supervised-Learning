'''
generalized Gaussian distribution,GGD
<< Estimation of Shape Parameter for Generalized Gaussian Distributions in Subband Decompositions of Video >>
'''

import numpy as np
from scipy.special import gamma

def GGD(vec):
    # 產生候選的 γ 
    gam = np.arange(0.2, 10.0, 0.001)
    
    # 根據候選 γ 計算 r(γ)
    r_gam = (gamma(1/gam)*gamma(3/gam))/((gamma(2/gam))**2)
    
    # σ^2 的零均值估計，非零均值需要計算 然後按照公式（3）
    
    sigma_sq = np.mean((vec)**2)
    sigma = np.sqrt(sigma_sq)
    
    E = np.mean(np.abs(vec))
    
    # 根據 sigma 和 E 計算 r(γ)
    r = sigma_sq / (E**2)
    
    diff = np.abs(r - r_gam)
    gamma_param = gam[np.argmin(diff, axis=0)]
    
    return sigma, gamma_param
    
    
    

