# 基於自監督學習之零參考樣本的低光影像增強與雜訊抑制之研究 <br> A-Study-on-Zero-Reference-Image-Enhancement-with-Noise-Reduction-Based-on-Self-Supervised-Learning

# 簡介
本專案利用self supervised learning的方式建構了一種無需正常光影像作為參考的低光影像增強系統，系統由Image Enhancement Module 與 Noise Reduction Module組成

# 網路架構
![image](img/1.png)

# 實驗結果圖
LOL Dataset : https://daooshee.github.io/BMVC2018website/
| 增強前 | 增強後 |
| :----: | :----: |
| ![image](img/Visual_Quality_Evaluation/low/1.png) | ![image](img/Visual_Quality_Evaluation/enhanced/1.png) |
| ![image](img/Visual_Quality_Evaluation/low/111.png) | ![image](img/Visual_Quality_Evaluation/enhanced/111.png) |
| ![image](img/Visual_Quality_Evaluation/low/22.png) | ![image](img/Visual_Quality_Evaluation/enhanced/22.png) |
| ![image](img/Visual_Quality_Evaluation/low/547.png) | ![image](img/Visual_Quality_Evaluation/enhanced/547.png) |

測試低光影像增強系統對人臉偵測([RetinaFace](https://github.com/peteryuX/retinaface-tf2))的影響
| 增強前 | 增強後 |
| :----: | :----: |
| ![image](img/face_detection/low/11.png) | ![image](img/face_detection/enhanced/11.png) |
| ![image](img/face_detection/low/16.png) | ![image](img/face_detection/enhanced/16.png) |
| ![image](img/face_detection/low/17.png) | ![image](img/face_detection/enhanced/17.png) |
| ![image](img/face_detection/low/22.png) | ![image](img/face_detection/enhanced/22.png) |
| ![image](img/face_detection/low/85.png) | ![image](img/face_detection/enhanced/85.png) |
| ![image](img/face_detection/low/95.png) | ![image](img/face_detection/enhanced/95.png) |

# 實驗結果
| Method | SSIM |	PSNR(dB) | Parameter(M) | Speed/ms |
| :----: | :--: |	:------: | :----------: | :------: |
| Input  |	0.195 |	7.77 | - |	- |
| HE     |0.496	| 14.10	| -	| 6.28 |
| LIME   | 0.514 | 14.22 | - | 2466 |
| RetinexNet | 0.419 | 16.77 | 0.55 | 216 |
| EnlightenGAN | 0.651 | 17.48 | 8.64 | 385 |
| Zero-DCE | 0.532 | 16.88 | 0.07 | 48 |
| Proposed | 0.734 | 17.58 | 0.34 | 95 |

# 論文
撰寫中...
