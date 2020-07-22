import numpy as np
import cv2

def masked_normxcorr(img, templ, mask):
    templ *= mask
    mask_sum = mask.sum()
    templ_sum = templ.sum()
    templ_mean = templ_sum/mask_sum
    templ_norm = (templ - templ_mean)*mask
    templ_norm_sum = templ_norm.sum()
    sum_img = cv2.filter2D(img, -1, mask, borderType=cv2.BORDER_CONSTANT)
    mean_img = sum_img/mask_sum
    Q1 = cv2.filter2D(img, -1, templ_norm, borderType=cv2.BORDER_CONSTANT)
    Q2 = mean_img*templ_norm_sum
    Q = Q1 - Q2
    D1 = cv2.filter2D(img*img, -1, mask, borderType=cv2.BORDER_CONSTANT)
    D2 = -2*mean_img*sum_img
    D3 = mask_sum*(mean_img*mean_img)
    D4 = (templ_norm*templ_norm).sum()
    D = np.sqrt(D1 + D2 + D3 + 1e-7)*np.sqrt(D4)
    return Q/D
