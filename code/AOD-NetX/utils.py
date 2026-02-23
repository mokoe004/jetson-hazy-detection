import numpy as np
import cv2

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def ssim(img1, img2):
    C1 = (0.01*255)**2
    C2 = (0.03*255)**2

    img1 = rgb2gray(img1).astype(np.float64)
    img2 = rgb2gray(img2).astype(np.float64)

    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1*mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1*img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_psnr(img1, img2):
    # Convert images to the same data type (float64)
    img1 = rgb2gray(img1).astype(np.float64)
    img2 = rgb2gray(img2).astype(np.float64)

    # Compute Mean Squared Error (MSE)
    mse = np.mean((img1 - img2) ** 2)
    
    # Check if MSE is zero
    if mse == 0:
        return float('inf')  # PSNR is infinite if MSE is zero

    # Calculate PSNR
    psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    return psnr



    
