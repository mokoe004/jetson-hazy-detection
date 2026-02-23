import os
import cv2
import numpy as np
import numpy as np
import cv2


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

def calculate_ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = rgb2gray(img1).astype(np.float64)
    img2 = rgb2gray(img2).astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    #means
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2

    #standard deviations
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq

    #convariances
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    #SSIM formula
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


# Paths to the folders
foggy_folder = 'path_to_foggy_images_directory'
dehazed_folder = 'path_to_dehazed_images_directory'

# Function to calculate PSNR
def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:  # MSE is zero means no noise is present in the signal.
                 # Therefore PSNR have no importance.
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

# def calculate_ssim(img1, img2):
#     ssim_value, _ = ssim(img1, img2, full=True, multichannel=True)
#     return ssim_value
# Initialize sums for PSNR and SSIM
total_psnr = 0
total_ssim = 0
count = 0

# Iterate through each image in the foggy folder and calculate PSNR and SSIM with the corresponding image in the dehazed folder
for foggy_image_name in os.listdir(foggy_folder):
    foggy_image_path = os.path.join(foggy_folder, foggy_image_name)
    dehazed_image_path = os.path.join(dehazed_folder, foggy_image_name)
    
    if os.path.exists(dehazed_image_path):
        foggy_image = cv2.imread(foggy_image_path)
        dehazed_image = cv2.imread(dehazed_image_path)
        
        # Resize dehazed image to match the size of the foggy image
        dehazed_image = cv2.resize(dehazed_image, (foggy_image.shape[1], foggy_image.shape[0]))
        
        psnr_value = calculate_psnr(foggy_image, dehazed_image)
        ssim_value = calculate_ssim(foggy_image, dehazed_image)
        
        total_psnr += psnr_value
        total_ssim += ssim_value
        count += 1

# Calculate average PSNR and SSIM
average_psnr = total_psnr / count if count > 0 else 0
average_ssim = total_ssim / count if count > 0 else 0

# Print the results
print(f'Average PSNR: {average_psnr:.2f}')
print(f'Average SSIM: {average_ssim:.4f}')