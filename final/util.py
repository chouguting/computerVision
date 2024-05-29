import cv2
import numpy as np
 
def feature_match(img,img_ref1,img_ref2,feature_num,method='orb'):
    """
    從image, img_ref1, img_ref2中提取特徵點並進行特徵匹配
    :param img : input image
    :param img_ref1 : reference image 1
    :param img_ref2 : reference image 2
    :param feature_num : number of features to extract
    :param method : feature extraction method
    :return kp : keypoints of input image
    :return kp1 : keypoints of reference image 1
    :return kp2 : keypoints of reference image 2
    :return matches : matches from reference image 1 to input image 
    :return matches2 : matches from reference image 2 to input image 
    """
    if method == 'sift':
        feature = cv2.SIFT_create(nfeatures=feature_num,nOctaveLayers=5,contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)
    elif method == 'akaze':
        feature = cv2.AKAZE_create()
    elif method == 'orb':
        feature = cv2.ORB_create()
    kp, descriptors   = feature.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None)
    kp1, descriptors1 = feature.detectAndCompute(cv2.cvtColor(img_ref1, cv2.COLOR_BGR2GRAY), None)
    kp2, descriptors2 = feature.detectAndCompute(cv2.cvtColor(img_ref2, cv2.COLOR_BGR2GRAY), None)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(descriptors1, descriptors, k=2)
    matches2 = bf.knnMatch(descriptors2, descriptors, k=2)
    return kp, kp1, kp2, matches, matches2
def get_matrix(kp, kp2, matches, threshold=0.75):
    """
    param kp : keypoints of input image
    param kp2 : keypoints of reference image
    param matches : matches from input image to reference image
    param threshold : threshold for matching
    return matrix_perspective : perspective matrix from input image to reference image
    return matrix_affine : affine matrix from input image to reference image
    """
    good = []
    for m, n in matches:
        if m.distance < threshold * n.distance:
            good.append(m)
    src_pts = np.float32([kp[m.queryIdx].pt for m in good])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good])
    matrix_perspective, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
    matrix_affine, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
    return matrix_perspective, matrix_affine
def get_transform_img(img, matrix_perspective, matrix_affine):
    transformed_img_perspective = cv2.warpPerspective(img, matrix_perspective, (img.shape[1], img.shape[0]))
    transformed_img_affine = cv2.warpAffine(img, matrix_affine, (img.shape[1], img.shape[0]))
    transformed_img_add = cv2.addWeighted(transformed_img_perspective, 0.5, transformed_img_affine, 0.5, 0)
    return transformed_img_perspective, transformed_img_affine, transformed_img_add

def mse(imageA, imageB):
    # 計算兩個影像之間的均方誤差
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

def psnr(imageA, imageB):
    # 計算 PSNR
    mse_value = mse(imageA, imageB)
    if mse_value == 0:
        return float('inf')  # 如果 MSE 是 0，PSNR 無窮大
    max_pixel = 255.0
    return 10 * np.log10(max_pixel**2 / mse_value)

def mse1D(imageA, imageB):
    # 計算兩個影像之間的均方誤差
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(len(imageA))
    return err
def psnr1D(imageA, imageB):
    # 計算 PSNR
    mse_value = mse1D(imageA, imageB)
    if mse_value == 0:
        return float('inf')  # 如果 MSE 是 0，PSNR 無窮大
    max_pixel = 255.0
    return 10 * np.log10(max_pixel**2 / mse_value)

def out_selected(sorted_idx_list, file_path):
    file = open(file_path, 'w+')
    out_idxs = np.zeros((32400,1), dtype=int)
    for i in range(0, 13000):
        val = int(sorted_idx_list[i][2]/16)*240+int(sorted_idx_list[i][3]/16)
        out_idxs[val] = 1
    for i in range(0, 32400):
        line = f'{str(out_idxs[i][0])}'
        file.write(line + '\n')
    file.close()
def out_model(sorted_idx_list, file_path):
    file = open(file_path, 'w+')
    out_idxs = np.zeros((32400,1), dtype=int)
    for i in range(0, 13000):
        val = int(sorted_idx_list[i][2]/16)*240+int(sorted_idx_list[i][3]/16)
        out_idxs[val] = sorted_idx_list[i][1]+1
    for i in range(0, 32400):
        line = f'{str(out_idxs[i][0])}'
        file.write(line + '\n')
    file.close()