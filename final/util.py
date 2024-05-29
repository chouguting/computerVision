import cv2
import numpy as np
 
def feature_match(img_target, img_ref1, img_ref2, feature_num, method='orb'):
    """
    從image, img_ref1, img_ref2中提取特徵點並進行特徵匹配
    :param img_target : input image
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
        feature = cv2.SIFT_create(nfeatures=feature_num,nOctaveLayers=5,contrastThreshold=0.04, edgeThreshold=10, sigma=1.6) # 創建 SIFT 特徵檢測器，並設置參數
    elif method == 'akaze':
        feature = cv2.AKAZE_create()
    elif method == 'orb':
        feature = cv2.ORB_create()
    # 先將圖片轉為灰階，再進行特徵提取
    kp_target, descriptors_target   = feature.detectAndCompute(cv2.cvtColor(img_target, cv2.COLOR_BGR2GRAY), None) # target frame的特徵點和描述子
    kp_ref1, descriptors_ref1 = feature.detectAndCompute(cv2.cvtColor(img_ref1, cv2.COLOR_BGR2GRAY), None) # 第一個reference frame的特徵點和描述子
    kp_ref2, descriptors_ref2 = feature.detectAndCompute(cv2.cvtColor(img_ref2, cv2.COLOR_BGR2GRAY), None) # 第二個reference frame的特徵點和描述子
    bf = cv2.BFMatcher(cv2.NORM_L2)  # 創建Brute-force descriptor matcher， 使用 L2 距離
    matches = bf.knnMatch(descriptors_ref1, descriptors_target, k=2)  # 進行特徵匹配，這邊會得到很多個target frame和第一個reference frame的特徵點匹配
    matches2 = bf.knnMatch(descriptors_ref2, descriptors_target, k=2) # 進行特徵匹配，這邊會得到很多個target frame和第二個reference frame的特徵點匹配
    #因為是用knnMatch，所以對於每個reference frame特徵點 (query)，會得到k個最近的target frame (train)特徵點，
    #又因這邊取k=2，所以matches裡面的每一個元素都是兩個DMatch對象(比較接近的match排前面)，也因此這兩個DMatch對象的queryIdx都是同一個，但trainIdx不同
    #每個DMatch對象，包含了queryIdx、trainIdx及distance，分別代表了reference frame的特徵點index、target frame的特徵點index及這兩個特徵點的距離
    #假如你想取matches裡面的第i個元素的第一個DMatch對象所對應的target frame的特徵點，可以用kp[matches[i][0].trainIdx]
    return kp_target, kp_ref1, kp_ref2, matches, matches2 # 回傳特徵點和特徵匹配結果
def get_matrix(kp, kp2, matches, threshold=0.75):
    """
    param kp : keypoints of input image
    param kp2 : keypoints of reference image
    param matches : matches from input image to reference image
    param threshold : threshold for matching
    return matrix_perspective : perspective matrix from input image to reference image
    return matrix_affine : affine matrix from input image to reference image
    """
    good = [] #用來存放好的match
    for m, n in matches: #因為matches是用knnMatch得到的，而且k=2，所以每個元素都是兩個DMatch對象
        #因為這兩個Dmatch是用distance排序的，所以第一個的distance一定比第二個小
        #我們可以利用這個特性來過濾掉不好的match
        if m.distance < threshold * n.distance: #如果第一個DMatch對象的distance比第二個小很多，就代表這個match是好的
            good.append(m) #把這個match加入good list
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