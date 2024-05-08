import numpy as np
import cv2
import random
from tqdm import tqdm
from utils import solve_homography, warping

random.seed(999)

def panorama(imgs):
    """
    Image stitching with estimated homograpy between consecutive
    :param imgs: list of images to be stitched
    :return: stitched panorama
    """
    h_max = max([x.shape[0] for x in imgs])
    w_max = sum([x.shape[1] for x in imgs])

    # create the final stitched canvas
    dst = np.zeros((h_max, w_max, imgs[0].shape[2]), dtype=np.uint8)
    dst[:imgs[0].shape[0], :imgs[0].shape[1]] = imgs[0]
    last_best_H = np.eye(3)
    out = None

    N = 3000 #RANSAC的次數
    K = 11 #隨機取出的座標數
    MATCH_COUNT = 50 #取前MATCH_COUNT個最佳的matches
    INLIER_THRESHOLD = 0.21 #inlier的threshold

    # for all images to be stitched:
    for idx in tqdm(range(len(imgs)-1)):
        im1 = imgs[idx]
        im2 = imgs[idx + 1]

        # TODO: 1.feature detection & matching
        # 用ORB找特徵點
        orb = cv2.ORB_create()
        #利用ORB抓出keypoints和descriptors
        #im1是trainImage(原圖), im2是queryImage(要準備接上去的圖)
        keyPoints1, descriptors1 = orb.detectAndCompute(im1, None)
        keyPoints2, descriptors2 = orb.detectAndCompute(im2, None)
        #用BFMatcher找出最佳的matches
        bfMatcher = cv2.BFMatcher(cv2.NORM_HAMMING , crossCheck=True)
        #這邊query是im1, train是im2
        matches = bfMatcher.match(descriptors2, descriptors1)
        #找出最佳的matches,距離小的排前面
        matches = sorted(matches, key=lambda x: x.distance)
        bestMatches = matches[:MATCH_COUNT] #取前MATCH_COUNT個最佳的matches
        #取出keypoints的index，並用來找出對應的座標
        sourceKeyPoints = np.array([keyPoints2[match.queryIdx].pt for match in bestMatches])
        destinationKeyPoints = np.array([keyPoints1[match.trainIdx].pt for match in bestMatches])
        maxInliers = 0
        bestInliersIndexSet = None

        # TODO: 2. apply RANSAC to choose best H
        #RANSAC要執行N次
        for i in range(N):
            #隨機取出K個座標，用來計算homography
            randomIdx = random.sample(range(len(sourceKeyPoints)), K) #隨機取出K個座標
            selectedSource = sourceKeyPoints[randomIdx]
            selectedDestination = destinationKeyPoints[randomIdx]
            #計算homography
            H = solve_homography(selectedSource, selectedDestination)
            #用homography對source做變換
            # 先做transpose，讓sourceKeyPoints變成2xMATCH_COUNT的矩陣
            # 再把座標轉成homography coordinate，
            # 本來sourceKeyPoints是2xMATCH_COUNT的矩陣，現在要變成3xMATCH_COUNT的矩陣
            sourceKeyPointsHomography = np.concatenate((sourceKeyPoints.T, np.ones((1, sourceKeyPoints.shape[0]))), axis=0)
            # 用homography做變換
            # H是3x3的矩陣
            # sourceKeyPointsHomography是3xMATCH_COUNT的矩陣，相當於有MATCH_COUNT個 1x3 的vector
            # 用np.dot(H, source)可以得到3xMATCH_COUNT的矩陣，相當於有MATCH_COUNT個 1x3 的vector結果
            transformedSourceKeyPoints = np.dot(H, sourceKeyPointsHomography) #3xMATCH_COUNT的矩陣
            #轉回2D座標(非homography coordinate)，並也做transpose，讓transformedSourceKeyPoints變成MATCH_COUNTx2的矩陣
            transformedSourceKeyPoints = (transformedSourceKeyPoints / transformedSourceKeyPoints[2]).T[:, :2]
            #計算error
            error = np.linalg.norm(transformedSourceKeyPoints - destinationKeyPoints, axis=1)
            #計算inliers
            inliers = (error < INLIER_THRESHOLD).sum()
            #如果inliers比較多，則更新best_H
            if inliers > maxInliers:
                best_H = H.copy()
                maxInliers = inliers
                bestInliersIndexSet = np.where(error < INLIER_THRESHOLD)

        #用所有的inliers重新計算homography (Opimal Estimation)
        bestSource = sourceKeyPoints[bestInliersIndexSet]
        bestDestination = destinationKeyPoints[bestInliersIndexSet]
        best_H = solve_homography(bestSource, bestDestination)

        # TODO: 3. chain the homographies
        #將所有的homography相乘，這邊是將上一次的best_H和這次的best_H接起來
        # 新圖片(img2)乘上這個矩陣時是做左乘運算，也就是 last_best_H(上次的) * best_H * img2
        # 由右往左依序就是先對img2做這次的best_H，轉換到img1的座標系統，
        # 再對從img1系統做上次的last_best_H，轉換到更前面圖片的座標系統
        last_best_H = np.dot(last_best_H, best_H)
        # TODO: 4. apply warping
        #將im2接到im1的右邊，由於im1已經接到dst上了，所以這邊只要將im2接到dst上即可
        #xMin, yMin, xMax, yMax代表的是做warp以後，圖片能夠到達的最大範圍
        #因為是backward warp，所以xMin, yMin, xMax, yMax是用destination的座標來決定(也就是整個dst_image的範圍)
        out = warping(im2, dst, last_best_H, 0, h_max, 0, w_max, direction='b')
    return out 

if __name__ == "__main__":
    # ================== Part 4: Panorama ========================
    # TODO: change the number of frames to be stitched
    FRAME_NUM = 3
    imgs = [cv2.imread('../resource/frame{:d}.jpg'.format(x)) for x in range(1, FRAME_NUM + 1)]
    output4 = panorama(imgs)
    cv2.imwrite('output4.png', output4)