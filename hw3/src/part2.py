import numpy as np
import cv2
from cv2 import aruco
from tqdm import tqdm
from utils import solve_homography, warping


def planarAR(REF_IMAGE_PATH, VIDEO_PATH):
    """
    Reuse the previously written function "solve_homography" and "warping" to implement this task
    :param REF_IMAGE_PATH: path/to/reference/image
    :param VIDEO_PATH: path/to/input/seq0.avi
    """
    video = cv2.VideoCapture(VIDEO_PATH)
    ref_image = cv2.imread(REF_IMAGE_PATH)
    h, w, c = ref_image.shape
    film_h, film_w = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    film_fps = video.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    videowriter = cv2.VideoWriter("output2.avi", fourcc, film_fps, (film_w, film_h))
    arucoDict = aruco.Dictionary_get(aruco.DICT_4X4_50)
    arucoParameters = aruco.DetectorParameters_create()
    ref_corns = np.array([[0, 0], [w, 0], [w, h], [0, h]])

    # TODO: find homography per frame and apply backward warp
    pbar = tqdm(total = 353)
    while (video.isOpened()):
        ret, frame = video.read()
        if ret:  ## check whethere the frame is legal, i.e., there still exists a frame
            # TODO: 1.find corners with aruco
            # function call to aruco.detectMarkers()
            #corners是aruco marker的座標(可能有多個)
            corners, arucoMarkerIds, rejected = aruco.detectMarkers(frame, arucoDict, parameters=arucoParameters)
            if len(corners) < 1:
                videowriter.write(frame)
                pbar.update(1)
                continue
            #本次作業中，只有一個aruco marker
            cornersOnImage = corners[0][0].astype(int)

            # TODO: 2.find homograpy
            # function call to solve_homography()
            #ref_corns是已知的座標(在還沒有任何變形之前)
            #corner是aruco marker的座標 (在變形之後，也就是拍到的情況下)
            #因為我們想要把一張圖片貼到ArUco marker上，所以我們要找的是從ref_corns到cornersOnImage的homography
            H = solve_homography(ref_corns, cornersOnImage)

            # TODO: 3.apply backward warp
            # function call to warping()
            #因為是backward warp，所以xMin, yMin, xMax, yMax是用destination的座標來決定(也就是cornersOnImage)
            xMin, yMin = np.min(cornersOnImage, axis=0)
            xMax, yMax = np.max(cornersOnImage, axis=0)
            frame = warping(ref_image, frame, H, yMin, yMax, xMin, xMax, direction='b')

            videowriter.write(frame)
            pbar.update(1)

        else:
            break

    pbar.close()
    video.release()
    videowriter.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":

    # ================== Part 2: Marker-based planar AR========================
    VIDEO_PATH = '../resource/seq0.mp4'
    # TODO: you can change the reference image to whatever you want
    REF_IMAGE_PATH = '../resource/arknights.png' 
    planarAR(REF_IMAGE_PATH, VIDEO_PATH)