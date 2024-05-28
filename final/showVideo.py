import cv2
#read video
import cv2
import numpy as np
import os
import subprocess as sp

# Build synthetic video and read binary data into memory (for testing):
#########################################################################
yuv_filename = 'DaylightRoad2_27.yuv'
width, height = 1280, 720
fps = 1 # 1Hz (just for testing)

# Build synthetic video, for testing (the mp4 is used just as reference):
# sp.run('ffmpeg -y -f lavfi -i testsrc=size={}x{}:rate=1 -vcodec libx264 -crf 18 -t 10 {}'.format(width, height, mp4_filename))
sp.run('ffmpeg -y -f lavfi -i testsrc=size={}x{}:rate=1 -pix_fmt yuv420p -t 10 {}'.format(width, height, yuv_filename))
#########################################################################


file_size = os.path.getsize(yuv_filename)

# Number of frames: in YUV420 frame size in bytes is width*height*1.5
n_frames = file_size // (width*height*3 // 2)

# Open 'input.yuv' a binary file.
f = open(yuv_filename, 'rb')

for i in range(n_frames):
    # Read Y, U and V color channels and reshape to height*1.5 x width numpy array
    yuv = np.frombuffer(f.read(width*height*3//2), dtype=np.uint8).reshape((height*3//2, width))

    # Convert YUV420 to BGR (for testing), applies BT.601 "Limited Range" conversion.
    bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)

    # Convert YUV420 to Grayscale
    gray = cv2.cvtColor(yuv, cv2.COLOR_YUV2GRAY_I420)

    #Show RGB image and Grayscale image for testing
    cv2.imshow('rgb', bgr)
    cv2.waitKey(500)  # Wait a 0.5 second (for testing)
    cv2.imshow('gray', gray)
    cv2.waitKey(500)  # Wait a 0.5 second (for testing)

f.close()

cv2.destroyAllWindows()