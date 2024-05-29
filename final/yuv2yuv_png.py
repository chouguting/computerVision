import os, sys, argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2


def convert2yuv(yuv_file, output_dir):
    f_y = open(yuv_file, "rb")
    w ,h = 3840, 2160
    seq_len = 129
    frame_size = int(3/2 * w * h)
    for frame_num in range(seq_len):
        pixels = np.zeros((w, h, 3), dtype=np.uint8)

        f_y.seek(frame_size * frame_num)
        
        for i in range(h):
            for j in range(w):
                y = ord(f_y.read(1))
                pixels[j,i,0] = int(y)
        for i in range(h//2):
            for j in range(w//2):
                u = ord(f_y.read(1))
                pixels[j*2,i*2,1] = int(u)
                pixels[j*2+1,i*2,1] = int(u)
                pixels[j*2,i*2+1,1] = int(u)
                pixels[j*2+1,i*2+1,1] = int(u)
        for i in range(h//2):
            for j in range(w//2):
                v = ord(f_y.read(1))
                pixels[j*2,i*2,2] = int(v)
                pixels[j*2+1,i*2,2] = int(v)
                pixels[j*2,i*2+1,2] = int(v)
                pixels[j*2+1,i*2+1,2] = int(v)
        # pixels = pixels.reshape((h,w,3))
        ## rotate
        pixels = np.rot90(pixels, 3)
        pixels = np.flip(pixels, 1)
        pixels = cv2.cvtColor(pixels, cv2.COLOR_YUV2RGB)
        plt.imsave(os.path.join(output_dir, '%03d.png' % frame_num), pixels)

    f_y.close()
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-y', '--yuv_file', type=str)
    parser.add_argument('-o', '--output_dir', type=str)
    args = parser.parse_args()

    yuv_file, output_dir = args.yuv_file, args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # convert(yuv_file, output_dir)
    convert2yuv(yuv_file, output_dir)