import numpy as np
import cv2
import argparse
import os
from JBF import Joint_bilateral_filter


def main():
    parser = argparse.ArgumentParser(description='main function of joint bilateral filter')
    parser.add_argument('--image_path', default='./testdata/2.png', help='path to input image')
    parser.add_argument('--setting_path', default='./testdata/2_setting.txt', help='path to setting file')
    args = parser.parse_args()

    img = cv2.imread(args.image_path)
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # rgb to y channel with the formula: Y = a * R + b * G + c * B
    # with a, b, c =
    # 0.1,0.0,0.9
    # 0.2,0.0,0.8
    # 0.2,0.8,0.0
    # 0.4,0.0,0.6
    # 1.0,0.0,0.0
    image_y_1 = (0.1 * img_rgb[:,:,0] + 0.0 * img_rgb[:,:,1] + 0.9 * img_rgb[:,:,2]).clip(0, 255).astype(np.uint8)
    image_y_2 = (0.2 * img_rgb[:,:,0] + 0.0 * img_rgb[:,:,1] + 0.8 * img_rgb[:,:,2]).clip(0, 255).astype(np.uint8)
    image_y_3 = (0.2 * img_rgb[:,:,0] + 0.8 * img_rgb[:,:,1] + 0.0 * img_rgb[:,:,2]).clip(0, 255).astype(np.uint8)
    image_y_4 = (0.4 * img_rgb[:,:,0] + 0.0 * img_rgb[:,:,1] + 0.6 * img_rgb[:,:,2]).clip(0, 255).astype(np.uint8)
    image_y_5 = (1.0 * img_rgb[:,:,0] + 0.0 * img_rgb[:,:,1] + 0.0 * img_rgb[:,:,2]).clip(0, 255).astype(np.uint8)




    ### TODO ###
    img_guidance = image_y_1
    jbf = Joint_bilateral_filter(1, 0.05 )
    img_origin_filtered = jbf.joint_bilateral_filter(img_rgb, img_rgb)
    img_guidance_filtered = jbf.joint_bilateral_filter(img_rgb, img_guidance)
    img_guidance_filtered_bgr = cv2.cvtColor(img_guidance_filtered, cv2.COLOR_RGB2BGR)
    cv2.imwrite('output.png', img_guidance_filtered_bgr)
    cv2.imwrite('output_gray.png', img_guidance)

    #calculate the cost(L1 norm)
    cost = np.sum(np.abs(img_guidance_filtered.astype('int32')-img_origin_filtered.astype('int32')))
    print('Cost: {}'.format(cost))




if __name__ == '__main__':
    main()