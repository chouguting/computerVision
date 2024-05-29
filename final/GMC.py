import cv2
import numpy as np
from util import *
from tqdm import tqdm
#pictures_dict紀錄的是每一個frame要參考的前後frame
#題目要求要用hierarchical B
pictures_dict = {
    16: [0, 32],
    8: [0, 16],
    4: [0, 8],
    2: [0, 4],
    1: [0, 2],
    3: [2, 4],
    6: [4, 8],
    5: [4, 6],
    7: [6, 8],
    12: [8, 16],
    10: [8, 12],
    9: [8, 10],
    11: [10, 12],
    14: [12, 16],
    13: [12, 14],
    15: [14, 16],
    24: [16, 32],
    20: [16, 24],
    18: [16, 20],
    17: [16, 18],
    19: [18, 20],
    22: [20, 24],
    21: [20, 22],
    23: [22, 24],
    28: [24, 32],
    26: [24, 28],
    25: [24, 26],
    27: [26, 28],
    30: [28, 32],
    29: [28, 30],
    31: [30, 32],
    48: [32, 64],
    40: [32, 48],
    36: [32, 40],
    34: [32, 36],
    33: [32, 34],
    35: [34, 36],
    38: [36, 40],
    37: [36, 38],
    39: [38, 40],
    44: [40, 48],
    42: [40, 44],
    41: [40, 42],
    43: [42, 44],
    46: [44, 48],
    45: [44, 46],
    47: [46, 48],
    56: [48, 64],
    52: [48, 56],
    50: [48, 52],
    49: [48, 50],
    51: [50, 52],
    54: [52, 56],
    53: [52, 54],
    55: [54, 56],
    60: [56, 64],
    58: [56, 60],
    57: [56, 58],
    59: [58, 60],
    62: [60, 64],
    61: [60, 62],
    63: [62, 64],
    80: [64, 96],
    72: [64, 80],
    68: [64, 72],
    66: [64, 68],
    65: [64, 66],
    67: [66, 68],
    70: [68, 72],
    69: [68, 70],
    71: [70, 72],
    76: [72, 80],
    74: [72, 76],
    73: [72, 74],
    75: [74, 76],
    78: [76, 80],
    77: [76, 78],
    79: [78, 80],
    88: [80, 96],
    84: [80, 88],
    82: [80, 84],
    81: [80, 82],
    83: [82, 84],
    86: [84, 88],
    85: [84, 86],
    87: [86, 88],
    92: [88, 96],
    90: [88, 92],
    89: [88, 90],
    91: [90, 92],
    94: [92, 96],
    93: [92, 94],
    95: [94, 96],
    112: [96, 128],
    104: [96, 112],
    100: [96, 104],
    98: [96, 100],
    97: [96, 98],
    99: [98, 100],
    102: [100, 104],
    101: [100, 102],
    103: [102, 104],
    108: [104, 112],
    106: [104, 108],
    105: [104, 106],
    107: [106, 108],
    110: [108, 112],
    109: [108, 110],
    111: [110, 112],
    120: [112, 128],
    116: [112, 120],
    114: [112, 116],
    113: [112, 114],
    115: [114, 116],
    118: [116, 120],
    117: [116, 118],
    119: [118, 120],
    124: [120, 128],
    122: [120, 124],
    121: [120, 122],
    123: [122, 124],
    126: [124, 128],
    125: [124, 126],
    127: [126, 128]
}
def main():
    for target, ref in tqdm(pictures_dict.items()):  #對於每一個 target frame及對應的參考frame
        img_name = '%03d'%target 
        ref1_name = '%03d'%ref[0] 
        ref2_name = '%03d'%ref[1]
        #從yuv_imgs資料夾中讀取圖片
        # (yuv_imgs資料夾中的圖片是用yuv2png.py轉換而來，裡面的圖片是影片裡的frame轉出來的RGB圖片)
        img = cv2.imread(f'./yuv_imgs/{img_name}.png', cv2.IMREAD_COLOR)
        img_ref1 = cv2.imread(f'./yuv_imgs/{ref1_name}.png', cv2.IMREAD_COLOR)
        img_ref2 = cv2.imread(f'./yuv_imgs/{ref2_name}.png', cv2.IMREAD_COLOR)
        transformed_imgs = []
        

        kp, kp1, kp2, matches, matches2 = feature_match(img,img_ref1,img_ref2, method='orb',feature_num=300) #使用orb特徵，提取出300個特徵點
        matrix_perspective, matrix_affine = get_matrix(kp1, kp, matches)  #利用matches,得到perspective matrix和affine matrix
        matrix_perspective2, matrix_affine2 = get_matrix(kp2, kp, matches2)
        t1, t2, t3 = get_transform_img(img_ref1, matrix_perspective, matrix_affine)
        transformed_imgs.append(t1)
        transformed_imgs.append(t2)
        transformed_imgs.append(t3)
        t4, t5, t6 = get_transform_img(img_ref2, matrix_perspective2, matrix_affine2)
        transformed_imgs.append(t4)
        transformed_imgs.append(t5)
        transformed_imgs.append(t6)

        kp, kp1, kp2, matches, matches2 = feature_match(img,img_ref1,img_ref2, method='sift',feature_num=300)
        matrix_perspective, matrix_affine = get_matrix(kp1, kp, matches,0.75)
        matrix_perspective2, matrix_affine2 = get_matrix(kp2, kp, matches2,0.75)

        t7, t8, t9 = get_transform_img(img_ref1, matrix_perspective, matrix_affine)
        transformed_imgs.append(t7)
        transformed_imgs.append(t8)
        transformed_imgs.append(t9)
        t10, t11, t12 = get_transform_img(img_ref2, matrix_perspective2, matrix_affine2)
        transformed_imgs.append(t10)
        transformed_imgs.append(t11)
        transformed_imgs.append(t12)

        idx_list = []
        for i in range(0,2160,16):
            for j in range(0,3840,16):
                block = img[i:i+16, j:j+16]
                block = cv2.cvtColor(block, cv2.COLOR_BGR2GRAY)
                # gray
                vals = []
                for k in range(len(transformed_imgs)):
                    block_transformed = transformed_imgs[k][i:i+16,j:j+16]
                    block_transformed = cv2.cvtColor(block_transformed, cv2.COLOR_BGR2GRAY)
                    # val = ssim(block, block_transformed)
                    val = psnr(block, block_transformed)
                    vals.append(val)
                val = max(vals)
                idx = np.argmax(vals)

                idx_list.append([val, idx, i, j])
                

        idx_list = sorted(idx_list, key=lambda x: x[0], reverse=True)
        img_out = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        selected_idxs = []
        for i in range(13000):
            img_out[idx_list[i][2]:idx_list[i][2]+16, idx_list[i][3]:idx_list[i][3]+16] = transformed_imgs[idx_list[i][1]][idx_list[i][2]:idx_list[i][2]+16, idx_list[i][3]:idx_list[i][3]+16]
            selected_idxs.append(idx_list[i])
        selected_idxs = sorted(selected_idxs, key=lambda x: x[2]*240+x[3])
        img_out_gray = cv2.cvtColor(img_out, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(f'./out_imgs/{img_name}.png', img_out_gray)
        out_selected(selected_idxs, f'./out_imgs/s_{img_name}.txt')
        out_model(selected_idxs, f'./model_map/{img_name}_model.txt')
if __name__ == '__main__':
    main()