# 1. Preparation
- make sure conda is available
# 2. Enviroment Create
- create and activate conda enviroment
```
conda create --name cv2024_Final python==3.10
conda activate cv2024_Final
```
# 3. Install package
- Package :
    - Numpy
    - Opencv-python   
    - tqdm
- Type below command
```
pip install -r requirements.txt
```
# 4. yuv & gray image generation (如果還沒產生)
- 先把DaylightRoad2_27.yuv放到資料夾中
- 建立資料夾用來存放gray images 
`mkdir gray_imgs`
- 建立資料夾用來存放yuv images
`mkdir yuv_imgs`
- 利用yuv2png.py來把DaylightRoad2_27.yuv轉成灰階影像並存到gray_imgs資料夾
`python3 yuv2png.py --yuv_file ./DaylightRoad2_27.yuv --output_dir ./gray_imgs`
- 利用yuv2yuv_png.py來把DaylightRoad2_27.yuv轉成yuv影像並存到yuv_imgs資料夾 (要花一段時間)
`python3 yuv2yuv_png.py --yuv_file ./DaylightRoad2_27.yuv --output_dir ./yuv_imgs`
# 5. 利用GMC.py來進行global motion compensation
- 先建立fold來存放輸出的images和model map
`mkdir out_imgs`
`mkdir model_map`
- 執行GMC.py，GMC.py會調用util.py的function，對於兩張reference pictures用sift, orb等方法抽取特徵，再以perspective或 affine等transform，然後在每個16X16的區塊，根據12種不同的transform結果，找出PSNR最大的，能產生最大PSNR的transform作為這個區塊的motion model，然後選出32400個區塊中PSNR最大的13000個
`python3 GMC.py`
- 利用eval.py來評估所有輸出的images的PSNR
`python3 eval.py --so_path ./out_imgs --gt_path ./gray_imgs`

## 補充
- 用的12個motion model的產生方式分別為
    1. 用reference1和target 取sift特徵，再用brute force matching所得出的perspective matrix，用這個perspective matrix進行的transformation
    2. 用reference1和target 取sift特徵，再用brute force matching所得出的affine matrix，用這個affine matrix進行的transformation
    3. model1和model2的結果以0.5權重混合
    4. 用reference2和target 取sift特徵，再用brute force matching所得出的perspective matrix，用這個perspective matrix進行的transformation
    5. 用reference2和target 取sift特徵，再用brute force matching所得出的affine matrix，用這個affine matrix進行的transformation
    6. model 4和model 5的結果以0.5權重混合
    7. 用reference1和target 取orb特徵，再用brute force matching所得出的perspective matrix，用這個perspective matrix進行的transformation
    8. 用reference1和target 取orb特徵，再用brute force matching所得出的affine matrix，用這個affine matrix進行的transformation
    9. model1和model2的結果以0.5權重混合
    10. 用reference2和target 取orb特徵，再用brute force matching所得出的perspective matrix，用這個perspective matrix進行的transformation
    11. 用reference2和target 取orb特徵，再用brute force matching所得出的affine matrix，用這個affine matrix進行的transformation
    12. model 4和model 5的結果以0.5權重混合
# 一次跑全部 (不含yuv & gray image generation )
`source run.sh`