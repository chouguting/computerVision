conda create --name cv2024_Final python==3.10
conda activate cv2024_Final
pip install -r requirements.txt
mkdir gray_imgs
mkdir yuv_imgs
# python3 yuv2png.py --yuv_file ./DaylightRoad2_27.yuv --output_dir ./gray_imgs
# python3 yuv2yuv_png.py --yuv_file ./DaylightRoad2_27.yuv --output_dir ./yuv_imgs
mkdir out_imgs
mkdir model_map
python3 GMC.py
python3 eval.py