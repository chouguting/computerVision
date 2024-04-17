import json
import os
import sys
import time
import argparse

import torch
from tqdm import tqdm

from model import MyNet, ResNet18
from dataset import get_dataloader
from utils import write_csv

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_datadir', help='test dataset directory', type=str, default='../hw2_data/p2_data/unlabel/') #從unlabel拿資料
    parser.add_argument('--model_type', help='mynet or resnet18', type=str, default='resnet18') #選擇模型
    parser.add_argument('--output_path', help='output csv file path', type=str, default='./output/pred_semi.csv') #輸出的csv檔
    args = parser.parse_args()

    model_type = args.model_type
    test_datadir = args.test_datadir
    output_path = args.output_path

    jsonResult = {}
    with open(os.path.join("../hw2_data/p2_data/train/", 'annotations.json'), 'r') as f: #先從label好的資料集拿annotations.json
        jsonResult = json.load(f)



    # jsonResult = {
    #     'filenames': [],
    #     'labels': []
    # }

    # Check if GPU is available, otherwise CPU is used
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    ##### MODEL #####
    ##### TODO: check model.py #####
    ##### NOTE: Put your best trained models to checkpoint/ #####
    if model_type == 'mynet':
        model = MyNet()
        model.load_state_dict(torch.load('./checkpoint/mynet_best.pth', map_location=torch.device('cpu')))
    elif model_type == 'resnet18':
        model = ResNet18()
        model.load_state_dict(torch.load('./checkpoint/resnet18_best.pth', map_location=torch.device('cpu')))
    else:
        raise NameError('Unknown model type')
    model.to(device)

    ##### DATALOADER #####
    ##### TODO: check dataset.py #####
    test_loader = get_dataloader(test_datadir, batch_size=1, split='test')

    ##### INFERENCE #####
    predictions = []
    model.eval()
    with torch.no_grad():
        test_start_time = time.time()
        #############################################################
        # TODO:                                                     #
        # Finish forward part in inference process, similar to      #
        # validation, and append predicted label to 'predictions'   #
        # list.                                                     #
        #                                                           #
        # NOTE:                                                     #
        # You don't have to calculate accuracy and loss since you   #
        # don't have labels.                                        #
        #############################################################
        index = 0
        for data in tqdm(test_loader):
            images = data['images'].to(device)
            outputs = model(images)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted = torch.argmax(outputs, 1)
            max, argmax = torch.max(outputs, 1)
            probabilitiesMax, probabilitiesArgmax = torch.max(probabilities, 1)

            if(probabilitiesMax>0.98):
                #add to json
                jsonResult['filenames'].append(test_loader.dataset.image_names[index])
                jsonResult['labels'].append(argmax.cpu().numpy().item())


            if predicted != probabilitiesArgmax:
                print("error")

            predictions.extend(predicted.cpu().numpy())
            # if(probabilities[0][predicted] < 0.5):
            #     # print(probabilities[0][predicted])
            #     # print("error i = ", index)
            index += 1



        ######################### TODO End ##########################
    #將結果寫入semi all的 json
    with open(os.path.join('../hw2_data/p2_data/semi_all/', 'annotations.json'), 'w+') as f:
        json.dump(jsonResult, f)

    test_time = time.time() - test_start_time
    print()
    print(f'Finish testing {test_time:.2f} sec(s), dumps result to {output_path}')

    ##### WRITE RESULT #####
    write_csv(output_path, predictions, test_loader)
    
if __name__ == '__main__':
    main()
