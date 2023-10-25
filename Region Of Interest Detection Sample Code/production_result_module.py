# By Rohollah Moosavi Tayebi, email: rohollah.moosavi@uwaterloo.ca/moosavi.tayebi@gmail.com

import torch.nn.functional as F
from torchvision import transforms
from model import get_densenet_model
from collections import OrderedDict
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from PIL import Image
from skimage.transform import rescale, resize, downscale_local_mean
import os, shutil
import warnings
warnings.filterwarnings("ignore")
device = "cuda" if torch.cuda.is_available() else "cpu"

class Dataset(Dataset):
    def __init__(self, image_list, images_list_path, transform=None):
        self.image_list = image_list
        self.path = images_list_path
        self.transform = transform

    def __getitem__(self, idx):
        x = self.image_list[idx]
        if self.transform:
            x = self.transform(x)
        else:
            x = torch.tensor(x)
        x = torch.tensor(x).cpu()
        p = self.path[idx]
        return x, p

    def __len__(self):
        return len(self.image_list)

def load_all_images_production_result(input_tile_size, image_list, down_scale):
    dum_path_List = []
    image_list2 = []
    dum_path = "test"

    for x in image_list:
        width, height = x.size
        if width == input_tile_size and height == input_tile_size:
            image_list2.append(x)
            dum_path_List.append(dum_path)

    return image_list2, dum_path_List

def get_data_loader_production_result(images_list, images_list_path, batch_size, shuffle, transform=None):
    dataset = Dataset(images_list, images_list_path, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return data_loader

def real_result(tile_size, image_list, down_scale, model_name, threshold, ROI_path, weight_path):
    rgb_mean = (0.4914, 0.4822, 0.4465)
    rgb_std = (0.2023, 0.1994, 0.2010)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(rgb_mean, rgb_std),
    ])

    print("loading data...")
    test_image_list, dum_image_path = load_all_images_production_result(tile_size , image_list, down_scale)
    batch_size = len(test_image_list) if len(test_image_list) < 32 else 32

    test_loader = get_data_loader_production_result(
        test_image_list,
        dum_image_path,
        batch_size,
        shuffle=False,
        transform=transform_test
    )

    best_model = torch.load(weight_path + '/BestModel_1.pt')
    torch.save(best_model.state_dict(), weight_path + '/BestModel_Dict_1.pt')
    best_model = get_densenet_model(model_name)
    temp_dict = torch.load(weight_path + '/BestModel_Dict_1.pt')
    new_temp_dict = OrderedDict()

    for key, value in temp_dict.items():
        new_key = key[7:]
        new_temp_dict[new_key] = value

    best_model.load_state_dict(new_temp_dict)
    best_model1 = best_model.to(device)

    best_model1.eval()
    with torch.no_grad():
        for i, (x, z) in enumerate(test_loader):
            x = x.to(device)
            output = best_model1(x)
            if threshold == None:
                pred = torch.argmax(output, dim=1)
            else:
                soft = F.softmax(output)
                pred = torch.zeros(output.shape[0])
                for j in range(pred.shape[0]):
                    if soft[j, 1] > threshold:
                        pred[j] = 1
                    else:
                        pred[j] = 0
            if i == 0:
                pred_tot = pred.cpu()
            else:
                pred_tot = torch.cat((pred_tot.cpu(), pred.cpu()), dim=0)

    pos = 0
    neg = 0

    for i in range(pred_tot.shape[0]):
        if pred_tot[i] == 0:
            neg += 1
        else:
            pos += 1

    print(f'Number of not selected tiles: {neg}')
    print(f'Number of (ROI) selected tiles: {pos}')

    num =0
    exportFileName = 'Tile'
    isdir = os.path.isdir(ROI_path)
    if isdir == True:
        print("Deleting current directory")
        shutil.rmtree(ROI_path)
    print(f"Create new directory: {ROI_path}")
    os.mkdir(ROI_path)

    for i in range(len(pred_tot)):
        if pred_tot[i] == 1:
            img = image_list[i]
            cv2.imwrite(ROI_path +'/' + exportFileName + '_' + str(num) + '.png', cv2.cvtColor(np.float32(img), cv2.COLOR_RGB2BGR))
            num += 1
