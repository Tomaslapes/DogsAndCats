import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from os import walk
import tqdm

PROCESSDATA = False

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 8, kernel_size = 5)
        self.conv2 = nn.Conv2d(in_channels = 8, out_channels =20, kernel_size=5)

        self.fc1 = nn.Linear(20*13*13,140)
        self.fc2 = nn.Linear(140,280)
        self.out = nn.Linear(280,2)

    def forward(self, t):
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t,kernel_size = 2, stride = 2)

        t = self.conv2(t)
        t = F.relu(t)
        t= F.max_pool2d(t,kernel_size = 2, stride = 2)

        t = t.reshape(-1, 20*13*13)
        t = self.fc1(t)
        t = F.relu(t)

        t = self.fc2(t)
        t = F.relu(t)

        t = self.out(t)
        return(t)


def getImages():
    images = []
    for (dirpath, dirnames, filenames) in walk("C:\Coding\AI\PetImages\Cat"):
        for file in filenames:
            try:
                img = cv2.imread(dirpath + "\\" + file, cv2.IMREAD_GRAYSCALE)
                img = resizeImage(img, 64)
                images.append(img)
                # cv2.imshow("image",img)
                # cv2.waitKey(0)
                #print(img)
            except Exception as e:
                print(e)
    print("******* Images loaded successfully! *******")
    return np.array(images)

def resizeImage(image, dimension):
    img = cv2.resize(image,(dimension,dimension))
    return img

if (PROCESSDATA):
    images = getImages()
    print(images.shape)
    np.save("Data\\Cats_resized",images)
    cv2.imshow("image",images[0])
    cv2.waitKey(0)