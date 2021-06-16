import cv2
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from unrealcv import client
from unrealcv.util import read_npy, read_png

from io import BytesIO
import PIL.Image

import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
import torchsummary as summary

# Load in pretrained mobilenet V2 network and reduce to feature extractor
mobilenet = models.mobilenet_v2(pretrained=True)
mobilenet.eval()
feature_network = nn.Sequential(*(list(mobilenet.children())[0]))

# Preprocess data before inference --> not sure why
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

print("######################### connecting client ##########################")
client.connect()
if not client.isconnected():
    print('UnrealCV server is not running. Run the game from http://unrealcv.github.io first.')
else:
    print('UnrealCV server is running.')

time.sleep(5)

print("######################### readin png to RGB ##########################")
res = client.request('vget /camera/0/lit png')
img_decode =  PIL.Image.open(BytesIO(res)).convert('RGB')
print(type(img_decode))


print("######################### convert to tensor ##########################")
img_tensor = preprocess(img_decode)
print(img_tensor.shape)
img_tensor = img_tensor.unsqueeze(0)
print(img_tensor.shape)

print("########################## inferin features ##########################")
features = feature_network(img_tensor)
print(features)
print(features.shape)
print(features.detach().numpy())
print(features.detach().numpy().shape)
print(features.detach().numpy().flatten().shape)
