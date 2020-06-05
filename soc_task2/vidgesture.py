# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'
import torch
import torchvision
from torch import nn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np
import copy
import cv2
from PIL import Image
from matplotlib import pyplot as plt

img_rows, img_cols = 200, 200
#number of output classes
nb_classes = 5
# Number of epochs to train 
nb_epoch = 15  #25
# Total number of convolutional filters to use
nb_filters = 32
# Max pooling
nb_pool = 2
# Size of convolution kernel
nb_conv = 3
#batch size in training the model
batch_size = 50

img_dir = r'C:/Users/Meeta Malviya/Videos/data2/train'
test_dir = r'C:/Users/Meeta Malviya/Videos/data2/test'

train_transforms = transforms.Compose([transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
                                      transforms.RandomRotation(degrees=10),
                                      transforms.ColorJitter(),
                                      transforms.Grayscale(num_output_channels=3),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.CenterCrop(size=224),  # Image net standards
                                      transforms.ToTensor()
                                      ])

test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.Grayscale(num_output_channels=3),
                                      transforms.ToTensor()
                                     ])

validation_transforms = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.Grayscale(num_output_channels=3),
                                            transforms.ToTensor()
                                           ])

#Loading in the dataset

train_data = datasets.ImageFolder(img_dir, transform=train_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

# obtain training indices that will be used for validation
valid_size = 0.2

num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

# define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# load training data in batches
train_loader = torch.utils.data.DataLoader(train_data,
                                           batch_size=batch_size,
                                           sampler=train_sampler,
                                           num_workers=0)

# load validation data in batches
valid_loader = torch.utils.data.DataLoader(train_data,
                                           batch_size=batch_size,
                                           sampler=valid_sampler,
                                           num_workers=0)

# load test data in batches
test_loader = torch.utils.data.DataLoader(test_data,
                                          batch_size=batch_size,
                                          num_workers=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#pretrained=True will download a pretrained network for us
model_t = models.resnet50(pretrained=True)

#Freezing model parameters and defining the fully connected network to be attached to the model, loss function and the optimizer.
#We there after put the model on the GPUs
for param in model_t.parameters():
    param.require_grad = False 
model_t.fc = nn.Sequential(
    nn.Linear(2048, 5),
    nn.LogSoftmax(dim=1)    
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adagrad(model_t.parameters(), lr=0.001, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10)
model_t.to(device)


PATH=r'C:/Users/Meeta Malviya/Videos/Vgg/restnet_weights'

model_t.load_state_dict(torch.load(PATH))
model_t.eval()

def image_process(image):
    predict_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.Grayscale(num_output_channels=3),
                                      transforms.ToTensor()
                                            ])
    image = predict_transforms(image)
    return image 

def predict(img, model_t):
    
    img = Image.fromarray(img)
    img = image_process(img)

    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img)
    
    model_t.eval()
    target = model_t(img)
    
    out = F.softmax(target,dim=1)
    topk = out.cpu().topk(5)
    
    return (e.data.numpy().squeeze().tolist() for e in topk)

def plotting(prob, class_names):
    
    fig,ax = plt.subplots()
    y_pos = np.arange(len(prob))
    plt.bar(y_pos,prob)
    ax.set_xticks(y_pos)
    ax.set_xticklabels(class_names)
    plt.xticks(rotation=45)
    plt.grid(True, which='both')
    plt.show()
    
    
cap = cv2.VideoCapture(0)
cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
frame = 0
while(cap.isOpened()):
    ret, img = cap.read()
    cv2.rectangle(img,(300,300),(100,100),(0,255,0),0)
    cv2.imshow('frame',img)
    crop_img = img[100:300, 100:300]
    cv2.imshow('frame2',crop_img)
    
    while( frame % 60 == 30 ):
        plt.close('all')
        prob, classes = predict(crop_img, model_t.to(device))
        class_names = [test_loader.dataset.classes[e] for e in classes]
        
        print(prob)
        print(classes)
        print(class_names)

        plotting(prob,class_names)
        print(frame)
        frame = frame+1
    
    if frame == 45:
        plt.close('all')
        
    k = cv2.waitKey(100)
    frame = frame+1
    if k & 0xFF == ord('q'):
            break  