import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.optim as optim
import cv2
from os import walk
import tqdm
from model import Model
import matplotlib.pyplot as plt

class dataClass(Dataset):
    def __init__(self,data,labels):
        self.images = data
        self.labels = labels
        self.length = len(self.images)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return (self.images[index],self.labels[index])


train_data = []
train_labels = []
types = {"Cats":1,"Dogs":0}
for type_ in types:
    if type_ == "Dogs":
        images = np.load("Data\\Dogs_resized.npy")
    elif(type_ == "Cats"):
        images = np.load("Data\\Cats_resized.npy")
    for image in images:
        #train_data = np.append(train_data,np.array([image,types[type_]]))
        #print(type(image.tolist()),type(types[type_]))
        image = image.astype("int32")
        #image = torch.from_numpy(image)
        #print(image.shape)
        train_data.append(image)
        train_labels.append(types[type_])

imageData = np.array(train_data)

all_labels = train_labels
# Separate image data into train and test data

TEST_DATA_SAMPLE = 0.15/2

train_data = imageData[int(len(imageData)*TEST_DATA_SAMPLE):int(-len(imageData)*TEST_DATA_SAMPLE)]
train_labels = np.array(train_labels[int(len(imageData)*TEST_DATA_SAMPLE):int(-len(imageData)*TEST_DATA_SAMPLE)])

test_data = imageData[int(-len(imageData)*TEST_DATA_SAMPLE):]
print(test_data.shape)
test_data = np.append(test_data,imageData[:int(len(imageData)*TEST_DATA_SAMPLE)],axis=0)
print(test_data.shape)

test_labels = all_labels[-int(len(imageData)*TEST_DATA_SAMPLE):]
test_labels = np.append(test_labels,all_labels[0:int(len(imageData)*TEST_DATA_SAMPLE)],axis=0)
print("LabelsShape:   ",test_labels.shape)
test_labels = np.array(test_labels)

print("Whole Dataset: ", len(imageData))


print("Train labels:",len(train_labels))
print("Train data",len(train_data))
print("Test labels:",len(test_labels))
print("test data:",len(test_data))

train_data = torch.from_numpy(train_data).unsqueeze(dim=1).type(torch.long)
test_data = torch.from_numpy(test_data).unsqueeze(dim=1).type(torch.long)
train_labels = torch.from_numpy(train_labels).type(torch.long)#.unsqueeze(dim=1).type(torch.long)
test_labels = torch.from_numpy(test_labels).type(torch.long)
print("Data shape:",train_labels.shape,train_data.shape)

train_data = train_data.cuda()
train_labels = train_labels.cuda()

test_data = test_data.cuda()
test_labels = test_labels.cuda()

trainData = dataClass(train_data,train_labels)
testData = dataClass(test_data,test_labels)
#print(trainData[15000])

print("Test shape",test_data.shape,test_labels.shape)

dataLoader = DataLoader(dataset=trainData,batch_size=100,shuffle=True)
testDataLoader = DataLoader(dataset= testData,batch_size=100,shuffle=True)

#DogCount = 0
#CatCount = 0
#total = 0
#for image,label in dataLoader:
    #print(image)
    #print("Image: ", image, "Label: ",label)
    #if label == 1:
    #    CatCount += 1
    #if label == 0:
    #    DogCount += 1
    #total += 1
    #img_ = image.numpy()
    #print(img_[0].shape)
    #for img in img_:
     #   cv2.imshow("img",img.astype(np.uint8))
      #  cv2.waitKey(0)

#print("Dogs: ",DogCount,"Cats: ",CatCount,"Total: ",total)


network = Model()
network = network.cuda()

EPOCHS = 60
images = None
optimizer = optim.Adam(network.parameters(),lr = 0.00055)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience= 3, verbose=True)


lossList = []
bestRun = None

for epoch in range(EPOCHS):
    print(f"EPOCH: {epoch+1}/{EPOCHS} \n")
    lossPerEpoch = None
    avgLoss = None
    for images,labels in dataLoader:
        #test_img = images[0].squeeze(dim=0).detach().cpu()
        #plt.imshow(test_img,cmap="gray")
        #print(labels[0])
        #plt.show()
        #print(images.dtype,labels.dtype)
        images = images.type(torch.float)
        #print("\nLables:",labels)
        #print(images.dtype,"Labels Shape", labels.shape)
        preds = network(images.float())
        loss = F.cross_entropy(preds,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if avgLoss == None:
            avgLoss = loss
        if bestRun is None:
            bestRun = network.state_dict()
        if avgLoss > loss:
            bestRun = network.state_dict()
        avgLoss = (avgLoss+loss)/2
        lossPerEpoch = loss
    print(avgLoss)
    totalAvgLoss = avgLoss.mean()
    lossList.append(avgLoss)
    scheduler.step(totalAvgLoss)
    print(f"Loss after epoch: {epoch+1}",lossPerEpoch,f"Average loss: {avgLoss}")

plt.plot(lossList)
plt.yscale("log")
plt.show()

network.load_state_dict(bestRun)

network.eval()

correctPredicitons = 0

for img,label in testDataLoader:
    img = img.type(torch.float)
    preds = network(img.float())
    preds = torch.argmax(preds,dim = 1)
    #print("Predictions: ",preds,"Labels: ",label)
    for i,pred in enumerate(preds):
        if label[i] == pred:
            correctPredicitons +=1

print(f"Correct predicions: {correctPredicitons} out of {len(test_labels)} -> {correctPredicitons/len(test_labels)*100}%")