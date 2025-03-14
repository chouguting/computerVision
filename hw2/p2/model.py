import torch
import torch.nn as nn
import torchvision.models as models

class MyNet(nn.Module): 
    def __init__(self):
        super(MyNet, self).__init__()
        
        ################################################################
        # TODO:                                                        #
        # Define your CNN model architecture. Note that the first      #
        # input channel is 3, and the output dimension is 10 (class).  #
        ################################################################
        self.ConvLayer0 = nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1)
        self.ConvLayer1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.ConvLayer2 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.ConvLayer3 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.batchNorm = nn.BatchNorm2d(1024)
        self.flatten = nn.Flatten()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.fc0 = nn.Linear(4096, 1024)
        self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):

        ##########################################
        # TODO:                                  #
        # Define the forward path of your model. #
        ##########################################
        #輸入為(3,32,32)
        x = self.ConvLayer0(x) #輸出為(128,32,32)
        x = self.relu(x)  #輸出為(128,32,32)
        x = self.maxPool(x) #輸出為(128,16,16)
        x = self.ConvLayer1(x) #輸出為(256,16,16)
        x = self.relu(x)    #輸出為(256,16,16)
        x = self.maxPool(x) #輸出為(256,8,8)
        x = self.ConvLayer2(x) #輸出為(512,8,8)
        x = self.relu(x)    #輸出為(512,8,8)
        x = self.maxPool(x) #輸出為(512,4,4)
        x = self.ConvLayer3(x) #輸出為(1024,4,4)
        x = self.relu(x)    #輸出為(1024,4,4)
        x = self.maxPool(x) #輸出為(1024,2,2)

        x = self.batchNorm(x)
        x = self.flatten(x) #輸出為(4096)
        x = self.dropout1(x)
        x = self.fc0(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    
class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()

        ############################################
        # NOTE:                                    #
        # Pretrain weights on ResNet18 is allowed. #
        ############################################

        # (batch_size, 3, 32, 32)
        self.resnet = models.resnet18(pretrained=True)
        # (batch_size, 512)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 10)
        self.dropout = nn.Dropout(0.5)
        self.adaptive_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        # (batch_size, 10)

        #######################################################################
        # TODO (optinal):                                                     #
        # Some ideas to improve accuracy if you can't pass the strong         #
        # baseline:                                                           #
        #   1. reduce the kernel size, stride of the first convolution layer. # 
        #   2. remove the first maxpool layer (i.e. replace with Identity())  #
        # You can run model.py for resnet18's detail structure                #
        #######################################################################
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.resnet.maxpool = Identity()
        # add dropout
        self.resnet.avgpool = nn.Sequential(
            self.adaptive_pool,
            self.dropout
        )

    def forward(self, x):
        return self.resnet(x)

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
    
if __name__ == '__main__':
    model = MyNet()
    print(model)
