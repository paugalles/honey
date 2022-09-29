import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F

from MobileNetV2 import MobileNetV2

def build_model(
    last_layer_num_neurons,
    name = 'efficientnet_b0',
    num_classes=10
):
    
    last_layer_num_neurons = [int(l) for l in last_layer_num_neurons.split(',')]
    
    if name == 'MobileNetV2':
        
        model = MobileNetV2(n_classes=num_classes)
    
    elif name == 'efficientnet_b0':
        
        model = EfficientV1B0(
            last_layer_num_neurons,
            num_classes,
            pretrained=True,
            n_channels=3 ,
        )
        
    elif name == 'resnet18':
        
        model = ResNet(
            last_layer_num_neurons,
            num_classes,
            pretrained=True,
            n_channels=3
        )

    elif name == 'CNN1':
        
        model = CNN1(
            num_classes
        )

    elif name == 'CNN2':
        
        model = CNN2(
            num_classes
        )
        
    elif name == 'CNN3':
        
        model = CNN3(
            num_classes
        )
        
    return model


class EfficientV1B0(nn.Module):
    
    def __init__(
        self,
        last_layer_num_neurons,
        num_classes,
        pretrained=True,
        n_channels=3 ,
    ):
        """
        resnet18 architecture is designed with 3channels input
        """
        super(EfficientV1B0, self).__init__()
        eff = models.efficientnet_b0(pretrained=True)
        modules=list(eff.children())[:-1]
        
        self.basenet = nn.Sequential(*modules)
            
        fc_modules = [
            nn.Linear(
                last_layer_num_neurons[enu],
                last_layer_num_neurons[enu+1]
            )
            for enu in range(len(last_layer_num_neurons)-1)
        ] + [
            nn.Linear(
                last_layer_num_neurons[-1],
                num_classes
            )
        ]

        self.fc = nn.Sequential(*fc_modules)
    
    def forward(self, x):
        for i in range(len(self.basenet)):
            x = self.basenet[i](x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        return x


class ResNet(nn.Module):
    
    def __init__(
        self,
        last_layer_num_neurons,
        num_classes,
        pretrained=True,
        n_channels=3 ,
    ):
        """
        resnet18 architecture is designed with 3channels input
        """
        super(ResNet, self).__init__()
        resnet18 = models.resnet18(pretrained=pretrained)
        modules=list(resnet18.children())[:-1]
        
        if n_channels!=3:
            modules[0] = nn.Conv2d(n_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        self.basenet = nn.Sequential(*modules)
        
        fc_modules = [
            nn.Linear(
                last_layer_num_neurons[enu],
                last_layer_num_neurons[enu+1]
            )
            for enu in range(len(last_layer_num_neurons)-1)
        ] + [
            nn.Linear(
                last_layer_num_neurons[-1],
                num_classes
            )
        ]

        self.fc = nn.Sequential(*fc_modules)
    
    def forward(self, x):
        for i in range(len(self.basenet)):
            x = self.basenet[i](x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        return x

    
class CNN1(nn.Module):
    def __init__(
        self,
        num_classes
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 8, 5)
        self.conv3 = nn.Conv2d(8, 8, 5)
        self.conv4 = nn.Conv2d(8, 12, 5)
        self.fc1   = nn.Linear(12*3*3, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        #x = self.pool(F.relu(self.conv5(x)))
        #x = self.pool(F.relu(self.conv6(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        return x

class CNN2(nn.Module):
    def __init__(
        self,
        num_classes
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.conv3 = nn.Conv2d(32, 64, 5)
        self.conv4 = nn.Conv2d(64, 64, 5)
        self.fc1   = nn.Linear(64*3*3, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        #x = self.pool(F.relu(self.conv5(x)))
        #x = self.pool(F.relu(self.conv6(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        return x

# class CNN3(nn.Module):
#     def __init__(
#         self,
#         num_classes
#     ):
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 32)
#         self.fc3 = nn.Linear(32, num_classes)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = torch.flatten(x, 1) # flatten all dimensions except batch
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x