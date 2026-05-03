import torch
import torch.nn as nn
import torchvision

g_resnet18 = torchvision.models.resnet18()
g_resnet34 = torchvision.models.resnet34()   
g_resnet50 = torchvision.models.resnet50()   

class FangCNN(nn.Module):
    def __init__(self):
        super(FangCNN, self).__init__()

        self.net = nn.Sequential(nn.Conv2d(3, 30, 5),
                                nn.ReLU(),
                                nn.MaxPool2d(2, stride=2),
                                nn.Conv2d(30, 50, 5),
                                nn.ReLU(),
                                nn.MaxPool2d(2, stride=2),
                                nn.Flatten(),
                                nn.Linear(1250, 512),
                                nn.ReLU(),
                                nn.Linear(512, 10)
        )

    def init_ones(self,m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def init_zeros(self,m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)


    def init_xavier(self,m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight, gain=2.24)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.net(x)
        return x
    
class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()

        self.net = nn.Sequential(*list(g_resnet18.children())[:-1], # [b, 512, 1, 1]
                                nn.Flatten(), # [b, 512, 1, 1] => [b, 512]
                                nn.Linear(512, 10)
                                )

    def forward(self, x):
        x = self.net(x)

        return x
    
class ResNet34(nn.Module):
    def __init__(self):
        super(ResNet34, self).__init__()

        self.net = nn.Sequential(*list(g_resnet34.children())[:-1], # [b, 512, 1, 1]
                                nn.Flatten(), # [b, 512, 1, 1] => [b, 512]
                                nn.Linear(512, 10)
                                )

    def forward(self, x):
        x = self.net(x)
        return x

class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()

        self.net = nn.Sequential(*list(g_resnet50.children())[:-1], # [b, 512, 1, 1]
                                nn.Flatten(), # [b, 512, 1, 1] => [b, 512]
                                nn.Linear(2048, 10)
                                )

    def forward(self, x):
        x = self.net(x)
        return x