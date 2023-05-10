import torch 
import torch.nn as nn 
import torch.nn.functional as F

class MemoNet(nn.Module):
    
    def __init__(self):
        super(MemoNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # input[1, 28, 28]  output[16, 28, 28]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # output[16, 14, 14]
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # output[32, 14, 14]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # output[32, 7, 7]
        )
        self.classifier = nn.Sequential(
            # nn.Dropout(p=0.2),   Dropout不要，因为我就是要让网络过拟合
            nn.Linear(32*7*7, 128, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(128, 48, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(48, 3),
        )
    
    def forward(self, x):
        x = x / 255
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)  # 这里不能把batch维度给展平
        x = self.classifier(x)
        return x
    

if __name__ == "__main__":
    net = MemoNet()
    print(net)

