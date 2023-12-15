import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import pyplot as plt
x = torch.tensor([[[[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 2, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 2, 4, 2, 1, 0, 0, 0, 0, 0],
                    [0, 1, 2, 4, 8, 4, 2, 1, 0, 0, 0, 0],
                    [1, 2, 4, 8, 16,8, 4, 2, 1, 2, 0, 0],
                    [0, 1, 2, 4, 8, 4, 4, 4, 4, 4, 1, 0],
                    [0, 0, 1, 4, 4, 4, 4, 4, 2, 2, 1, 0],
                    [0, 0, 0, 1, 4, 4, 2, 2, 1, 1, 0, 0],
                    [0, 0, 0, 0, 1, 4, 2, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]]],dtype = torch.float)
y = torch.tensor([[[[0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 4, 4, 2, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 4, 8, 4, 4, 2, 1, 0, 0, 0, 0],
                    [0, 2, 8, 8, 8, 8, 2, 1, 0, 0, 0, 0],
                    [1, 2, 4, 8, 16,8, 4, 2, 1, 2, 0, 0],
                    [0, 1, 2, 8, 8, 4, 4, 4, 4, 4, 1, 0],
                    [0, 0, 2, 4, 4, 4, 4, 4, 2, 2, 1, 0],
                    [0, 0, 0, 1, 4, 4, 2, 2, 1, 1, 0, 0],
                    [0, 0, 0, 0, 1, 4, 2, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]]],dtype = torch.float)

data = torch.empty(0,1,12,12)
data = torch.cat((data,x),dim=0)
data = data.repeat(50,1,1,1)
label = torch.empty(0,1,12,12)
label = torch.cat((label,y),dim=0)
label = label.repeat(50,1,1,1)
def paint_tensor(x):
    x = x.detach().numpy()
    for _ in range(2):
        x = x.squeeze(0)
    cmap = colors.LinearSegmentedColormap.from_list(
        'custom_cmap', ['blue', 'white', 'red']
    )
    norm = colors.Normalize(vmin=0, vmax=16)
    plt.imshow(x, cmap=cmap, norm=norm)
    plt.colorbar()
    plt.show()

#paint_tensor(x)

class DoubleConv(nn.Module):  
    def __init__(self,in_channels,out_channels):
        super(DoubleConv,self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(out_channels),                                               
            nn.ReLU(inplace=True),                                                       
            nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self,x): 
        x = self.double_conv(x)
        return x
class Down(nn.Module):  
    def __init__(self,in_channels,out_channels):
        super(Down, self).__init__()
        self.downsampling = nn.Sequential(
            nn.MaxPool2d(kernel_size=2,stride=2),
            DoubleConv(in_channels,out_channels)
        )
    def forward(self,x):
        x = self.downsampling(x)
        return x
class Up(nn.Module): 
    def __init__(self, in_channels, out_channels):
        super(Up,self).__init__()
        self.upsampling = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.upsampling(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x
class OutConv(nn.Module): 
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)
    def forward(self, x):
        return self.conv(x)
class Encoder(nn.Module):
    def __init__(self, in_channels = 1, num_classes = 1):
        super(Encoder, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.in_conv = DoubleConv(in_channels, 8)
        self.down1 = Down(8, 16)
        self.down2 = Down(16, 32)

    def forward(self, x):
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        return x1, x2, x3

class Decoder(nn.Module):
    def __init__(self, in_channels = 1, num_classes = 1):
        super(Decoder, self).__init__()
        self.in_channels = in_channels                  
        self.num_classes = num_classes                       
        self.up1 = Up(32, 16)                       
        self.up2 = Up(16, 8) 
        self.out_conv = OutConv(8, num_classes) 
    def forward(self, x, x1, x2, x3):
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        x = self.out_conv(x)
        return x

class UNet(nn.Module): 
    def __init__(self, in_channels = 1, num_classes = 1):
        super(UNet, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    def forward(self, x):
        x1, x2, x3 = self.encoder(x)
        x = self.decoder(x, x1, x2, x3)
        return x

model = UNet(in_channels=1,num_classes=1)
print(model(data[0].unsqueeze(0)).shape)
print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in model.parameters())))    
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr = 0.001)
loss_container = []
t = [10.0 * i for i in range(0, 51)]
for epoch in range(501):
    optimizer.zero_grad()
    outputs = model(data)
    loss = criterion(outputs, label)
    loss.backward()
    optimizer.step()
    if epoch%10 == 0:
        loss_container.append(loss.item())

print(loss)
#paint_tensor(model(data[0].unsqueeze(0)))

#paint_tensor(label[0].unsqueeze(0))
plt.plot(t, loss_container)
plt.show()