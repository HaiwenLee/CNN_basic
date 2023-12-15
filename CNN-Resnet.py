import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from matplotlib import pyplot as plt
x = torch.tensor([[[[0,1,2,1,0.0],
                    [1,2,4,2,1.0],
                    [2,4,8,4,2.0],
                    [1,2,4,2,1.0],
                    [0,1,2,1,0.0]]]])
y = torch.tensor([[[[8,4,2,4,8.0],
                    [4,2,1,2,4.0],
                    [2,1,0,1,2.0],
                    [4,2,1,2,4.0],
                    [8,4,2,4,8.0]]]])
data = torch.empty(0,1,5,5)
data = torch.cat((data,x),dim=0)
data = torch.cat((data,y),dim=0)
data = data.repeat(50,1,1,1)
label = torch.tensor([[1,0],[0,1]])
label = label.repeat(50,1)
class CNN(nn.Module):
    def __init__(self, input_channels=1,num_channels=4,
                 use_1x1conv=False, strides=1):
        super(CNN, self).__init__()
        """
        ###Resnet_block
        """
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        
        self.conv4 = nn.Conv2d(4,8,kernel_size=(3,3),padding = 1)
        self.pool1 = nn.AvgPool2d(kernel_size=(2,2))
        self.linear1 = nn.Linear(8 * 2 * 2, 8)
        self.linear2 = nn.Linear(8, 2)
        

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        x = F.relu(Y)
        x = self.pool1(self.conv4(x))
       
        x = F.relu(x)
        x = x.view(-1, 8 * 2 * 2)
        x = self.linear1(x)
        x = self.linear2(x)
        x = F.sigmoid(x)
 
        return x
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr = 0.01)
loss_container = []
t = [10.0 * i for i in range(0, 11)]
print(model(data[0].unsqueeze(0)).shape)

for epoch in range(101):
    optimizer.zero_grad()
    outputs = model(data)
    loss = criterion(outputs, torch.argmax(label,dim=1))
    loss.backward()
    optimizer.step()
    if epoch%10 == 0:
        loss_container.append(loss.item())
print(loss.item())
print(model(data[0].unsqueeze(0)))
plt.plot(t, loss_container)
plt.show()

