import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
import torch.nn.functional as F
x = torch.tensor([[1.0, 2.0, 3.0]])
y = torch.tensor([[3.0, 2.0, 1.0]])
label = torch.tensor([[1, 0],[0, 1]])
label = label.repeat(50 ,1)
data = torch.cat((x,y), 0)
data = data.repeat(50, 1)

class MyMLP(nn.Module):
    def __init__(self):
        super(MyMLP, self).__init__()
        self.fc1 = nn.Linear(3, 4)
        self.fc2 = nn.Linear(4, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.sigmoid(x)
        return x
model = MyMLP()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_container = []
t = [50.0 * i for i in range(0, 11)]


for epoch in range(501):
    optimizer.zero_grad()
    outputs = model(data)
    loss = criterion(outputs, torch.argmax(label,dim=1))
    loss.backward()
    optimizer.step()
    if epoch%50 == 0:
        loss_container.append(loss.item())

print(loss, model(data[0]))
plt.plot(t, loss_container)
plt.show()
