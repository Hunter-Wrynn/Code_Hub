import torch

from torch import nn
from torch import optim
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from accelerate import Accelerator


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=16 * 6 * 6, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#Multi_Modal
class UnitModel(nn.Module):
    def __init__(self, model_1, model_2):
        super(UnitModel, self).__init__()
        self.model_1 = model_1
        self.model_2 = model_2

    def forward(self, x, flag):
        if flag == 1:
            x = self.model_1(x)
            return x
        else:
            x = self.model_2(x)
            return x


accelerator = Accelerator()
train_data = MNIST("MNIST", download=True, transform=transforms.Compose({
    transforms.Resize((32, 32)),
    transforms.ToTensor()
}))
train_dataloader = DataLoader(dataset=train_data, batch_size=1024)

#Multi_Model
model = UnitModel(LeNet(), LeNet())
model.train()  
epochs = 10  
lr = 0.01  

criterion = nn.CrossEntropyLoss()
optimizer_1 = optim.SGD(model.model_1.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
optimizer_2 = optim.SGD(model.model_2.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

model, optimizer_1, optimizer_2, train_dataloader \
    = accelerator.prepare(model, optimizer_1, optimizer_2, train_dataloader)

for epoch in range(epochs):
    correct, total = 0, 0
    for index, (inputs, targets) in enumerate(train_dataloader):
        optimizer_1.zero_grad()
        outputs = model(inputs, flag=1) 
        loss_1 = criterion(outputs, targets)  
        accelerator.backward(loss_1) 
        optimizer_1.step()  

        optimizer_2.zero_grad()
        outputs = model(inputs, flag=2)  
        loss_2 = criterion(outputs, targets)  
        accelerator.backward(loss_2) 
        optimizer_2.step()  

        _, predict = outputs.max(1)
        total += targets.size(0)
        correct += predict.eq(targets).sum().item()
    print(
        f"epoch: {epoch + 1} / {epochs}, loss 1: {loss_1:.7f}, loss 2: {loss_2:.7f}, accuracy: {(100 * correct / total):.2f}%")
