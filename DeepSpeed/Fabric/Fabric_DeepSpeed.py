import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import Accuracy
from fabric import Fabric
import deepspeed


class SimpleNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleNet, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    input_dim = 10
    hidden_dim = 20
    output_dim = 2
    batch_size = 64
    data_size = 10000

    input_data = torch.randn(data_size, input_dim)
    labels = torch.randn(data_size, output_dim)

    dataset = TensorDataset(input_data, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SimpleNet(input_dim, hidden_dim, output_dim)

    optimizer = optim.Adam(model.parameters(), lr=0.00015)
    criterion = torch.nn.MSELoss()


    fabric = Fabric(accelerator="auto", precision=16)
    fabric.setup(model=model, optimizer=optimizer)


    deepspeed.init_distributed()


    for epoch in range(1000):
        model.train()
        for batch in dataloader:
            inputs, labels = batch
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            fabric.backward(loss)
            optimizer.step()

        print(f"Epoch {epoch} loss: {loss.item()}")

 
    fabric.save(model.state_dict(), "model.pth")
