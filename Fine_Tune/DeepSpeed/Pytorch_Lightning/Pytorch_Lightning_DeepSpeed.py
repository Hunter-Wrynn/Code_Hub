import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.plugins import DeepSpeedPlugin


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


class LitModel(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.model = SimpleNet(input_dim, hidden_dim, output_dim)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=0.00015)
        return optimizer


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

    model = LitModel(input_dim, hidden_dim, output_dim)

    
    deepspeed_plugin = DeepSpeedPlugin(stage=2, gradient_clip_val=1.0)

    trainer = Trainer(
        max_epochs=1000,
        gpus=1,  
        precision=16, 
        plugins=[deepspeed_plugin],
    )

  
    trainer.fit(model, dataloader)

  
    trainer.save_checkpoint("model.ckpt")
