from diloco_sim import *
from diloco_sim.diloco import *
from models import ModelArchitecture
import torch.nn.functional as F
from data import train_dataset, test_dataset 
simulator = DilocoSimulator(
    model_cls=CNNModel,
    model_kwargs={
      "num_classes": 100,
      "input_channels": 3,
      "input_height": 32,
      "input_width": 32
    },
    optimizer_kwargs={"lr": 0.001},
    num_nodes=2,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    loss_fn=F.cross_entropy,
    num_epochs=10
  )

simulator.train()