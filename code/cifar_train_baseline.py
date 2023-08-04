# Copyright 2018 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Trains baseline models.

See the README.md file for compilation and running instructions.
"""

import os
import time
import cifar_data_provider
from inception_model import CifarNet
import torch 
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms 
from torch.utils.tensorboard import SummaryWriter 
from tqdm import tqdm
import torchvision.models as models
from datetime import datetime

FLAGS = {
  'batch_size': 64, # The number of images in each batch.
  'train_log_dir': r'./train_logs/train',
  'studentnet': 'inception',
  'learning_rate': 0.05,
  'learning_rate_decay_factor': 0.1,
  'num_epochs': 1,
  'num_epochs_per_decay': 10, #'Number of epochs after which learning rate decays.
  'save_summaries_secs': 120, # The frequency with which summaries are saved, in seconds.
  'save_interval_secs': 1200, # The frequency with which the model is saved, in seconds.
  'dropout': 0.2,
  'checkpoint_dir': "",
}

def train_inception_baseline():
  """Trains the inception baseline model."""
  
  current_datetime = datetime.now()

  # Format the date and time into the desired string format
  timestamp_when_start = current_datetime.strftime("%Y%m%d%H%M")

  checkpoint_dir = f"checkpoints/{timestamp_when_start}"
  os.makedirs(checkpoint_dir, exist_ok=True)
  FLAGS['checkpoint_dir'] = checkpoint_dir
  if not os.path.exists(FLAGS['train_log_dir']):
    os.makedirs(FLAGS['train_log_dir'])
  

  train_loader = cifar_data_provider.provide_cifarnet_data(train_or_test='train', batch_size=FLAGS['batch_size'])
  
  model = CifarNet(num_classes=10, dropout_keep_prob=FLAGS['dropout']) 
  
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(model.parameters(), lr=FLAGS["learning_rate"], momentum=0.9)

  # Set the device to CPU/GPU
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model.to(device)

  #TB writer 
  writer = SummaryWriter(log_dir = FLAGS["train_log_dir"])

  for epoch in tqdm(range(FLAGS['num_epochs'])):
    train_loss = 0.0
    for i, data in tqdm(enumerate(train_loader, 0)):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        outputs = model(inputs)          
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if i % 200 == 199:  # 
          writer.add_scalar(f'[Epoch {epoch + 1}, Batch {i + 1}], Loss/train', train_loss/200)
          train_loss = 0.0
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch{epoch + 1}.pth")
    torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': train_loss,
    }, checkpoint_path)

  FLAGS['checkpoint_dir'] = f"checkpoints/{timestamp_when_start}/checkpoint_epoch{FLAGS['num_epochs']}.pth"


# The training loss and test accuracy will be logged to the TensorBoard. 
# To launch TensorBoard and visualize the training progress, run the following command in the terminal:

# `tensorboard --logdir=runs``
# You can then access TensorBoard in your web browser at the given URL (e.g., http://localhost:6006/).