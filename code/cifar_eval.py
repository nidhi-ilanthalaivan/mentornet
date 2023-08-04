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

"""Evaluates a trained model.

See the README.md file for compilation and running instructions.
"""

import torch.nn as nn
import cifar_data_provider
from inception_model import CifarNet
from tqdm import tqdm
import torch
from cifar_train_baseline import FLAGS 

def eval_inception():
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  checkpoint = torch.load(FLAGS["checkpoint_dir"])
  model = CifarNet(num_classes=10)
  model.to(device)
  model.load_state_dict(checkpoint['model_state_dict'])
  model.eval()
  test_loader = cifar_data_provider.provide_cifarnet_data(train_or_test='test', batch_size=FLAGS['batch_size'])
  total_samples = 0
  correct_predictions = 0
  running_loss = 0.0

  criterion = nn.CrossEntropyLoss()
  with torch.no_grad():
      for inputs, labels in tqdm(test_loader, desc="Evaluating"):
          inputs, labels = inputs.to(device), labels.to(device)

          outputs = model(inputs)
          loss = criterion(outputs, labels)
          running_loss += loss.item()

          _, predicted = torch.max(outputs.data, 1)
          total_samples += labels.size(0)
          correct_predictions += (predicted == labels).sum().item()

      accuracy = correct_predictions / total_samples
      avg_loss = running_loss / len(test_loader)
      print(f"Accuracy on test dataset: {accuracy:.4f}, Avg Loss: {avg_loss:.4f}") # todo, recall, f1score,  interpretation
