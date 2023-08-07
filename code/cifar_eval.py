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

# Function to evaluate the trained Inception model
def eval_inception():
  # Set the device to GPU if available, otherwise to CPU
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
  # Load the model checkpoint
  checkpoint = torch.load(FLAGS["checkpoint_dir"])
  
  # Create the CifarNet model with 10 output classes
  model = CifarNet(num_classes=10)
  
  # Move the model to the chosen device (CPU/GPU)
  model.to(device)
  
  # Load the model's state dictionary from the checkpoint
  model.load_state_dict(checkpoint['model_state_dict'])
  
  # Set the model to evaluation mode (no gradient calculation)
  model.eval()
  
  # Load the test data using the cifar_data_provider
  test_loader = cifar_data_provider.provide_snake_data(train_or_test='test', batch_size=FLAGS['batch_size'])
  
  # Variables to keep track of the evaluation metrics
  total_samples = 0
  correct_predictions = 0
  running_loss = 0.0

  # Define the CrossEntropyLoss criterion for evaluation
  criterion = nn.CrossEntropyLoss()
  
  # Perform the evaluation
  # `with torch.no_grad()` is used to disable gradient calculation during 
  # the evaluation phase. When working with PyTorch, gradient calculation 
  # is essential during the training phase, as it enables backpropagation 
  # and parameter updates to improve the model. However, during the 
  # evaluation or inference phase, we do not need to compute gradients, 
  # as we are only interested in making predictions and evaluating the 
  # model's performance on unseen data.
  with torch.no_grad():
      
    for batch in test_loader:
        state_sequences = batch['state_sequence']
        action_sequences = batch['action_sequence']
    
        outputs = model(state_sequences)
        loss = criterion(outputs, action_sequences)
        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total_samples += action_sequences.size(0)
        correct_predictions += (predicted == action_sequences).sum().item()

        accuracy = correct_predictions / total_samples
        avg_loss = running_loss / len(test_loader)
        print(f"Accuracy on test dataset: {accuracy:.4f}, Avg Loss: {avg_loss:.4f}") # todo, recall, f1score,  interpretation
