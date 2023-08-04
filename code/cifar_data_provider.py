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

"""Contains code for loading and preprocessing the CIFAR data."""

import cifar10_dataset
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torchvision

def provide_cifarnet_data(train_or_test, batch_size):
  """Provides batches of CIFAR images for cifarnet.""" 

  ## transforms.RandomCrop(32, padding = 4) performs a random crop of size 32x32 pixels on imput and the image is first zero padded with a 4 pixel border 
  ## transforms.RandomHorizontalFlip() - randomly flips input image horizontally with probability of 0.5
  ## transforms.Resize((32,32)) - resizes input image to fixed size of 32x32 pizels and its used for test data to ensure that all test images have the same size as training images
  ## transforms.ToTensor() - converts input image to a pytorch tensor 
  ## transforms.Normalize(mean = [0.485...]) normalizes tensor by subracts mean and dividing by the standard deviation 
    
  image_size = 32
  
  # Define the transformations to be applied to the images based on train or test mode
  transform = transforms.Compose([transforms.RandomCrop(image_size, padding = 4), 
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(), 
                                  transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                                                      std = [0.229,0.224,0.225]),])


  # Load the training dataset
  if train_or_test == 'train':
    transform = transforms.Compose([
      transforms.RandomCrop(image_size, padding = 4), 
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(), 
      transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                           std = [0.229,0.224,0.225]),])
    train_dataset = torchvision.datasets.CIFAR10(root=str('./data') , train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader
  else: 
    # Load the testing dataset
    transform = transforms.Compose([
      transforms.Resize((image_size, image_size)),
      transforms.ToTensor(), 
      transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                           std = [0.229,0.224, 0.225]),])
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader