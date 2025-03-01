# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Provides data for the Cifar10 dataset."""

#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function

import os

import torch 

import torchvision 

import torchvision.transforms as transforms 

import torchvision.datasets as datasets 

_FILE_PATTERN = 'cifar10_%s-*'

_DATASET_DIR = ('')

_SPLITS_TO_SIZES = {'train': 50000, 'test': 10000}

_NUM_CLASSES = 10

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A [32 x 32 x 3] color image.',
    'label': 'A single integer between 0 and 9',
}

class CIFAR10Dataset(torch.utils.data.Dataset): 
  def __init__(self, split_name, dataset_dir = None, transform = None ): 
    if split_name is None: 
      raise ValueError('split name $s was not recognized.' %split_name)
    
    if dataset_dir is None: 
      dataset_dir = _DATASET_DIR
    
    self.split_name = split_name 
    self.dataset_dir = dataset_dir 
    self.transform = transform 
    
    self.data, self.targets = self._load_data()
    
  # used chatGPT (start)- 
  def _load_data(self): 
     dataset = torchvision.datasets.CIFAR10(
       root = self.dataset_dir,
       train = self.split_name =='train', 
       download = True, 
       transform = self. transform
     )
     return dataset.data, torch.tensor(dataset.targets)
   # end
   
  def __len__(self): 
    return len(self.data)
  def __getitem__(self, index): 
    img, target = self.data[index], self.targets[index]
    
    if self.transform: 
      img = self.transform(img)
      return img, target 
    
  def get_split(split_name, dataset_dir = None): 
    if split_name not in _SPLITS_TO_SIZES: 
      raise ValueError('split name %s was not recognized.' %split_name)
    return CIFAR10Dataset(split_name, dataset_dir)
  """Gets a dataset tuple with instructions for reading cifar10.

  Args:
    split_name: A train/test split name.
    dataset_dir: The base directory of the dataset sources.

  Returns:
    A `Dataset` namedtuple. Image tensors are integers in [0, 255].

  Raises:
    ValueError: if `split_name` is not a valid train/test split.
  """
  
