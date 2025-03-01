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

"""Wide ResNet-101 model.

Wider filters are used [16, 160, 320, 640] instead of [16, 16, 32, 64]
More details are in
https://arxiv.org/pdf/1605.07146v1.pdf

Related papers:
https://arxiv.org/pdf/1603.05027v2.pdf
https://arxiv.org/pdf/1512.03385v1.pdf
https://arxiv.org/pdf/1605.07146v1.pdf
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
import torch.optim as optim
import numpy as np



HParams = namedtuple('HParams',
                     'batch_size, num_classes, min_lrn_rate, lrn_rate, '
                     'num_residual_units, use_bottleneck, weight_decay_rate, '
                     'relu_leakiness, optimizer')


class ResNet(nn.Module):
  """ResNet model."""

  def __init__(self, hps, images, labels, mode):
    """ResNet constructor.

    Args:
      hps: Hyperparameters.
      images: Batches of images. [batch_size, image_size, image_size, 3]
      labels: Batches of labels. [batch_size, num_classes]
      mode: One of 'train' and 'eval'.
    """
    super(ResNet, self).__init__()
    self.hps = hps
    self._images = images
    self.labels = labels
    self.mode = mode

    self.extra_train_ops = []

  def build_graph_unused(self):
    """Build a whole graph for the model."""
    self.global_step = torch.nn.Parameter(torch.tensor(0), requires_grad = False)
    self.build_model()
    if self.mode == 'train':
      self._build_train_op()

  def _stride_arr(self, stride):
    """Map a stride scalar to the stride array for tf.nn.conv2d."""
    return [1, stride, stride, 1]

  def build_model(self):
    """Build the core model within the graph."""
    with torch.no_grad():
      x = self._images
      x = self._conv('init_conv', x, 3, 3, 16, self._stride_arr(1))

    strides = [1, 2, 2]
    activate_before_residual = [True, False, False]
    if self.hps.use_bottleneck:
      res_func = self._bottleneck_residual
      filters = [16, 64, 128, 256]
    else:
      res_func = self._residual
      # filters = [16, 16, 32, 64]
      # Uncomment the following codes to use w28-10 wide residual network.
      # It is more memory efficient than very deep residual network and has
      # comparably good performance.
      # https://arxiv.org/pdf/1605.07146v1.pdf
      filters = [16, 160, 320, 640]
      # Update hps.num_residual_units to 9

    with torch.no_grad():
      x = res_func(x, filters[0], filters[1], self._stride_arr(strides[0]),
                   activate_before_residual[0])
    for i in range(1, self.hps.num_residual_units):
      with torch.no_grad():
        x = res_func(x, filters[1], filters[1], self._stride_arr(1), False)

    with torch.no_grad():
      x = res_func(x, filters[1], filters[2], self._stride_arr(strides[1]),
                   activate_before_residual[1])
    for i in range(1, self.hps.num_residual_units):
      with torch.no_grad():
        x = res_func(x, filters[2], filters[2], self._stride_arr(1), False)

    with torch.no_grad():
      x = res_func(x, filters[2], filters[3], self._stride_arr(strides[2]),
                   activate_before_residual[2])
    for i in range(1, self.hps.num_residual_units):
      with torch.no_grad():
        x = res_func(x, filters[3], filters[3], self._stride_arr(1), False)

    with torch.no_grad():
      x = self._batch_norm('final_bn', x)
      x = self._relu(x, self.hps.relu_leakiness)
      x = self._global_avg_pool(x)

    with torch.no_grad():
      logits = self._fully_connected(x, self.hps.num_classes)

    return logits

  def _build_train_op(self):
    """Build training specific ops for the graph."""
    self.lrn_rate = torch.tensor(self.hps.lrn_rate, dtype = torch.float32)


    trainable_parameters = filter(lambda p: p.requires_grad, self.parameters())
    optimizer = None

    if self.hps.optimizer == 'sgd':
      optimizer = optim.SGD(trainable_parameters, lr = self.lrn_rate)
    elif self.hps.optimizer == 'mom':
      optimizer = optim.SGD(trainable_parameters, lr = self.lrn_rate, momentum = 0.9)

    self.optimizer = optimizer

  def _batch_norm(self, name, x):
      """Batch normalization."""
      params_shape = [x.size()[-1]]

      beta = nn.Parameter(torch.zeros(params_shape))
      gamma = nn.Parameter(torch.ones(params_shape))

      if self.mode == 'train':
        mean = x.mean(dim = [0,2,3], keepdim = True) 
        variance = x.var (dim = [0,2,3],unbiased = False, keepdim = True )
      else:
        mean = nn.Parameter(torch.zeros(params_shape), requires_grad = False)
        variance = nn.Parameter(torch.ones(params_shape), requires_grad = False)
      y = F.batch_norm(x, mean, variance, beta, gamma, eps = 0.001, training = self.mode == 'train')
      y.set_shape(x.size())
      return y

  def _residual(self, x, in_filter, out_filter, stride,
                activate_before_residual=False):
    """Residual unit with 2 sub layers."""
    if activate_before_residual:
      with torch.no_grad():
        x = self._relu(x, self.hps.relu_leakiness)
        x = self._batch_norm('init_bn', x)
        orig_x = x
    else:
      with torch.no_grad():
        orig_x = x
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, self.hps.relu_leakiness)

    with torch.no_grad():
      x = self._conv('conv1', x, 3, in_filter, out_filter, stride)

    with torch.no_grad():
      x = self._batch_norm('bn2', x)
      x = self._relu(x, self.hps.relu_leakiness)
      x = self._conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1])

    with torch.no_grad():
      if in_filter != out_filter:
        orig_x = F.avg_pool2d(orig_x, stride)
        pad = [0,0,0,0, (out_filter - in_filter) // 2, (out_filter - in_filter) // 2]
        orig_x = F.pad(orig_x, pad)
      x += orig_x

    print('image after unit', x.size())
    return x

  def _bottleneck_residual(self, x, in_filter, out_filter, stride,
                           activate_before_residual=False):
    """Bottleneck residual unit with 3 sub layers."""
    if activate_before_residual:
      with torch.no_grad():
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, self.hps.relu_leakiness)
        orig_x = x
    else:
      with torch.no_grad():
        orig_x = x
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, self.hps.relu_leakiness)

    with torch.no_grad():
      x = self._conv('conv1', x, 1, in_filter, out_filter//4, stride)

    with torch.no_grad():
      x = self._batch_norm('bn2', x)
      x = self._relu(x, self.hps.relu_leakiness)
      x = self._conv('conv2', x, 3, out_filter//4, out_filter//4, [1, 1, 1, 1])

    with torch.no_grad():
      x = self._batch_norm('bn3', x)
      x = self._relu(x, self.hps.relu_leakiness)
      x = self._conv('conv3', x, 1, out_filter//4, out_filter, [1, 1, 1, 1])

    with torch.no_grad():
      if in_filter != out_filter:
        orig_x = self._conv('project', orig_x, 1, in_filter, out_filter, stride)
      x += orig_x

    print('image after unit', x.size())
    return x

  def decay(self):
    """L2 weight decay loss."""
    costs = []
    for var in self.parameters():
      if 'DW' in var.op.name:
        costs.append(torch.nn.functional.mse_loss(var, torch.zeros_like(var)))

    return self.hps.weight_decay_rate * sum(costs)

  def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
    """Convolution."""
    with torch.no_grad():
      n = filter_size * filter_size * out_filters
      kernel = torch.nn.Parameter(torch.randn(filter_size,filter_size, in_filters, out_filters)*np.sqrt(2.0/n))
      return torch.nn.functional.conv2d(x, kernel, stride = strides, padding='SAME')

  def _relu(self, x, leakiness=0.0):
    """Relu, with optional leaky support."""
    if leakiness > 0:
      return torch.where(x < 0, leakiness * x, x)
    else:
      return F.relu(x)
  def _fully_connected(self, x, out_dim):
    """FullyConnected layer for final output."""
    x = x.view(self.hps.batch_size, -1)
    w = torch.empty(x.size(1), out_dim).uniform_(-1,1)
    b = torch.zeros(out_dim)
    return torch.matmul(x,w) + b

  def _global_avg_pool(self, x):
    assert x.dim() == 4
    return torch.mean(x, [2, 3])
