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

"""Utility functions for training the MentorNet models."""

import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as 
from torch.utils.tensorboard import SummaryWriter # you need to have TensorBoard installed 

def summarize_data_utilization(v, global_step, batch_size, epsilon=0.001):
  """Summarizes the samples of non-zero weights during training.

  Args:
    v: a tensor [batch_size, 1] represents the sample weights.
      0: loss, 1: loss difference to the moving average, 2: label and 3: epoch,
      where epoch is an integer between 0 and 99 (the first and the last epoch).
    tf_global_step: the tensor of the current global step.
    batch_size: an integer batch_size
    epsilon: the rounding error. If the weight is smaller than epsilon then set
      it to zero.
  Returns:
    data_util: a tensor of data utilization.
  """
  nonzero_v = nn.Parameter(torch.zeros([]).to(dtype=torch.float32), requires_grad = False )
  nonzero_v.name = 'data_util/nonzero_v'
      
                          
  rounded_v = torch.maximum(v - epsilon, torch.tensor(0.0))


  # Log data utilization
  nonzero_v = torch.count_nonzero(rounded_v, dtype=torch.float32)

  # slim runs extra sessions to log, causing
  # the value lager than 1 (data are fed but the global step is not changed)
  # so we use tf_global_step + 2
  data_util = (nonzero_v) / float(batch_size) / (
      float(global_step) + 2)
  data_util = torch.minimum(data_util, torch.tensor(1.0))
  data_util = data_util.detach

   data_util = data_util.item()
   'data_util/data_util'
  batch_sum_v = torch.sum(v).item()
   'data_util/batch_sum_v'
  return data_util

def parse_dropout_rate_list(str_list):
  """Parse a comma-separated string to a list.

  The format follows [dropout rate, epoch_num]+ and the result is a list of 100
  dropout rate.

  Args:
    str_list: the input string.
  Returns:
    result: the converted list
  """
  str_list = np.array(str_list)
  values = str_list[np.arange(0, len(str_list), 2)]
  indexes = str_list[np.arange(1, len(str_list), 2)]

  values = [float(t) for t in values]
  indexes = [int(t) for t in indexes]

  assert len(values) == len(indexes) and np.sum(indexes) == 100
  for t in values:
    assert t >= 0.0 and t <= 1.0

  result = []
  for t in range(len(str_list) // 2):
    result.extend([values[t]] * indexes[t])
  return result


def mentornet_nn(input_features,
                 label_embedding_size=2,
                 epoch_embedding_size=5,
                 num_fc_nodes=20):
  
  """The neural network form of the MentorNet.

  An implementation of the mentornet. The details are in:
  Jiang, Lu, et al. "MentorNet: Learning Data-Driven Curriculum for Very Deep
  Neural Networks on Corrupted Labels." ICML. 2018.
  http://proceedings.mlr.press/v80/jiang18c/jiang18c.pdf

  Args:
    input_features: a [batch_size, 4] tensor. Each dimension corresponds to
      0: loss, 1: loss difference to the moving average, 2: label and 3: epoch,
      where epoch is an integer between 0 and 99 (the first and the last epoch).
    label_embedding_size: the embedding size for the label feature.
    epoch_embedding_size: the embedding size for the epoch feature.
    num_fc_nodes: number of hidden nodes in the fc layer.
  Returns:
    v: [batch_size, 1] weight vector.
  """
  batch_size = int(input_features.size(0))

  losses = input_features[:, 0].view(-1,1)
  loss_diffs =input_features[:, 1].view(-1, 1)
  labels=input_features[:,2].long().view(-1,1)
  epochs=input_features[:,3].long().view(-1,1)
  epochs = torch.min(epochs, torch.ones(batch_size, 1, dtype=torch.int32) * 99)
  if losses.dim() <= 1:
    num_steps = 1
  else:
    num_steps = int(losses.size(1))

  with torch.no_grad():
    label_embedding = nn.Parameter(torch.empty(2, label_embedding_size))
    epoch_embedding =nn.Parameter(torch.empty(100, epoch_embedding_size), requires_grad=False)

    lstm_inputs = torch.stack([losses, loss_diffs], feat_dim=1)
    lstm_inputs = lstm_inputs.squeeze()
    

    forward_cell = nn.LSTMCell(1, bias = False)
    backward_cell = nn.LSTMCell(1, bias = False)
    
    ## used chatgpt for reference  
    out_state_fw = []
    out_state_bw = []
    hidden_fw = torch.zeros(batch_size, forward_cell.hidden_size)
    hidden_bw = torch.zeros(batch_size,backward_cell.hidden_size)
    cell_fw = torch.zeros(batch_size,forward_cell.hidden_size)
    cell_bw = torch.zeros(batch_size, backward_cell.hidden_size)

    for timestep in range(num_steps):
      hidden_fw, cell_fw = forward_cell(lstm_inputs[:, timestep],(hidden_fw, cell_fw))
      hidden_bw, cell_bw = backward_cell(lstm_inputs[:, num_steps - timestep - 1], (hidden_bw, cell_bw))
      out_state_fw.append(hidden_fw)
      out_state_bw.append(hidden_bw)

    out_state_fw = torch.stack(out_state_fw, dim = 1)
    out_state_bw = torch.stack(out_state_bw, dim = 1)
    ## END used chatgpt for reference
   

    label_inputs = label_embedding[labels.squeeze()]
    epoch_inputs = epoch_embedding[epochs.squeeze()]
    h = torch.cat([out_state_fw[:,0], out_state_bw[:,0]], dim = 1)
    ## torch.cat function is used to concatenate tensors along certain dimension and is exact equivalent of tf.concat, the code concatenates first timestep's hidden states from fw and bw directions along the second dimension (dim - 1), which creates tensor h 
    feat = torch.cat([label_inputs, epoch_inputs, h], dim = 1)
    ## basically same as previous line - feat = contains the concatenated values from label_inputsand epoch_inputs and h in that order
    feat_dim = feat.size(1)
    ##pytorch u use .size() to get shape of tensor

    fc_1 = torch.matmul(feat,torch.randn(feat_dim, num_fc_nodes)) + torch.randn(num_fc_nodes)
    ##with torch.randn(feat_dim,num_fc_nodes), you create tensor of random values drawn from normal distribution with standard deviation 1 and the shape of the tensor is (num_fc_nodes, 1) (this represents the weights for the first fully connected layer (set of parameters that determine influence inputs have on output))
    ## torch.matmul(fc_1, torch.randn(num_fc_nodes,1)) -- this performs matrix multiplications between output of first layer (fc_1) and weight tensor created prior
    fc_1 = torch.tanh(fc_1)
    ## applies hyperbolic tangent activation function to tensor fc_1
    # Output layer with linear activation
    out_layer = torch.matmul(fc_1, torch.randn(num_fc_nodes, 1)) + torch.randn(1)
    ## creates tensor of random values w normal distribution and standard dev 1 - shape is (num_fc_nodes) , matrix multiplication betwene fc_1 and weigh tensor in previous line, torch.randn(1) creates a tensor of random values w normal distribution shape of tensor is 1, which reps bias for output layer and bias tensor is added to matrx multiplication prior
    return out_layer


def mentornet(epoch,
              loss,
              labels,
              loss_p_percentile,
              example_dropout_rates,
              burn_in_epoch=18,
              fixed_epoch_after_burn_in=True,
              loss_moving_average_decay=0.9,
              debug=False):
  """The MentorNet to train with the StudentNet.

     The details are in:
    Jiang, Lu, et al. "MentorNet: Learning Data-Driven Curriculum for Very Deep
    Neural Networks on Corrupted Labels." ICML. 2018.
    http://proceedings.mlr.press/v80/jiang18c/jiang18c.pdf

  Args:
    epoch: a tensor [batch_size, 1] representing the training percentage. Each
      epoch is an integer between 0 and 99.
    loss: a tensor [batch_size, 1] representing the sample loss.
    labels: a tensor [batch_size, 1] representing the label. Every label is set
      to 0 in the current version.
    loss_p_percentile: a 1-d tensor of size 100, where each element is the
      p-percentile at that epoch to compute the moving average.
    example_dropout_rates: a 1-d tensor of size 100, where each element is the
      dropout rate at that epoch. Dropping out means the probability of setting
      sample weights to zeros proposed in Liang, Junwei, et al. "Learning to
      Detect Concepts from Webly-Labeled Video Data." IJCAI. 2016.
    burn_in_epoch: the number of burn_in_epoch. In the first burn_in_epoch, all
      samples have 1.0 weights.
    fixed_epoch_after_burn_in: whether to fix the epoch after the burn-in.
    loss_moving_average_decay: the decay factor to compute the moving average.
    debug: whether to print the weight information for debugging purposes.

  Returns:
    v: [batch_size, 1] weight vector.
  """
  class MentorNet(nn.Module): 
    def __init__(self, burn_in_epoch = 18, fixed_epoch_after_in = True): 
      super(MentorNet, self).__init__()
      self.burn_in_epoch = burn_in_epoch 
      self.fixed_epoch_after_burn_in = fixed_epoch_after_burn_in 
      cur_epoch = epoch
    else:
      cur_epoch = tf.to_int32(tf.minimum(epoch, burn_in_epoch))

    v_ones = tf.ones(tf.shape(loss), tf.float32)
    v_zeros = tf.zeros(tf.shape(loss), tf.float32)
    upper_bound = tf.cond(cur_epoch < (burn_in_epoch - 1), lambda: v_ones,
                          lambda: v_zeros)

    this_dropout_rate = tf.squeeze(
        tf.nn.embedding_lookup(example_dropout_rates, cur_epoch))
    this_percentile = tf.squeeze(
        tf.nn.embedding_lookup(loss_p_percentile, cur_epoch))

    percentile_loss = tf.contrib.distributions.percentile(
        loss, this_percentile * 100)
    percentile_loss = tf.convert_to_tensor(percentile_loss)

    loss_moving_avg = loss_moving_avg.assign(
        loss_moving_avg * loss_moving_average_decay +
        (1 - loss_moving_average_decay) * percentile_loss)

    slim.summaries.add_scalar_summary(percentile_loss, 'debug/percentile_loss')
    slim.summaries.add_scalar_summary(this_dropout_rate, 'debug/dropout_rate')
    slim.summaries.add_scalar_summary(cur_epoch, 'debug/epoch_step')
    slim.summaries.add_scalar_summary(loss_moving_avg,
                                      'debug/loss_moving_percentile')

    ones = tf.ones([tf.shape(loss)[0], 1], tf.float32)

    epoch_vec = tf.scalar_mul(tf.to_float(cur_epoch), ones)
    lossdiff = loss - tf.scalar_mul(loss_moving_avg, ones)

  input_data = tf.squeeze(tf.stack([loss, lossdiff, labels, epoch_vec], 1))
  v = tf.nn.sigmoid(mentornet_nn(input_data), name='v')
  # Force select all samples in the first burn_in_epochs
  v = tf.maximum(v, upper_bound, 'v_bound')

  v_dropout = tf.py_func(probabilistic_sample,
                         [v, this_dropout_rate, 'random'], tf.float32)
  v_dropout = tf.reshape(v_dropout, [-1, 1], name='v_dropout')

  # Print information in the debug mode.
  if debug:
    v_dropout = tf.Print(
        v_dropout,
        data=[cur_epoch, loss_moving_avg, percentile_loss],
        summarize=64,
        message='epoch, loss_moving_avg, percentile_loss')
    v_dropout = tf.Print(
        v_dropout, data=[lossdiff], summarize=64, message='loss_diff')
    v_dropout = tf.Print(v_dropout, data=[v], summarize=64, message='v')
    v_dropout = tf.Print(
        v_dropout, data=[v_dropout], summarize=64, message='v_dropout')
  return v_dropout


def probabilistic_sample(v, rate=0.5, mode='binary'):
  """Implement the sampling techniques.

  Args:
    v: [batch_size, 1] the weight column vector.
    rate: in [0,1]. 0 indicates using all samples and 1 indicates
      using zero samples.
    mode: a string. One of the following 1) actual returns the actual sampling;
      2) binary returns binary weights; 3) random performs random sampling.
  Returns:
    v: [batch_size, 1] weight vector.
  """
  assert rate >= 0 and rate <= 1
  epsilon = 1e-5
  p = np.copy(v)
  p = np.reshape(p, -1)
  if mode == 'random':
    ids = np.random.choice(
        p.shape[0], int(p.shape[0] * (1 - rate)), replace=False)
  else:
    # Avoid 1) all zero loss and 2) zero loss are never selected.
    p += epsilon
    p /= np.sum(p)
    ids = np.random.choice(
        p.shape[0], int(p.shape[0] * (1 - rate)), p=p, replace=False)
    result = np.zeros(v.shape, dtype=np.float32)
    if mode == 'binary':
      result[ids, 0] = 1
    else:
      result[ids, 0] = v[ids, 0]
    return result
