#-*- coding: utf-8 -*-
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Configuration file')
arg_lists = []


def add_argument_group(name):
  arg = parser.add_argument_group(name)
  arg_lists.append(arg)
  return arg


def str2bool(v):
  return v.lower() in ('true', '1')


# Network
net_arg = add_argument_group('Network')
net_arg.add_argument('--input_embed', type=int, default=128, help='actor input embedding')
net_arg.add_argument('--hidden_dim', type=int, default=128, help='actor LSTM num_neurons')

# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('--batch_size', type=int, default=256, help='batch size') 
data_arg.add_argument('--input_dimension', type=int, default=2, help='city dimension')
data_arg.add_argument('--max_length', type=int, default=20, help='number of deliveries') # this excludes depot
data_arg.add_argument('--speed', type=float, default=10.0, help='agent speed') ############################### speed 10
data_arg.add_argument('--kNN', type=int, default=5, help='int for random k_nearest_neighbor') ################ kNN 5
data_arg.add_argument('--width_mean', type=float, default=30.0, help='tw width gaussian distribution mean') ### [5,2] n20w20, [11,5] n20w40, [17,7] n20w60
data_arg.add_argument('--width_std', type=float, default=11.0, help='tw width gaussian distribution std') ##### [22,9] n20w80, [30,11] n20w100
data_arg.add_argument('--dir_', type=str, default='n20w100', help='Dumas benchmarch instances') ###############

# Training / test parameters
train_arg = add_argument_group('Training')
train_arg.add_argument('--nb_epoch', type=int, default=220000, help='nb epoch')
train_arg.add_argument('--lr1_start', type=float, default=0.001, help='actor learning rate')
train_arg.add_argument('--lr1_decay_step', type=int, default=5000, help='lr1 decay step')
train_arg.add_argument('--lr1_decay_rate', type=float, default=0.96, help='lr1 decay rate')

train_arg.add_argument('--beta', type=int, default=10, help='weight for TW constraint') ###################### 3 during training / 10 for test
train_arg.add_argument('--temperature', type=float, default=3.0, help='pointer_net initial temperature') #####
train_arg.add_argument('--C', type=float, default=10.0, help='pointer_net tan clipping')

# Misc
misc_arg = add_argument_group('User options') #####################################################

misc_arg.add_argument('--pretrain', type=str2bool, default=False, help='faster datagen for infinite speed')
misc_arg.add_argument('--inference_mode', type=str2bool, default=True, help='switch to inference mode when model is trained') 
misc_arg.add_argument('--restore_model', type=str2bool, default=True, help='whether or not model is retrieved')

misc_arg.add_argument('--save_to', type=str, default='speed10/s10_k5_n20w100', help='saver sub directory') #####################
misc_arg.add_argument('--restore_from', type=str, default='speed10/s10_k5_n20w100', help='loader sub directory') ###############
misc_arg.add_argument('--log_dir', type=str, default='summary/test', help='summary writer log directory') 



def get_config():
  config, unparsed = parser.parse_known_args()
  return config, unparsed


def print_config():
  config, _ = get_config()
  print('\n')
  print('Data Config:')
  print('* Batch size:',config.batch_size)
  print('* Sequence length:',config.max_length)
  print('* City coordinates:',config.input_dimension)
  print('\n')
  print('Network Config:')
  print('* Restored model:',config.restore_model)
  print('* Actor input embedding:',config.input_embed)
  print('* Actor hidden_dim (num neurons):',config.hidden_dim)
  print('* Actor tan clipping:',config.C)
  print('\n')
  if config.inference_mode==False:
  	print('Training Config:')
  	print('* Nb epoch:',config.nb_epoch)
  	print('* Temperature:',config.temperature)
  	print('* Actor learning rate (init,decay_step,decay_rate):',config.lr1_start,config.lr1_decay_step,config.lr1_decay_rate)
  else:
  	print('Testing Config:')
  print('* Summary writer log dir:',config.log_dir)
  print('\n')
