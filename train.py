import os
import argparse

from networks.lenet import LeNet
#from networks.pure_cnn import PureCnn
#from networks.network_in_network import NetworkInNetwork
from networks.resnet import ResNet


if __name__ == '__main__':
  models = {
    'lenet': LeNet,
    #'pure_cnn': PureCnn,
    #'net_in_net': NetworkInNetwork,
    'resnet': ResNet
  }
  
  parser = argparse.ArgumentParser(description='Train models on Cifar10')
  parser.add_argument('--model', choices=models.keys(), required=True, help='Specify a model by name to train.')
  parser.add_argument('--epochs', default=None, type=int)
  parser.add_argument('--batch_size', default=None, type=int)
  
  args = parser.parse_args()
  model_name = args.model
  
  args = {k: v for k,v in vars(args).items() if v != None }
  del args['model']
  
  model = models[model_name](**args,load_weights=True)
  
  model.train()