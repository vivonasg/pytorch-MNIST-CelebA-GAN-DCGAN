import os, time
import itertools
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torchvision.utils as vutils
from pytorch_MNIST_CAPS_DCGAN import *



import argparse
[0.9,0.1,0.5,0.005]
param_hyper=[[0.7,0.8,0.9],[0.1,0.2,0.3],[0.2,0.5,0.7],[0.005,0.01,0.1]]
lr_hyper=[0.02,0.002]
SN_hyper=[True,False]


for param_0 in param_hyper[0]:
	for param_1 in param_hyper[1]:
		for param_2 in param_hyper[2]:
			for param_3 in param_hyper[3]:
				for lr in lr_hyper:
					for SN in SN_hyper:

						hyper_tag=str(param_0)+'-'+str(param_1)+'-'+str(param_2)+'-'+str(param_3)+'-'+str(lr)+'-'+str(SN)
						
						print(hyper_tag)

						run_model(lr=lr,
				            SN_bool=SN, 
				            D_param=[param_0,param_1,param_2,param_3],
				            num_iter_limit=1000,
				            hyperparam_tag=hyper_tag)
