import torch.nn as nn
from torch import cuda
from itertools import permutations
from utils import train_with_early_stopping
import numpy as np
from torch.autograd import Variable
import random
import torch
from itertools import product
from utils import reservoir_sample

class Rd_difference_approximator(nn.Module):
    def __init__(self):
        super(Rd_difference_approximator, self).__init__()
        self.linear1 = nn.Linear(2,10)
        self.linear2 = nn.Linear(10,50)
        self.linear3 = nn.Linear(50,50)
        self.linear4 = nn.Linear(50,20)
        self.linear5 = nn.Linear(20,1)
        
    def forward(self,input):
        return self.linear5(nn.Tanh()(self.linear4(nn.Tanh()(self.linear3(nn.Tanh()(self.linear2(nn.Tanh()(self.linear1(input)))))))))

def compute_input_differences(input_pair_gen,domain):
    input_diffs = []
    for x1,x2 in input_pair_gen:
        input_diffs.append(domain.add_samples(x1,[-1*x for x in x2]))
    return input_diffs

def generate_differences_batches(data,domain,batch_size=10,is_y=True):
    def generate():
        perm_gen = permutations(data,2)
        #print(len(perm_gen))
        while(True):
            batch_data = []
            for i in range(batch_size):
                z1,z2 = next(perm_gen)
                batch_data.append((z1,z2))
            x_diff = compute_input_differences(((z1[0],z2[0]) for z1,z2 in batch_data), domain)
            if is_y:
                y_diff = [z1[1]-z2[1] for z1,z2 in batch_data]
                yield (torch.FloatTensor(x_diff),torch.FloatTensor(y_diff))
            else:
                yield (torch.FloatTensor(x_diff) )
    return generate


class Differential_function_approximator():
    '''
        Approximate a function to map difference between del_x and y
        For test samples - sample a fixed number of examples combine results
    '''
    def __init__(self,train_data,domain,diff_train_sample_size=50,eval_sample_size=10,differential_model=None,model_file=''):
        self.train_data = train_data
        self.domain = domain
        self.diff_train_sample_size = diff_train_sample_size
        self.eval_sample_size = eval_sample_size
        self.diff_train_data = random.sample(self.train_data,self.diff_train_sample_size)
        self.differential_model = differential_model
        self.model_file = model_file
        
    def approximate(self,val_gen,optimizer,criterion,num_epochs=100):
        #diff_train_data_gen = lambda : ((torch.stack(batch_x),torch.stack(batch_y)) for batch_x,batch_y in generate_differences_batches(self.diff_train_data, self.domain)())
        #diff_val_data_gen = lambda :((torch.stack(batch_x),torch.stack(batch_y)) for batch_x,batch_y in generate_differences_batches(val_gen, self.domain)())
        diff_train_data_gen = generate_differences_batches(self.diff_train_data, self.domain)
        diff_val_data_gen = generate_differences_batches(reservoir_sample(val_gen,20), self.domain)
        train_losses,val_losses = train_with_early_stopping(self.differential_model,diff_train_data_gen,diff_val_data_gen,criterion,optimizer,num_epochs,tolerance=0.0001,max_epochs_without_improv=2000,verbose=True,model_out=self.model_file)
        print(np.mean(train_losses),np.mean(val_losses))
        #self.evaluate(val_gen)
        
    def predict(self,test_data):
        reference_data = random.sample(self.train_data,self.eval_sample_size)
        ref_x = [x[0] for x in reference_data]
        ref_y = [x[1] for x in reference_data]
        augmented_test_data = product(test_data,ref_x)
        differences = compute_input_differences(augmented_test_data, self.domain)
        output_differences = self.net(differences)
        
    def evaluate(self,test_data):
        pass