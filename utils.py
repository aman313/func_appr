import numpy as np
import csv
from torch.autograd import Variable
import torch
from torch import cuda
from torch import optim
import sys
import time
GPU = cuda.is_available()
#GPU=False
def create_sample_from_domain_with_filter_functions(domain,filter_funcs,regression_func,sample_size,outfile):
    samples = []
    while len(samples)< sample_size:
        sample = domain.sample()
        forbidden = False
        for f in filter_funcs:
            if f(sample):
                forbidden = False
                break
        if not forbidden:
            samples.append(sample)
    with open(outfile,'w', newline='') as of:
        writer = csv.writer(of)
        for sample in samples:
            x_y = sample
            x_y.append(regression_func(sample))
            writer.writerow(x_y)


class Domain():
    def contains(self,x):
        raise NotImplementedError
    
    def sample(self):
        raise NotImplementedError
    
class BoundedDomain(Domain):
    pass

class DiscreteDomain(Domain):
    pass

class ContinuousDomain(Domain):
    pass

class R_d(ContinuousDomain):
    d = -1;
    def __init__(self,d):
        self.d = d 

class Bounded_Rd(R_d,BoundedDomain):
    def __init__(self,d,bounds):
        R_d.__init__(self, d)
        if (len(bounds) != d):
            raise Exception("bounds for only " + str(len(bounds)) + ' out of ' +str(d) + 'specified.')
        self.bounds = bounds
    
    def contains(self, x):
        if len(x) != self.d:
            raise Exception('Dimension mismatch',len(x),self.d)
        for i in range(self.d):
            if x[i] < self.bounds[i][0] or x[i] > self.bounds[i][1]:
                return False
        return True
    
    def sample(self):
        s = [None] *self.d
        for i in range(self.d):
            s[i] = np.random.uniform(self.bounds[i][0],self.bounds[i][1])
        return s
        
    def add_samples(self,x1,x2):
        return [x1_i+x2_i for x1_i,x2_i in zip(x1,x2)]
        
'''
    Regression functions
'''
def sum_func(x):
    return sum(x)


'''
    Classification functions
'''

def region_to_class_function(region_to_class_map):
    def class_func(x):
        for region,cls in region_to_class_map.items():
            if region.contains(x):
                return cls
        
    return class_func


'''
    Filter functions
'''
def get_filter_region_in_Rd(region):
    def filter_func(x):
        if region.contains(x):
            return False
        else:
            return True
        
    return filter_func


'''
    learning utils
'''

def run_epoch(net,train_data_gen,criterion,opt):
    net.train()
    train_loss = 0
    num_batches = 0
    for (X,y) in train_data_gen():
        #print('generated batch')
        if GPU:
            X,y = Variable(X.cuda()),Variable(y.cuda())
        else:
            X,y = Variable(X),Variable(y)
        
        opt.zero_grad()
        #print(type(X))
        output = net((X))
        #print (output,y)
        loss = criterion(output.squeeze(1),y)
        loss.backward()
        #print('X, y, loss ',X,y,loss)
        train_loss += loss
        num_batches+=1
        opt.step()
    
    return train_loss/num_batches

def test(net,test_data_gen,criterion,verbose=False):
    net.eval()
    total_loss = 0
    num_batches = 0
    generator = test_data_gen
    def present_single(batched_generator):
        def single_generator():
            for batch_x,batch_y in batched_generator():
                batch_x = batch_x.tolist()
                batch_y = batch_y.tolist()
                for i in range(len(batch_x)):
                    yield (torch.FloatTensor([batch_x[i]]),torch.FloatTensor([batch_y[i]]))
            
        return single_generator
        
    if verbose:
        generator = present_single(test_data_gen)
        
    for X,y in generator():
        if GPU:
            X,y = Variable(X.cuda()),Variable(y.cuda())
        else:
            X,y = Variable(X),Variable(y)
        num_batches += 1
        output = net(X)
        avg_loss = criterion(output.squeeze(1), y)
        if verbose:
            x_list = X.data.tolist()
            print ('x,y,o,l',x_list,y.data.tolist(),output.data.tolist(),avg_loss.data.tolist())
        
        total_loss += (avg_loss)
    return total_loss/num_batches
def timeSince(startTime):
    return time.time()-startTime
import gc
def train_with_early_stopping(net,train_data_gen,val_data_gen,criterion,optimizer,num_epochs,tolerance=0.001,max_epochs_without_improv=20,verbose=False,model_out=''):
    val_loss_not_improved=0
    best_val_loss = None
    train_losses_list = []
    val_losses_list = []
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, 
                                                         patience=int(0.9*max_epochs_without_improv), verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=50, min_lr=0, eps=1e-08)
    startTime=time.time()
    for i in range(num_epochs):
        #print('start epoch ',i)
        train_loss = run_epoch(net, train_data_gen, criterion, optimizer)
        val_loss = test(net, val_data_gen, criterion, False)
        if GPU:
            train_losses_list.append(train_loss.data.cpu())
            val_losses_list.append(val_loss.data.cpu())
            del train_loss,val_loss

        else:
            train_losses_list.append(train_loss.data)
            val_losses_list.append(val_loss.data)
        scheduler.step(val_losses_list[i].item())
        #optimizer.step()
        if i > 0:
            if best_val_loss.item() ==0.0:
                break
            if ((best_val_loss.item() -val_losses_list[i].item())/best_val_loss.item()) > tolerance:
                val_loss_not_improved = 0
                torch.save(net, model_out)
            else:
                val_loss_not_improved +=1
        if verbose:
            if i%10 ==0:
                print ('Epoch',i)
                print('Time Per Epoch',"%.2f" % (timeSince(startTime)/(i+1)/60), " minutes")
                print('Remaining Estimate',"%.2f" % (timeSince(startTime)*(num_epochs-i-1)/(i+1)/60), " minutes")
                print ('Train loss',train_losses_list[i].item())
                print ('Val loss', val_losses_list[i].item())
                print('No improvement epochs ',val_loss_not_improved)
                if best_val_loss:
                    print('Best val loss yet ',best_val_loss.item())
                sys.stdout.flush()
        if  best_val_loss is None or val_losses_list[i].item() < best_val_loss.item():
            best_val_loss = val_losses_list[i]
        if val_loss_not_improved >= max_epochs_without_improv:
            print('Early stopping at epoch',i)
            break
        gc.collect()
    net = torch.load(model_out)
    return (f,val_losses_list)


'''
    data read utils
'''
def read_samples(filename):
    data  = []
    with open(filename) as fl:
        reader = csv.reader(fl)
        for row in reader:
            data.append(([float(x) for x in row[:-1]],float(row[-1])))
    return data


'''
    Misc
'''
import random
def reservoir_sample(iterable, n):
    """
    Returns @param n random items from @param iterable.
    """
    reservoir = []
    for t, item in enumerate(iterable):
        if t < n:
            reservoir.append(item)
        else:
            m = random.randint(0,t)
            if m < n:
                reservoir[m] = item
    return reservoir  

def load_pytorch_model(file_path):
    return torch.load(file_path)
