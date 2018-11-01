import numpy as np
import csv
from torch.autograd import Variable
import torch
from torch import cuda
from torch import optim
import sys
import time
from math import sin
from functools import reduce
import matplotlib.pyplot as plt
import math
import torchvision
import torch
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset

GPU = cuda.is_available()
#GPU=False
def create_sample_from_domain_with_filter_functions(domain,filter_funcs,regression_func,sample_size,outfile):
    samples = []
    while len(samples)< sample_size:
        sample = domain.sample()
        forbidden = False
        for f in filter_funcs:
            if f(sample):
                forbidden = True
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
    
    def visualize(self,data_points=None,colors=None):
        raise NotImplementedError
    
class BoundedDomain(Domain):
    def size(self):
        raise NotImplementedError

class DiscreteDomain(Domain):
    pass

class ContinuousDomain(Domain):
    pass

class R_d(ContinuousDomain):
    d = -1;
    def __init__(self,d):
        self.d = d 


class BoundedRingDomain(ContinuousDomain,BoundedDomain):
    
    def __init__(self,inner_radius=0,outer_radius=1,lower_angle_bound=-math.pi,upper_angle_bound=math.pi):
        if outer_radius <=0:
            raise Exception('Ring outer radius must be positive')
        self.outer_radius = outer_radius
        self.inner_radius = inner_radius
        self.lower_angle_bound = lower_angle_bound
        self.upper_angle_bound = upper_angle_bound

    def size(self):
        gap_size = 0.5*(self.upper_angle_bound-self.lower_angle_bound)* self.inner_radius*self.inner_radius
        return 0.5*(self.upper_angle_bound-self.lower_angle_bound)* self.outer_radius*self.outer_radius - gap_size
    
    def contains(self,x):
        origin_distance = math.sqrt(math.pow(x[0],2)+math.pow(x[1],2))
        if origin_distance <= self.outer_radius and (origin_distance >self.inner_radius or origin_distance==self.inner_radius==0) and np.arctan2(x[1],x[0])>=self.lower_angle_bound and np.arctan2(x[1],x[0]) <self.upper_angle_bound:
            return True
        else:
            return False
    
    def sample(self):
        origin_distance_sample = np.random.uniform(self.inner_radius,self.outer_radius)
        theta_sample = np.random.uniform(self.lower_angle_bound,self.upper_angle_bound)
        return [origin_distance_sample*math.cos(theta_sample),origin_distance_sample*math.sin(theta_sample)]
    
    def visualize(self,data_points=None,colors=None):
        #TODO:remove out of bound points
        plt.scatter([x[0] for x in data_points],[x[1] for x in data_points],c=colors)
        plt.show()

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
    
    def size(self):
        return reduce(lambda x,y:x*y,[x[1]-x[0] for x in self.bounds])

    def visualize(self, data_points=None, colors=None):
        if self.d !=2:
            raise NotImplementedError
        #TODO:remove out of bound points
        plt.scatter([x[0] for x in data_points],[x[1] for x in data_points],c=colors)
        plt.show()

class BoundedContainerDomain(Domain):
    def __init__(self,list_of_bounded_domains):
        self._domains = list_of_bounded_domains
        
    def sample(self):
        chosen_one = np.random.choice(self._domains,1,p=[float(i.size())/sum([x.size() for x in self._domains]) for i in self._domains])[0]
        return chosen_one.sample()
    
    def contains(self, x):
        for d in self._domains:
            if d.contains(x):
                return True
        return False
        
'''
    Regression functions
'''
def sum_func(x):
    return sum(x)

def square_sum_func(x):
    return sum(x)*sum(x)

def sine_sum_func(x):
    return sum([sin(y) for y in x])

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
            return True
        else:
            return False
        
    return filter_func


'''
    learning utils
'''

def run_epoch(net,train_data_gen,criterion,opt):
    net.train()
    train_loss = 0
    num_batches = 0
    total =0
    correct=0
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
        _, predicted = output.squeeze(1).max(len(output.shape)-1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()

    
    return train_loss/num_batches,correct*1.0/total

def test(net,test_data_gen,criterion,verbose=False,compute_acc=False):
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
    total = 0
    correct = 0
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
        
        _, predicted = output.max(len(output.shape)-1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()

        total_loss += (avg_loss)
    return total_loss/num_batches,correct*1.0/total
def timeSince(startTime):
    return time.time()-startTime
import gc
def train_with_early_stopping(net,train_data_gen,val_data_gen,criterion,optimizer,num_epochs,tolerance=0.001,max_epochs_without_improv=20,verbose=False,model_out='',min_val=-1):
    val_loss_not_improved=0
    best_val_loss = None
    train_losses_list = []
    val_losses_list = []
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, 
                                                         patience=int(0.9*max_epochs_without_improv), verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=50, min_lr=0, eps=1e-08)
    startTime=time.time()
    for i in range(num_epochs):
        #print('start epoch ',i)
        train_loss,train_acc = run_epoch(net, train_data_gen, criterion, optimizer)
        val_loss,val_acc = test(net, val_data_gen, criterion, False,compute_acc=True)
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
                print('Train accuracy',train_acc)
                print ('Val loss', val_losses_list[i].item())
                print('Val accuracy',val_acc)
                print('No improvement epochs ',val_loss_not_improved)
                if best_val_loss:
                    print('Best val loss yet ',best_val_loss.item())
                sys.stdout.flush()
        if  best_val_loss is None or val_losses_list[i].item() < best_val_loss.item():
            best_val_loss = val_losses_list[i]
            if min_val > 0 and best_val_loss <= min_val:
                print('Early stopping at epoch as min val reached',i)
                break                
        if val_loss_not_improved >= max_epochs_without_improv:
            print('Early stopping at epoch',i)
            break
        gc.collect()
    net = torch.load(model_out)

    return (train_losses_list,val_losses_list)


'''
    data read utils
'''
def read_samples(filename,classes=None):
    data  = []
    with open(filename) as fl:
        reader = csv.reader(fl)
        for row in reader:
            if not classes:
                y = float(row[-1])
            else:
                y = [0]*len(classes)
                y[classes.index(int(row[-1]))] = 1
            data.append(([float(x) for x in row[:-1]],y))
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

def load_pytorch_model(*args,**kwargs):
    return torch.load(*args,**kwargs)

'''
cifar
'''

class GenHelper(Dataset):
    def __init__(self, mother, length, mapping):
        # here is a mapping from this index to the mother ds index
        self.mapping=mapping
        self.length=length
        self.mother=mother

    def __getitem__(self, index):
        return self.mother[self.mapping[index]]

    def __len__(self):
        return self.length


def train_valid_split(ds, split_fold=10, random_seed=None):
    '''
    This is a pytorch generic function that takes a data.Dataset object and splits it to validation and training
    efficiently.
    :return:
    '''
    if random_seed!=None:
        np.random.seed(random_seed)

    dslen=len(ds)
    indices= list(range(dslen))
    valid_size=dslen//split_fold
    np.random.shuffle(indices)
    train_mapping=indices[valid_size:]
    valid_mapping=indices[:valid_size]
    train=GenHelper(ds, dslen - valid_size, train_mapping)
    valid=GenHelper(ds, valid_size, valid_mapping)

    return train, valid


# Data
def get_cifar_data():
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    testset,valset = train_valid_split(testset)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=1)
    valloader = torch.utils.data.DataLoader(valset, batch_size=1, shuffle=True, num_workers=1)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=1)
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return trainloader,valloader,testloader,classes
