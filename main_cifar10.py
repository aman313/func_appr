import sys
sys.path.insert(0, '/home/aman/Documents/func_appr/pytorch_cifar/models')

from pytorch_cifar.models import PreActResNet18
import torchvision
import torch
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset
from utils import GPU
from approximator import SingleSampleFunctionApproximator,\
    Generic_recurrent_classifier, SamplePairCoApproximator,\
    MultiClassificationLoss
from torch import optim
import torch.nn as nn

NUM_EPOCHS=200

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

def learn_to_classify_using_single(model_file='circle-single-cifar.model',reload=False,reloadName=False):
    train_data,val_data,test_data,classes = get_cifar_data()
    model = PreActResNet18(include_last=True)
    if GPU:
        model = model.cuda()
    approximator = SingleSampleFunctionApproximator(train_data,model=model,model_file=model_file)
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()
    approximator.approximate(val_data, optimizer, criterion, NUM_EPOCHS)

def learn_to_classify_using_copairs(model_file='circle-copairs-cifar.model',reload=False,reloadName=False):
    train_data,val_data,test_data,classes = get_cifar_data()
    processor =  PreActResNet18(include_last=False)
    model = Generic_recurrent_classifier(processor,512,100,len(classes))
    if GPU:
        model = model.cuda()
    domain = None
    color_map ={0:'b',1:'g'}
    #domain.visualize([x[0] for x in samples], [color_map[x[1].index(1)] for x in samples])
    #exit()
    approximator = SamplePairCoApproximator(train_data,differential_model=model,model_file=model_file,domain=domain)
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    criterion = MultiClassificationLoss(nn.CrossEntropyLoss())
    approximator.approximate(val_data, optimizer, criterion, NUM_EPOCHS)

if __name__=='__main__':
    learn_to_classify_using_single()
    #learn_to_classify_using_copairs()

