import sys
sys.path.insert(0, '/home/arvindsgulati/func_appr/pytorch_cifar/models')

from pytorch_cifar.models import PreActResNet18
from utils import GPU
from utils import get_cifar_data
from approximator import SingleSampleFunctionApproximator,\
    Generic_recurrent_classifier, SamplePairCoApproximator,\
    MultiClassificationLoss
from torch import optim
import torch.nn as nn

NUM_EPOCHS=200


def learn_to_classify_using_single(model_file='circle-single-cifar.model',reload=False,reloadName=False):
    train_data,val_data,test_data,classes = get_cifar_data()
    model = PreActResNet18(include_last=True)
    if GPU:
        model = model.cuda()
    approximator = SingleSampleFunctionApproximator(train_data,model=model,model_file=model_file)
    optimizer = optim.SGD(model.parameters(), lr=0.1,momentum=0.9, weight_decay=5e-4)
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
    #optimizer = optim.Adam(model.parameters(), lr=0.1)
    optimizer = optim.SGD(model.parameters(), lr=0.1,momentum=0.9, weight_decay=5e-4)
    criterion = MultiClassificationLoss(nn.CrossEntropyLoss())
    approximator.approximate(val_data, optimizer, criterion, NUM_EPOCHS)

if __name__=='__main__':
    #learn_to_classify_using_single()
    learn_to_classify_using_copairs()

