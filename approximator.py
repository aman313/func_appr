import torch.nn as nn
from torch import cuda
from itertools import permutations
from utils import train_with_early_stopping, read_samples, Bounded_Rd
import numpy as np
from torch.autograd import Variable
import random
import torch
from itertools import product
from utils import reservoir_sample
from cytoolz import comp
from utils import load_pytorch_model
from spacy.tests.parser.test_nn_beam import batch_size

class Rd_difference_approximator(nn.Module):
    def __init__(self):
        super(Rd_difference_approximator, self).__init__()
        self.linear1 = nn.Linear(2,10)
        self.linear2 = nn.Linear(10,20)
        self.linear3 = nn.Linear(20,40)
        self.linear4 = nn.Linear(40,1)
        
    def forward(self,input):
        return self.linear4(nn.Tanh()(self.linear3(nn.Tanh()(self.linear2(nn.Tanh()(self.linear1(input)))))))

class Rd_siamese_approximator(nn.Module):
    def __init__(self):
        super(Rd_siamese_approximator, self).__init__()
        self.linear = nn.Linear(2,10)
        self.bilinear = nn.Bilinear(20,20,2)
        self.linear1 = nn.Linear(10,20)

    def forward(self,input):
        inp1 = input[:,0,:]
        inp2 = input[:,1,:]
        repr1 = nn.Tanh()(self.linear1(nn.Tanh()(self.linear(inp1))))
        repr2 = nn.Tanh()(self.linear1(nn.Tanh()(self.linear(inp2))))
        comp = self.bilinear(repr1,repr2)
        return comp

class Rd_symmetric_siamese_approximator(nn.Module):
    def __init__(self):
        super(Rd_symmetric_siamese_approximator, self).__init__()
        self.linear = nn.Linear(2,12)
        self.bilinear = nn.Bilinear(24,24,1)
        self.linear1 = nn.Linear(12,24)

    def forward(self,input):
        inp1 = input[:,0,:]
        inp2 = input[:,1,:]
        repr1 = nn.Tanh()(self.linear1(nn.Tanh()(self.linear(inp1))))
        repr2 = nn.Tanh()(self.linear1(nn.Tanh()(self.linear(inp2))))
        comp1 = self.bilinear(repr1,repr2)
        comp2= self.bilinear(repr2,repr1)
        comp=torch.cat([comp1,comp2],dim=-1)
        return comp

def compute_input_differences(input_pair_gen,domain):
    input_diffs = []
    for x1,x2 in input_pair_gen:
        input_diffs.append(domain.add_samples(x1,[-1*x for x in x2]))
    return input_diffs

def generate_differences_batches(data,domain,batch_size=10,is_y=True):
    def generate():
        perm_gen = permutations(data,2)
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

def generate_pairs_batch(data,domain,batch_size=10,is_y=True):
    def generate():
        random.shuffle(data)
        ldata=data[:50]
        perm_gen = permutations(ldata,2)
        while True:
            batch_data = []
            for i in range(batch_size):
                z1,z2 = next(perm_gen)
                batch_data.append((z1,z2))
            if len(batch_data) == 0:
                break
            if is_y:
                yield (torch.FloatTensor([[z[0][0],z[1][0]] for z in batch_data]),torch.FloatTensor([[z[0][1],z[1][1]] for z in batch_data]))
            else:
                yield (torch.FloatTensor([[z[0][0],z[1][0]] for z in batch_data]))
    return generate

def generate_batch(data,batch_size=50,is_y=True):
    data = (x for x in data)
    def generate():
        while(True):
            batch_data = []
            for i in range(batch_size):
                batch_data.append(next(data))
            if len(batch_data) == 0:
                break
            if is_y:
                yield (torch.FloatTensor([x[0] for x in batch_data]),torch.FloatTensor([x[1] for x in batch_data]))
            else:
                yield (torch.FloatTensor([x[0] for x in batch_data]))
                
    return generate

class FunctionApproximator():
    pass

class SingleSampleFunctionApproximator(FunctionApproximator):
    def __init__(self,train_data,model,model_file=''):
        self.model = model
        self.train_data = train_data
        self.model_file=model_file
    
    def approximate(self,val_gen,optimizer,criterion,num_epochs=100):
        train_data_gen = lambda : ((batch_x,batch_y) for batch_x,batch_y in generate_batch(self.train_data)())
        val_data_gen = lambda :((batch_x,batch_y) for batch_x,batch_y in generate_batch(val_gen)())
        train_losses,val_losses = train_with_early_stopping(self.model,train_data_gen,val_data_gen,criterion,optimizer,num_epochs,tolerance=0.0001,max_epochs_without_improv=2000,verbose=True,model_out=self.model_file)
        print(np.mean(train_losses),np.mean(val_losses))
    
    def predict(self,test_data):
        outputs = self.model(torch.FloatTensor([x[0] for x in test_data]))
        return outputs

    def evaluate(self,test_data,criterion):
        predictions = self.predict(test_data)
        print(criterion(predictions.squeeze(1),Variable(torch.FloatTensor([x[1] for x in test_data]))))    
    
class RandomSamplePairFunctionApproximator(FunctionApproximator):
    def __init__(self,train_data,domain,diff_train_sample_size=50,eval_sample_size=10,differential_model=None,model_file=''):
        self.train_data = train_data
        self.domain = domain
        self.diff_train_sample_size = diff_train_sample_size
        self.eval_sample_size = eval_sample_size
        self.diff_train_data = train_data#random.sample(self.train_data,self.diff_train_sample_size)
        self.differential_model = differential_model
        self.model_file = model_file
   
    def approximate(self):
        raise NotImplementedError
    
class SamplePairDifferenceFunctionApproximator(RandomSamplePairFunctionApproximator):
    '''
        Approximate a function to map difference between del_x and y
        For test samples - sample a fixed number of examples combine results
    '''
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        
    def approximate(self,val_gen,optimizer,criterion,num_epochs=100):
        diff_train_data_gen = lambda : ((torch.stack(batch_x),torch.stack(batch_y)) for batch_x,batch_y in generate_pairs_batch(self.diff_train_data, self.domain)())
        diff_val_data_gen = lambda :((torch.stack(batch_x),torch.stack(batch_y)) for batch_x,batch_y in generate_pairs_batch(val_gen, self.domain)())
        train_losses,val_losses = train_with_early_stopping(self.differential_model,diff_train_data_gen,diff_val_data_gen,criterion,optimizer,num_epochs,tolerance=0.0001,max_epochs_without_improv=2000,verbose=True,model_out=self.model_file)
        print(np.mean(train_losses),np.mean(val_losses))
        #self.evaluate(val_gen)
    
    def predict(self,test_data):
        reference_data = random.sample(self.train_data,self.eval_sample_size)
        ref_x = [x[0] for x in reference_data]
        ref_y = [x[1] for x in reference_data]
        augmented_test_data = product(test_data,ref_x)
        pair_outputs = self.differential_model(augmented_test_data)
        pass
        
    def evaluate(self,test_data):
        pass
    
class SamplePairCoApproximator(RandomSamplePairFunctionApproximator):
    def __init__t(self,*args,**kwargs):
        super().__init(*args,**kwargs)

    def approximate(self,val_gen,optimizer,criterion,num_epochs=100):
        diff_train_data_gen = generate_pairs_batch(self.diff_train_data, self.domain,32)
        diff_val_data_gen = generate_pairs_batch(reservoir_sample(val_gen,20), self.domain)
        train_losses,val_losses = train_with_early_stopping(self.differential_model,diff_train_data_gen,diff_val_data_gen,criterion,optimizer,num_epochs,tolerance=0.0001,max_epochs_without_improv=2000,verbose=True,model_out=self.model_file)
        print(np.mean(train_losses),np.mean(val_losses))
        #self.evaluate(val_gen)
    
    def predict(self,test_data):
        reference_data = random.sample(self.train_data,self.eval_sample_size)
        ref_x = [x[0] for x in reference_data]
        ref_y = [x[1] for x in reference_data]
        test_x = [x[0] for x in test_data]
        paired_with_reference = [x for x in product(test_x,ref_x)]
        pair_outputs = self.differential_model(torch.FloatTensor(paired_with_reference))
        outputs = []
        for i in range(0,len(test_data)*len(ref_x),len(ref_x)):
            predictions = [x[0].detach().numpy() for x in pair_outputs[i:i+len(ref_x)]]
            predictions_reference = [x[1].detach() for x in pair_outputs[i:i+len(ref_x)] ]
            weights = [1-abs((x-y)/y) for x,y in zip(predictions_reference,ref_y) ] 
            mean_prediction = np.average(predictions,weights=weights)
            outputs.append(mean_prediction)
        return outputs
    
    def evaluate(self,test_data,criterion):
        predictions = self.predict(test_data)
        print(criterion(Variable(torch.FloatTensor(predictions)),Variable(torch.FloatTensor([x[1] for x in test_data]))))
        
def plot_figure(inputs,predictions,outfile):
    pass

if __name__=='__main__':
    siamese_model = load_pytorch_model('square-siamese.model')
    single_model = load_pytorch_model("square-single.model")
    ood_samples = read_samples('square.csv')
    samples = read_samples('square.csv')
    R_2 ={'num_dims':2,'bounds':[(-1,1),(-1,1)]}
    domain = Bounded_Rd(R_2['num_dims'],R_2['bounds'])
    approximator = SamplePairCoApproximator(samples,domain,differential_model=siamese_model)
    approximator_single = SingleSampleFunctionApproximator(samples,model=single_model)
    criterion = nn.MSELoss()
    approximator.evaluate(ood_samples, criterion)
    approximator_single.evaluate(ood_samples, criterion)
    siamese_predictions = approximator.predict(ood_samples)
    single_predictions = approximator_single.predict(ood_samples)
    plot_figure(ood_samples,siamese_predictions,'results-siamese.png')
    plot_figure(ood_samples,single_predictions,'results-single.png')
