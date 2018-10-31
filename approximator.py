import torch.nn as nn
from torch import cuda
from itertools import permutations
from utils import train_with_early_stopping, read_samples, Bounded_Rd,\
    BoundedRingDomain
import numpy as np
from torch.autograd import Variable
import random
import torch
from itertools import product
from utils import reservoir_sample
from cytoolz import comp
from utils import load_pytorch_model,GPU
from collections import Counter
from torch.tensor import Tensor
from torch.nn.modules.loss import BCEWithLogitsLoss

class Generic_recurrent_classifier(nn.Module):
    def __init__(self, single_item_processor_network, processor_out_dim,recurrent_hidden_size, num_classes,is_combiner_bidirectional=True):
        super(Generic_recurrent_classifier, self).__init__()
        self.single_item_processor_network = single_item_processor_network
        self.recurrent_hidden_size = recurrent_hidden_size
        self.num_classes = num_classes
        self.processor_out_dim = processor_out_dim
        self.is_combiner_bidirectional = is_combiner_bidirectional
        self.recurrent_combiner = nn.LSTM(processor_out_dim,recurrent_hidden_size,bidirectional=is_combiner_bidirectional)
        if self.is_combiner_bidirectional:
            recurrent_hidden_size = 2 * recurrent_hidden_size
        self.output_layer = nn.Linear(recurrent_hidden_size,num_classes)
        
    def forward(self,inputs):
        reprs = []
        for i in range(inputs.shape[1]):
            hidden_repr = nn.ReLU()(self.single_item_processor_network(inputs[:,i,:]))
            reprs.append(hidden_repr)
        
        output,hidden = self.recurrent_combiner(torch.stack(reprs))
        projected_per_time_step = []
        for i in range(output.shape[0]):
            projected_per_time_step.append(self.output_layer(output[i,:]))
        comp=torch.stack(projected_per_time_step).permute(1,0,2)
        return comp

class Rd_classifier(nn.Module):
    def __init__(self):
        super(Rd_classifier, self).__init__()
        self.linear1 = nn.Linear(2,1024)
        self.batch_norm1 = nn.BatchNorm1d(1024)
        self.linear2 = nn.Linear(1024,1024)
        self.batch_norm2 = nn.BatchNorm1d(1024)
        self.linear3 = nn.Linear(1024,2)
        
    
    def forward(self, input):
        return self.linear3(nn.Dropout(0.2)(self.batch_norm2(nn.ReLU()(self.linear2(nn.Dropout(0.2)(self.batch_norm1(nn.ReLU()(self.linear1(input)))))))))

class MultiBCEWithLogitsLoss(nn.Module):
    def __init__(self):
        super(MultiBCEWithLogitsLoss,self).__init__()
    
    def forward(self,pred,gold):
        loss = 0
        for i in range(pred.shape[1]):
            gold_i = gold[:,i]
            pred_i = pred[:,i]
            loss+=nn.BCEWithLogitsLoss()(pred_i,gold_i)
        return loss

class MultiClassificationLoss(nn.Module):
    def __init__(self,criterion):
        super(MultiClassificationLoss,self).__init__()
        self.criterion = criterion
    
    def forward(self,pred,gold):
        loss = 0
        for i in range(pred.shape[1]):
            gold_i = gold[:,i]
            pred_i = pred[:,i]
            loss+=self.criterion(pred_i,gold_i)
        return loss

class MultiCEWithLogitsLoss(nn.Module):
    pass

class Rd_siamese_classifier(nn.Module):
    def __init__(self):
        super(Rd_siamese_classifier, self).__init__()
        self.linear = nn.Linear(2,1024)
        self.batch_norm1 = nn.BatchNorm1d(1024)
        self.bilinear = nn.Bilinear(1024,1024,2)
        self.bilinear1 = nn.Bilinear(1024,1024,2)
    
    def forward(self, input):
        inp1 = input[:,0,:]
        inp2 = input[:,1,:]
        #repr1 = nn.Dropout(0.2)(self.batch_norm1(nn.ReLU()(self.linear(inp1))))
        #repr2 = nn.Dropout(0.2)(self.batch_norm1(nn.ReLU()(self.linear(inp2))))
        repr1 = nn.ReLU()(self.linear(inp1))
        repr2 = nn.ReLU()(self.linear(inp2))

        comp1 = self.bilinear(repr1,repr2)
        comp2= self.bilinear1(repr2,repr1)
        comp=torch.stack([comp1,comp2],dim=-1)
        return comp

class Rd_recurrent_classifier(nn.Module):
    def __init__(self):
        super(Rd_recurrent_classifier, self).__init__()
        self.linear = nn.Linear(2,1024)
        self.batch_norm1 = nn.BatchNorm1d(1024)
        self.rec = nn.LSTM(1024,100,bidirectional=True)
        self.linear1 = nn.Linear(100,2)
    
    def forward(self, input):
        inp1 = input[:,0,:]
        inp2 = input[:,1,:]
        #repr1 = nn.Dropout(0.2)(self.batch_norm1(nn.ReLU()(self.linear(inp1))))
        #repr2 = nn.Dropout(0.2)(self.batch_norm1(nn.ReLU()(self.linear(inp2))))
        repr1 = nn.ReLU()(self.linear(inp1))
        repr2 = nn.ReLU()(self.linear(inp2))

        output,hidden = self.rec(torch.stack([repr1,repr2]))
        h,c = hidden
        projected_hiddens =[]
        for i in range(h.shape[0]):
            projected_hiddens.append(self.linear1(h[i,:]))
        comp=torch.stack(projected_hiddens,dim=-1)
        return comp
      
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
'''
def generate_pairs_batch(data,domain,batch_size=128,is_y=True):
    
    def generate():        
        for _ in range(int(len(data)/batch_size)+1):
            # print("Returning batch ",torch.FloatTensor([[z[0][0],z[1][0]] for z in batch_data]),torch.FloatTensor([[z[0][1],z[1][1]] for z in batch_data]))
            batch_data = []
            for i in range(batch_size):
                # z1,z2 = next(perm_gen)
                z1=random.choice(data)
                z2=random.choice(data)
                batch_data.append((z1,z2))
            if len(batch_data) == 0:
                break
            if is_y:
                # print("Yielding",len(batch_data))
                yield (torch.FloatTensor([[z[0][0],z[1][0]] for z in batch_data]),torch.FloatTensor([[z[0][1],z[1][1]] for z in batch_data]))
            else:
                yield (torch.FloatTensor([[z[0][0],z[1][0]] for z in batch_data]))
    return generate
'''
def generate_pairs_batch(data,domain,batch_size=128,is_y=True):
    data_gen = (x for x in data)
    def generate():        
        batch_data = []
        from copy import copy
        data_gen = (x for x in data)
        for i in range(batch_size):
            # z1,z2 = next(perm_gen)
            z1=next(data_gen)
            z2=next(data_gen)
            batch_data.append((z1,z2))
            
        if is_y:
            # print("Yielding",len(batch_data))
            yield (torch.stack([torch.stack([z[0][0].squeeze(0),z[1][0].squeeze(0)]) for z in batch_data]),torch.stack([torch.stack([z[0][1].squeeze(0),z[1][1].squeeze(0)]) for z in batch_data]))
        else:
            yield (torch.stack([torch.stack([z[0][0].squeeze(0),z[1][0].squeeze(0)]) for z in batch_data]))
    return generate



def generate_batch(data,batch_size=128,is_y=True):
    data_gen = (x for x in data)
    def generate():
        while(True):
            batch_data = []
            for i in range(batch_size):
                batch_data.append(next(data_gen))
            if len(batch_data) < batch_size:
                break
            if is_y:
                yield (torch.stack([x[0].squeeze(0) for x in batch_data]),torch.stack([x[1].squeeze(0) for x in batch_data]))
            else:
                yield (torch.stack([x[0].squeeze(0) for x in batch_data]))
                
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
        #train_data_gen = generate_batch(self.train_data)
        #val_data_gen = generate_batch(val_gen)

        train_losses,val_losses = train_with_early_stopping(self.model,train_data_gen,val_data_gen,criterion,optimizer,num_epochs,tolerance=0.0001,max_epochs_without_improv=int(num_epochs*0.2),verbose=True,model_out=self.model_file,min_val=0.00001)
        print(np.mean(train_losses),np.mean(val_losses))
    
    def predict(self,test_data):
        outputs = self.model(torch.FloatTensor([x[0] for x in test_data]))
        return outputs

    def evaluate(self,test_data,criterion):
        predictions = self.predict(test_data)
        print(criterion(predictions.squeeze(1),Variable(torch.FloatTensor([x[1] for x in test_data]))))    
    
class RandomSamplePairFunctionApproximator(FunctionApproximator):
    def __init__(self,train_data,domain,diff_train_sample_size=10,eval_sample_size=10,differential_model=None,model_file=''):
        self.train_data = train_data
        self.domain = domain
        self.diff_train_sample_size = diff_train_sample_size
        self.eval_sample_size = eval_sample_size
        self.diff_train_data = self.train_data
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

def combine_predictions_average_axis(axis =None):
    def combine_predictions_average(predictions,weights=None):
        return np.average(predictions,weights=weights,axis=axis)
    return combine_predictions_average

def combine_predictions_vote(predictions,weights=None):
    predictions_index = [x.tolist().index(1) for x in predictions]
    max_vote_prediction = [0]*predictions[0].shape[0]
    max_index = Counter(predictions_index).most_common(1)
    max_vote_prediction[max_index] = 1
    return Tensor(max_vote_prediction)
    
    
class SamplePairCoApproximator(RandomSamplePairFunctionApproximator):
    def __init__t(self,*args,**kwargs):
        super().__init(*args,**kwargs)

    def approximate(self,val_gen,optimizer,criterion,num_epochs=100):
        diff_train_data_gen = generate_pairs_batch(self.train_data, self.domain,128)
        diff_val_data_gen = generate_pairs_batch(val_gen, self.domain)
        train_losses,val_losses = train_with_early_stopping(self.differential_model,diff_train_data_gen,diff_val_data_gen,criterion,optimizer,num_epochs,tolerance=0.0001,max_epochs_without_improv=int(0.2*num_epochs),verbose=True,model_out=self.model_file)
        print(np.mean(train_losses),np.mean(val_losses))
        #self.evaluate(val_gen)
    
    def predict(self,test_data,combination_function=combine_predictions_average_axis()):
        reference_data = random.sample(self.train_data,self.eval_sample_size)
        ref_x = [x[0] for x in reference_data]
        ref_y = [x[1] for x in reference_data]
        test_x = [x[0] for x in test_data]
        paired_with_reference = [x for x in product(test_x,ref_x)]
        
        #if GPU:
        #    paired_with_reference=torch.cuda.FloatTensor(paired_with_reference)
        
        #else:
        paired_with_reference=torch.FloatTensor(paired_with_reference)
        
        pair_outputs = self.differential_model(paired_with_reference)
        outputs = []
        for i in range(0,len(test_data)*len(ref_x),len(ref_x)):
            predictions = [x[0].detach().numpy() for x in pair_outputs[i:i+len(ref_x)]]
            predictions_reference = [x[1].detach() for x in pair_outputs[i:i+len(ref_x)] ]
            #weights = [1-abs((x-y)/y) for x,y in zip(predictions_reference,ref_y) ] 
            mean_prediction = combination_function(predictions,weights=None)
            outputs.append(mean_prediction)
        return outputs
    
    def evaluate(self,test_data,criterion,combination_function=combine_predictions_average_axis()):
        predictions = self.predict(test_data,combination_function)
        print(criterion(Variable(torch.FloatTensor(predictions).unsqueeze(1)),Variable(torch.FloatTensor([x[1] for x in test_data]).unsqueeze(1) )))
        
def plot_figure(inputs,predictions,outfile):
    pass

if __name__=='__main__':
    GPU=False
    siamese_model = load_pytorch_model('circle-copairs-class.model')
    siamese_model = siamese_model.cpu()
    single_model = load_pytorch_model("circle-single-class.model")
    single_model = single_model.cpu()
    samples = read_samples('circle-class.csv',classes=[0,1])
    ood_samples = read_samples('circle-class-ood.csv',classes=[0,1])
    #R_2 ={'num_dims':2,'bounds':[(-1,1),(-1,1)]}
    #domain = Bounded_Rd(R_2['num_dims'],R_2['bounds'])
    domain = BoundedRingDomain()
    approximator = SamplePairCoApproximator(samples,domain,differential_model=siamese_model)
    approximator_single = SingleSampleFunctionApproximator(samples,model=single_model)
    #criterion = nn.MSELoss()
    criterion_single = BCEWithLogitsLoss()
    criterion = MultiBCEWithLogitsLoss()
    combination_function = combine_predictions_average_axis(0)
    approximator.evaluate(ood_samples, criterion,combination_function)
    approximator_single.evaluate(ood_samples, criterion_single)
    samples = samples[:2500]
    siamese_predictions = approximator.predict(ood_samples,combination_function)
    single_predictions = approximator_single.predict(ood_samples)
    single_predictions = [x.tolist().index(max(x)) for x in single_predictions]
    siamese_predictions = [x.tolist().index(max(x)) for x in siamese_predictions]
    #plot_figure(ood_samples,siamese_predictions,'results-siamese.png')
    #plot_figure(ood_samples,single_predictions,'results-single.png')
    color_map ={0:'b',1:'g'}
    domain.visualize([x[0] for x in ood_samples], [color_map[x[1].index(1)] for x in ood_samples])
    domain.visualize([x[0] for x in ood_samples], [color_map[x] for x in single_predictions])
    domain.visualize([x[0] for x in ood_samples], [color_map[x] for x in siamese_predictions])


