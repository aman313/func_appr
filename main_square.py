'''
    Generate a sample for domain along with its regression value given a filter
    train an approximator on the sample
    test generalization for in an out sample validation sets 
'''

from utils import Bounded_Rd
from utils import sum_func
from utils import get_filter_region_in_Rd
from utils import create_sample_from_domain_with_filter_functions
from utils import read_samples
from approximator import SingleSampleFunctionApproximator
from approximator import Rd_difference_approximator
from utils import GPU
from approximator import Rd_siamese_approximator
from torch import nn 
from torch import optim
import random
sample_file ='square.csv'
ood_sample_file='square-ood.csv'
NUM_EPOCHS=20000

def generate_data():
    SAMPLE_SIZE=10000
    R_2 ={'num_dims':2,'bounds':[(-1,1),(-1,1)]}
    domain = Bounded_Rd(R_2['num_dims'],R_2['bounds'])
    filter_r_2 = {'num_dims':2,'bounds':[(0,1),(0,1)]}
    filter_domain = Bounded_Rd(filter_r_2['num_dims'],filter_r_2['bounds'])
    create_sample_from_domain_with_filter_functions(domain,[get_filter_region_in_Rd(filter_domain)],sum_func,SAMPLE_SIZE,sample_file)
    OOD_SAMPLE_SIZE=2500
    create_sample_from_domain_with_filter_functions(filter_domain,[],sum_func,OOD_SAMPLE_SIZE,ood_sample_file)

def approximate_function():
    samples = read_samples(sample_file)
    random.shuffle(samples)
    train_data = samples[:int(0.6*len(samples))]
    val_data = samples[int(0.6*len(samples)):int(0.8*len(samples))]
    test_data = samples[int(0.8*len(samples)):]
    model = Rd_difference_approximator()
    if GPU:
        differential_model = differential_model.cuda()
    R_2 ={'num_dims':2,'bounds':[(-1,1),(-1,1)]}
    domain = Bounded_Rd(R_2['num_dims'],R_2['bounds'])
    approximator = SingleSampleFunctionApproximator(train_data,model=model,model_file='square-single.model')
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.MSELoss()
    approximator.approximate(val_data, optimizer, criterion, NUM_EPOCHS)

if __name__ =='__main__':
    #approximate_function()
    SAMPLE_SIZE=1000
    sample_file ='square-5to10.csv'
    R_2 ={'num_dims':2,'bounds':[(5,10),(5,10)]}
    domain = Bounded_Rd(R_2['num_dims'],R_2['bounds'])
    create_sample_from_domain_with_filter_functions(domain,[],sum_func,SAMPLE_SIZE,sample_file)
