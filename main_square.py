'''
    Generate a sample for domain along with its regression value given a filter
    train an approximator on the sample
    test generalization for in an out sample validation sets 
'''

from utils import Bounded_Rd, BoundedContainerDomain, region_to_class_function
from utils import sum_func,square_sum_func,sine_sum_func
from utils import get_filter_region_in_Rd
from utils import create_sample_from_domain_with_filter_functions
from utils import read_samples
from approximator import SingleSampleFunctionApproximator,SamplePairCoApproximator,\
    Rd_symmetric_siamese_approximator, Rd_classifier, MultiBCEWithLogitsLoss,\
    Rd_siamese_classifier, Rd_recurrent_classifier
from approximator import Rd_difference_approximator
from utils import GPU
from approximator import Rd_siamese_approximator
from torch import nn 
from torch import optim
import random
sample_file ='square-class.csv'
ood_sample_file='square-class-ood.csv'
NUM_EPOCHS=20000

def generate_data():
    SAMPLE_SIZE=10000
    R_2 ={'num_dims':2,'bounds':[(-1,1),(-1,1)]}
    domain = Bounded_Rd(R_2['num_dims'],R_2['bounds'])
    filter_r_2 = {'num_dims':2,'bounds':[(0,5),(0,5)]}
    filter_domain = Bounded_Rd(filter_r_2['num_dims'],filter_r_2['bounds'])
    create_sample_from_domain_with_filter_functions(domain,[get_filter_region_in_Rd(filter_domain)],sum_func,SAMPLE_SIZE,sample_file)
    OOD_SAMPLE_SIZE=2500
    create_sample_from_domain_with_filter_functions(filter_domain,[],sum_func,OOD_SAMPLE_SIZE,ood_sample_file)

def generate_data2():
    SAMPLE_SIZE=10000
    R_2 ={'num_dims':2,'bounds':[(-1,1),(-1,1)]}
    domain = Bounded_Rd(R_2['num_dims'],R_2['bounds'])
    filter_r_2 = {'num_dims':2,'bounds':[(0,1),(0,1)]}
    filter_domain = Bounded_Rd(filter_r_2['num_dims'],filter_r_2['bounds'])
    create_sample_from_domain_with_filter_functions(domain,[get_filter_region_in_Rd(filter_domain)],square_sum_func,SAMPLE_SIZE,sample_file)
    OOD_SAMPLE_SIZE=2500
    create_sample_from_domain_with_filter_functions(filter_domain,[],square_sum_func,OOD_SAMPLE_SIZE,ood_sample_file)


def generate_data3():
    SAMPLE_SIZE=10000
    R_2_POS =[{'num_dims':2,'bounds':[(-0.25,0.25),(-0.25,0.25)]},{'num_dims':2,'bounds':[(-1.0,-0.75),(-1.0,1.0)]},
              {'num_dims':2,'bounds':[(-0.75,0.75),(0.75,1.0)]},
              {'num_dims':2,'bounds':[(0.75,1.0),(-1.0,1,.0)]},
              {'num_dims':2,'bounds':[(-0.75,0.75),(-1.0,-0.75)]}]
    R_2_NEG = [{'num_dims':2,'bounds':[(-0.75,0.75),(0.25,0.75)]},
               {'num_dims':2,'bounds':[(-0.75,-0.25),(-0.25,0.25)]},
               {'num_dims':2,'bounds':[(0.25,0.75),(-0.25,0.25)]},
               {'num_dims':2,'bounds':[(-0.75,0.75),(-0.75,-0.25)]}]
    R_2_POS_Domain = [Bounded_Rd(x['num_dims'],x['bounds']) for x in R_2_POS]
    R_2_NEG_Domain = [Bounded_Rd(x['num_dims'],x['bounds']) for x in R_2_NEG]
    
    region_to_class_map = {x:1 for x in R_2_POS_Domain}
    region_to_class_map.update({x:0 for x in R_2_NEG_Domain})
    domain = BoundedContainerDomain(R_2_NEG_Domain + R_2_POS_Domain)
    filter_r_2 = {'num_dims':2,'bounds':[(0,1),(0,1)]}
    filter_domain = Bounded_Rd(filter_r_2['num_dims'],filter_r_2['bounds'])
    create_sample_from_domain_with_filter_functions(domain,[get_filter_region_in_Rd(filter_domain)],region_to_class_function(region_to_class_map),SAMPLE_SIZE,sample_file)
    OOD_SAMPLE_SIZE=2500
    create_sample_from_domain_with_filter_functions(filter_domain,[],region_to_class_function(region_to_class_map),OOD_SAMPLE_SIZE,ood_sample_file)

def learn_to_approximate_function_using_copairs(model_file='square-siamese.model',reload=False,reloadName=None):
    samples = read_samples(sample_file)
    random.shuffle(samples)
    train_data = samples[:int(0.6*len(samples))]
    val_data = samples[int(0.6*len(samples)):int(0.8*len(samples))]
    test_data = samples[int(0.8*len(samples)):]
    model = Rd_siamese_approximator()
    if GPU:
        model = model.cuda()
    R_2 ={'num_dims':2,'bounds':[(-1,1),(-1,1)]}
    domain = Bounded_Rd(R_2['num_dims'],R_2['bounds'])
    approximator = SamplePairCoApproximator(train_data,differential_model=model,model_file=model_file,domain=domain)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.MSELoss()
    approximator.approximate(val_data, optimizer, criterion, NUM_EPOCHS)

def learn_to_approximate_function_using_copairs_symmetric(model_file='square-siamese-symmetric.model',reload=False,reloadName=None):
    samples = read_samples(sample_file)
    random.shuffle(samples)
    train_data = samples[:int(0.6*len(samples))]
    val_data = samples[int(0.6*len(samples)):int(0.8*len(samples))]
    test_data = samples[int(0.8*len(samples)):]
    model = Rd_symmetric_siamese_approximator()
    if GPU:
        model = model.cuda()
    R_2 ={'num_dims':2,'bounds':[(-1,1),(-1,1)]}
    domain = Bounded_Rd(R_2['num_dims'],R_2['bounds'])
    approximator = SamplePairCoApproximator(train_data,differential_model=model,model_file=model_file,domain=domain)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.MSELoss()
    approximator.approximate(val_data, optimizer, criterion, NUM_EPOCHS)
def learn_to_approximate_function_using_single(model_file='square-single.model',reload=False,reloadName=None):
    samples = read_samples(sample_file)
    random.shuffle(samples)
    train_data = samples[:int(0.6*len(samples))]
    val_data = samples[int(0.6*len(samples)):int(0.8*len(samples))]
    test_data = samples[int(0.8*len(samples)):]
    model = Rd_difference_approximator()
    if GPU:
        model = model.cuda()
    R_2 ={'num_dims':2,'bounds':[(-1,1),(-1,1)]}
    domain = Bounded_Rd(R_2['num_dims'],R_2['bounds'])
    approximator = SingleSampleFunctionApproximator(train_data,model=model,model_file=model_file)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.MSELoss()
    approximator.approximate(val_data, optimizer, criterion, NUM_EPOCHS)
    
def learn_to_classify_using_single(model_file='square-single-class.model',reload=False,reloadName=False):
    samples = read_samples(sample_file,classes=[0,1])
    random.shuffle(samples)
    train_data = samples[:int(0.6*len(samples))]
    val_data = samples[int(0.6*len(samples)):int(0.8*len(samples))]
    test_data = samples[int(0.8*len(samples)):]
    model = Rd_classifier()
    if GPU:
        model = model.cuda()
    R_2 ={'num_dims':2,'bounds':[(-1,1),(-1,1)]}
    domain = Bounded_Rd(R_2['num_dims'],R_2['bounds'])
    approximator = SingleSampleFunctionApproximator(train_data,model=model,model_file=model_file)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.BCEWithLogitsLoss()
    approximator.approximate(val_data, optimizer, criterion, NUM_EPOCHS)

def learn_to_classify_using_copairs(model_file='square-copairs-class.model',reload=False,reloadName=False):
    samples = read_samples(sample_file,classes=[0,1])
    
    random.shuffle(samples)
    train_data = samples[:int(0.6*len(samples))]
    val_data = samples[int(0.6*len(samples)):int(0.8*len(samples))]
    test_data = samples[int(0.8*len(samples)):]
    model = Rd_recurrent_classifier()
    if GPU:
        model = model.cuda()
    R_2 ={'num_dims':2,'bounds':[(-1,1),(-1,1)]}
    domain = Bounded_Rd(R_2['num_dims'],R_2['bounds'])
    color_map ={0:'b',1:'g'}
    #domain.visualize([x[0] for x in samples], [color_map[x[1].index(1)] for x in samples])
    #exit()
    approximator = SamplePairCoApproximator(train_data,differential_model=model,model_file=model_file,domain=domain)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    criterion = MultiBCEWithLogitsLoss()
    approximator.approximate(val_data, optimizer, criterion, NUM_EPOCHS)

if __name__ =='__main__':   
    #generate_data3()
    #print("Generated Data")
    learn_to_classify_using_single()
    #learn_to_classify_using_copairs()
    #learn_to_approximate_function_using_copairs("sine-sum-siamese.model")
    #print("Learnt square-siamese model")
    #learn_to_approximate_function_using_copairs_symmetric("sine-sum-siamese-symmetric-scaled-params-count.model")
    #print("Learnt square siamese symmetric model")
    #learn_to_approximate_function_using_single("sine-sum-single.model")
    #print("Learnt square single model")
    # for i in range(10):
    #     learn_to_approximate_function_using_single("square-single.model"+"_"+str(i))
    # for i in range(10):
    #     learn_to_approximate_function_using_copairs("square-siamese.model"+"_"+str(i))
    # for i in range(10):
    #     learn_to_approximate_function_using_copairs_symmetric("square-siamese-symmetric.model"+"_"+str(i))
    
    #SAMPLE_SIZE=1000
    #sample_file ='square-5to10.csv'
    #R_2 ={'num_dims':2,'bounds':[(5,10),(5,10)]}
    #domain = Bounded_Rd(R_2['num_dims'],R_2['bounds'])
    #create_sample_from_domain_with_filter_functions(domain,[],sum_func,SAMPLE_SIZE,sample_file)
