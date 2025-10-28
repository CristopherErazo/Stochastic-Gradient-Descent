from SGD.utils import make_params_dict , make_data_paths , download_cluster_data
import os

# Parameters fixed
snr=5.0
alpha = 2.0
teacher = 'He3'
rho = 0.7
N_walkers = 10
loss = 'corr'
lr = 0.2
student = 'He3'
datasize = 5.0 

# Parameter lists
ds = [100,200,400,800]
modes = ['online','repeat']
models = ['perceptron','skewed']

# Parameters to save

names_fixed = ['snr','alpha','teacher','loss','rho','N_walkers','datasize']
names_variable = ['d','mode','lr','student','model']


for d in ds:
    for mode in modes:
        for model in models:    
            params = make_params_dict(names_fixed,names_variable)
            _ , filename , path = make_data_paths('evolutions', experiment_name= 'time_traces', params=params,base_dir='')
            path_cluster = os.path.join('SGD/data',path)
            path_local = os.path.join('data',path)

            download_cluster_data('peralba',path_cluster,path_local,filename,show=True)

