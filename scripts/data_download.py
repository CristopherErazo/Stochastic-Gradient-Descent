from SGD.utils import make_params_dict , make_data_paths , download_cluster_data
import os

# Parameters fixed
snr=5.0
alpha = 70.0
teacher = 'He3'
rho = 0.7
N_walkers = 10
loss = 'corr'

# Parameter lists
ds = [128 ,256 , 512, 1024]
modes = ['online','repeat']
lrs = [0.05 , 0.1 , 0.5 , 1.0]
students = ['He3','He2+He3']
models = ['perceptron','skewed']

# Parameters to save
names_fixed = ['snr','alpha','teacher','loss','rho','N_walkers']
names_variable = ['d','mode','lr','student','model']

for d in ds:
    for mode in modes:
        for lr in lrs:
            for student in students:
                for model in models:    
                    params = make_params_dict(names_fixed,names_variable)
                    _ , filename , path = make_data_paths('evolutions', experiment_name= 'time_traces', params=params,base_dir='')
                    path_cluster = os.path.join('SGD_2.0','data',path)
                    # Transform the path_cluster for a ubuntu cluster
                    path_cluster = path_cluster.replace('\\','/')
                    path_local = os.path.join('data',path)
                    # print(path_cluster,path_local)

                    download_cluster_data('peralba',path_cluster,path_local,filename,show=True)
    
