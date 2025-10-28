from SGD.utils import make_params_dict , make_data_paths , download_cluster_data
import os

# Parameters fixed
alpha = 50.0
teacher = 'He3'
N_walkers = 20
loss = 'corr'
model = 'perceptron'
mode = 'online'

# Parameter lists
ds = [250,500,1000,2000]
lr_values = [0.01 , 0.05  , 0.1]
students = ['He3','relu']
variations = ['None','twice']

# Parameters to save
names_fixed = ['alpha','teacher','loss','N_walkers','model','mode']
names_variable = ['d','lr','student','variation']

for d in ds:
    for lr in lr_values:
        for student in students:
            for variation in variations:
                params = make_params_dict(names_fixed,names_variable)

                _ , filename , path = make_data_paths('evolutions', experiment_name= 'repetita_iuvant_check', params=params,base_dir='')
                path_cluster = os.path.join('SGD/data',path)
                path_local = os.path.join('data',path)

                download_cluster_data('peralba',path_cluster,path_local,filename,show=True)
