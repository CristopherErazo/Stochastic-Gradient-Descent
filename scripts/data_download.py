from SGD.utils import make_params_dict , make_data_paths , download_cluster_data
import os

# Parameters fixed
alpha = 100.0
teacher = 'He3'
N_walkers = 20
loss = 'corr'
model = 'skewed'
lrs = [0.01 , 0.05 , 0.1]
students  = ['relu','He3']
snr = 5.0
rho = 0.7
# Parameter lists
ds = [100,200,400,800]
combinations = [('None','online'),('twice','online'),('None','repeat')]


# Parameters to save
names_fixed = ['alpha','teacher','loss','N_walkers','model','mode']
names_variable = ['d','lr','student','variation','snr','rho']
for student in students:
    for lr in lrs:
        for d in ds:
            for variation , mode in combinations:
                params = make_params_dict(names_fixed,names_variable)

                _ , filename , path = make_data_paths('evolutions_spike', experiment_name= 'repetita_iuvant_check', params=params,base_dir='./',create_dir=False)
                path_cluster = os.path.join('SGD/data',path)
                path_local = os.path.join('./data',path)
                # print(path)

                download_cluster_data('peralba',path_cluster,path_local,filename,show=True)
                print('-----------------------------------')