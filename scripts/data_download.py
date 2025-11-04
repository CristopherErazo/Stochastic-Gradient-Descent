from SGD.utils import make_params_dict , make_data_paths , download_cluster_data
import os

# Parameters fixed
alpha = 100.0
teacher = 'He3'
N_walkers = 40
loss = 'corr'
model = 'skewed'
lrs = [0.01 , 0.05 , 0.1]
student  = 'relu'
snr = 5.0
rho = 0.7
# Parameter lists
ds = [500,1000,2000]
combinations = [('None','online'),('twice','online'),('None','repeat')]

# # lr_values = [0.1 , 0.2  , 0.4]
# variations = ['None','twice']

# Parameters to save
names_fixed = ['alpha','teacher','loss','N_walkers','model','mode']
names_variable = ['d','lr','student','variation','snr','rho']
for lr in lrs:
    for d in ds:
        for variation , mode in combinations:
            params = make_params_dict(names_fixed,names_variable)

            _ , filename , path = make_data_paths('evolutions_spike', experiment_name= 'repetita_iuvant_check', params=params,base_dir='./',create_dir=False)
            path_cluster = os.path.join('SGD/data',path)
            path_local = os.path.join('./data',path)
            # print(path)

            download_cluster_data('peralba',path_cluster,path_local,filename,show=True)
