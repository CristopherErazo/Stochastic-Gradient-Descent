from SGD.utils import make_params_dict , make_data_paths , download_cluster_data

# Parameters fixed
d = 100
alpha = 70.0
snr = 0.9
spike = True

# Parameter lists
modes = ['online','repeat']
losses = ['mse','corr']
students = ['relu','tanh']
lrs = [0.01 , 0.05 , 0.1 , 0.5]

# Parameters to save
names_fixed = ['d','snr','alpha']
names_variable = ['mode','loss','student','lr']

path_cluster = 'SGD_2.0/logs/'
for mode in modes:
    if mode == 'online':
        p_repeat = 0.0
        dataset_size = 0
    else:
        p_repeat = 1.0
        dataset_size = 10000
    for loss in losses:
        for student in students:
            for lr in lrs:
                filename_cluster = f"trajectories_spike{spike}_mode{mode}_d{d}_p{p_repeat}_alpha{alpha}_student{student}_loss{loss}_lr{lr}_snr{snr}_Ndataset_{dataset_size}.pkl"
                
                params = make_params_dict(names_fixed,names_variable)
                _ , filename_local , path_local = make_data_paths('overlap', experiment_name= 'test_spike', params=params,base_dir='./data')

                download_cluster_data('peralba',path_cluster,path_local,filename_cluster,filename_local,show=False)
    
