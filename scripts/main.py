import numpy as np
from SGD.dynamics import Trainer
from SGD.data import DataGenerator , SpikeGenerator
from SGD.utils import initialize_weights , save_data , make_params_dict
import argparse
import time

def main():
    # Define parameters
    parser = argparse.ArgumentParser(description="Train a student network with SGD.")
    parser.add_argument('--d', type=int, default=1000, help='Dimensionality of the data')
    parser.add_argument('--N_walkers', type=int, default=10, help='Number of parallel walkers (default is None for single chain)')
    parser.add_argument('--teacher', type=str, default='He4', help='Teacher activation function')
    parser.add_argument('--k',type=int,default=4,help='Information Exponent')
    parser.add_argument('--student', type=str, default='relu', help='Student activation function')
    parser.add_argument('--loss', type=str, default='mse', help='Loss function')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate in units of 1/d (default is 1/d)')
    parser.add_argument('--noise', type=float, default=0.0, help='Noise level')
    parser.add_argument('--dataset_size', type=float, default=0.0, help='Dataset size')
    parser.add_argument('--p_repeat', type=float, default=0.0, help='Probability of repeat')
    parser.add_argument('--alpha', type=float, default=1.0, help='Sample complexity (N_steps/d)')
    parser.add_argument('--snr',type=float,default=0.9,help='Signal to noise ratio in (0,1)')
    parser.add_argument('--f_spike',type=float,default=0.5,help='Fraction (probability) of spike during sampling')
    parser.add_argument('--mode',type=str,default='online',help='Sampling mode')
    parser.add_argument('--spike',type=str,default='False',help='Run with spike model or nn')
    parser.add_argument('--progress',type=str,default='False',help='Show progress bar or not')
    args = parser.parse_args()
    print("Parsed arguments:", args)
    d = args.d
    k = args.k
    N_walkers = args.N_walkers
    teacher = args.teacher
    student = args.student
    loss = args.loss
    lr = args.lr *d**(-0.5*k+1)#if args.lr is not None else 1/d
    noise = args.noise
    dataset_size = int(args.dataset_size * d**(2))
    p_repeat = args.p_repeat 
    alpha = args.alpha
    N_steps = int(alpha * d**(k-1))
    snr = args.snr
    f_spike = args.f_spike
    mode = args.mode
    spike = args.spike in ['True','1','yes']
    progress = args.progress in ['True','1','yes']


    # Initialize weights
    u_spike, w_initial = initialize_weights(d,N_walkers=N_walkers,m0=0.0,mode='fixed')

    # Initialize data generator
    if spike :
        data_generator = SpikeGenerator(d,u_spike,snr,f_spike=f_spike,N_walkers=N_walkers,dataset_size=dataset_size,p_repeat=p_repeat,mode=mode)
    else:
        data_generator = DataGenerator(d, teacher, u_spike, noise=noise, N_walkers=N_walkers, dataset_size=dataset_size, p_repeat=p_repeat,mode=mode)

    # Initialize Trainer
    trainer = Trainer(d, u_spike, student, loss, lr, data_generator,N_walkers=N_walkers)

    # Save data
    tprints = np.unique(np.logspace(-0.1,np.log10(N_steps),500).astype(int))
    data = {'overlap':[],'times':tprints}

    # Run evolution
    t0 = time.time()
    print(f"\nStarting training...")
    for step, (w_student, flag , grad) in enumerate(trainer.evolution(w_initial, N_steps, progress=progress)):
        if step in tprints: # Save some steps
            data['overlap'].append(w_student @ u_spike)
        
        condition_print = step in tprints[::20]
        if condition_print : # Print some steps
            print(f"Step {step + 1}/{N_steps} ...")

    dt = time.time() - t0
    print(f"End training... Took = {dt/60:.4} min\n")

    # Optional Steps
    for key in data:
        data[key] = np.array(data[key])    
    data['params'] = vars(args)

    # Save data
    # Fixed names will be set as subfolder name inside ../data/experiment_name/ folder
    # Variable names will go on the name of the file after file_name
    names_fixed = ['d','snr','alpha']
    names_variable = ['spike','p_repeat','student','loss']
    params = make_params_dict(names_fixed,names_variable)
    
    save_data(data,file_name='overlap',experiment_name='time_traces',params=params)

if __name__ == "__main__":
    main()
