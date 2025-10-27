import numpy as np
from SGD.dynamics import Trainer
from SGD.data import DataGenerator , Perceptron , RademacherCumulant, SkewedCumulant
from SGD.utils import initialize_weights , save_data , make_params_dict
import argparse
import time

def main():
    # Define parameters
    parser = argparse.ArgumentParser(description="Train a student network with SGD.")
    parser.add_argument('--d', type=int, default=100, help='Dimensionality of the data')
    parser.add_argument('--N_walkers', type=int, default=10, help='Number of parallel walkers (default is None for single chain)')
    parser.add_argument('--teacher', type=str, default='He3', help='Teacher activation function')
    parser.add_argument('--k',type=int,default=3,help='Information Exponent')
    parser.add_argument('--student', type=str, default='He3', help='Student activation function')
    parser.add_argument('--loss', type=str, default='corr', help='Loss function')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate in units of 1/d (default is 1/d)')
    parser.add_argument('--noise', type=float, default=0.0, help='Noise level')
    parser.add_argument('--alpha', type=float, default=1.0, help='Sample complexity (N_steps/d)')
    parser.add_argument('--snr',type=float,default=5,help='Signal to noise ratio')
    parser.add_argument('--mode',type=str,default='online',help='Sampling mode = online or repeat')
    parser.add_argument('--progress',type=str,default='False',help='Show progress bar or not')
    parser.add_argument('--model',type=str,default='perceptron',help='Data Model: perceptron, rademacher, skewed')
    parser.add_argument('--rho',type=float,default=0.7,help='Skewness parameter for skewed model')
    parser.add_argument('--datasize',type=float,default=1.0,help='Dataset size in units  of d (default is 1)')

    args = parser.parse_args()
    print("Parsed arguments:", args)

    # Extract parameters
    d = args.d
    k = args.k
    N_walkers = args.N_walkers
    teacher = args.teacher
    student = args.student
    loss = args.loss
    lr = args.lr
    noise = args.noise
    datasize = args.datasize
    dataset_size = int(datasize*d)
    p_repeat = 1.0
    alpha = args.alpha
    N_steps = int(alpha * d**(k-1) * (np.log(d))**2 )
    snr = args.snr
    mode = args.mode
    progress = args.progress in ['True','1','yes']
    model = args.model
    rho = args.rho

    # Define parameters to save

    # Fixed names will be set as subfolder name inside ../data/experiment_name/ folder
    # Variable names will go on the name of the file after file_name
    names_fixed = ['snr','alpha','teacher','loss','rho','N_walkers','datasize']
    names_variable = ['d','mode','lr','student','model']
    params = make_params_dict(names_fixed,names_variable)

    # Scale parameters
    lr = lr * d**(-0.5*k+1)

    # Initialize weights
    u_spike, w_initial = initialize_weights(d,N_walkers=N_walkers,m0=0.0,mode='random')


    # Initialize data model
    rng = np.random.default_rng()
    if model == 'perceptron':
        data_sampler = Perceptron(dim=d,w_teacher=u_spike,teacher=teacher,noise=noise,rng=rng)
    elif model == 'rademacher':
        data_sampler = RademacherCumulant(dim=d,spike=u_spike, snr=snr, rng=rng,p_spike=0.5)
    elif model == 'skewed':
        data_sampler = SkewedCumulant(dim=d,spike=u_spike, snr=snr, rng=rng, p_spike=0.5, rho=rho)
    else:
        raise ValueError(f"Unknown model type: {model}")
    
    # Initialize data generator
    data_generator = DataGenerator(data_sampler,N_walkers=N_walkers,mode=mode,dataset_size=dataset_size,p_repeat=p_repeat) 

    # Initialize Trainer
    trainer = Trainer(d, u_spike, student, loss, lr, data_generator,N_walkers=N_walkers)

    # Save data
    tprints = np.unique(np.logspace(-0.1,np.log10(N_steps),1000).astype(int))
    data = {'overlap':[],'times':[]}

    # Run evolution
    t0 = time.time()
    print("\nStarting training...")
    for step, (w_student, flag , grad) in enumerate(trainer.evolution(w_initial, N_steps, progress=progress)):
        if step in tprints: # Save some steps
            data['overlap'].append(w_student @ u_spike)
            data['times'].append(step)
        
        condition_print = step in tprints[::100]
        if condition_print : # Print some steps
            print(f"Step {step + 1}/{N_steps} ...")

    dt = time.time() - t0
    print(f"End training... Took = {dt/60:.4} min\n")

    # Optional Steps
    for key in data:
        data[key] = np.array(data[key])    
    data['params'] = vars(args)
    data['final_w'] = w_student

    # Save data    
    save_data(data,file_name='evolutions',experiment_name='time_traces',params=params)

if __name__ == "__main__":
    main()
