import numpy as np
from SGD.dynamics import Trainer
from SGD.data import DataGenerator
from SGD.utils import initialize_weights
import argparse
import pickle

def main():
    # Define parameters
    parser = argparse.ArgumentParser(description="Train a student network with SGD.")
    parser.add_argument('--d', type=int, default=10000, help='Dimensionality of the data')
    parser.add_argument('--N_walkers', type=int, default=10, help='Number of parallel walkers (default is None for single chain)')
    parser.add_argument('--teacher', type=str, default='He3', help='Teacher activation function')
    parser.add_argument('--k',type=int,default=3,help='Information Exponent')
    parser.add_argument('--k0',type=float,default=3,help='Interpolation exponent')
    parser.add_argument('--student', type=str, default='relu', help='Student activation function')
    parser.add_argument('--loss', type=str, default='mse', help='Loss function')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate in units of 1/d (default is 1/d)')
    parser.add_argument('--noise', type=float, default=0.0, help='Noise level')
    parser.add_argument('--dataset_size', type=int, default=1, help='Dataset size')
    parser.add_argument('--p_repeat', type=float, default=0.0, help='Probability of repeat')
    parser.add_argument('--alpha', type=float, default=1.0, help='Sample complexity (N_steps/d)')
    args = parser.parse_args()
    print("Parsed arguments:", args)
    d = args.d
    k = args.k
    k0 = args.k0
    N_walkers = args.N_walkers
    teacher = args.teacher
    student = args.student
    loss = args.loss
    lr = args.lr *d**(-0.5*k0+1)#if args.lr is not None else 1/d
    noise = args.noise
    dataset_size = args.dataset_size
    p_repeat = args.p_repeat * d**(-0.5*(k0-1))
    alpha = args.alpha
    N_steps = int(alpha * d)

    # Initialize weights
    w_teacher, w_initial = initialize_weights(d, N_walkers=N_walkers, m0=-1/np.sqrt(d), mode='fixed')

    # Initialize DataGenerator
    data_generator = DataGenerator(d, teacher, w_teacher,noise=noise, N_walkers=N_walkers, dataset_size=dataset_size, p_repeat=p_repeat,mode='repeat')

    # Extract Data as arrays
    X , Y = data_generator.get_dataset(how='arrays')
    data_init = (X[0],Y[0]) if N_walkers is None or N_walkers == 1 else (X[:,0],Y[:,0])
    # print(f"Data shape: X = {X.shape} , Y = {Y.shape}" )
    # print(f'initial norms: ||w_teacher|| = {np.linalg.norm(w_teacher):.4f} , ||w_initial|| = {np.linalg.norm(w_initial,axis=-1,keepdims=True)}')
    print(f"initial overlaps = {w_initial @ w_teacher }")

    # Initialize Trainer
    trainer = Trainer(d, w_teacher, teacher, student, loss, lr, data_generator,N_walkers=N_walkers)

    # Save data for plotting
    results = {
        'overlaps':[],
        'preactivations':[],
        'flags':[],
        'times':[],
        'grad':[],
    }
    # Run evolution
    tprints = np.unique(np.logspace(0,np.log10(N_steps),500).astype(int))
    print("Starting training...")
    for step, (w_student, flag , grad) in enumerate(trainer.evolution(w_initial, N_steps, progress=True,data_init=None)):
        condition_print = step % 1 == 0 or step == N_steps - 1
        condition_print = False
        if condition_print:
            print(f"Step {step + 1}/{N_steps}: {w_student.shape = }, {grad.shape = }, overlap = {w_student @ w_teacher} : flag = {flag} ")
        
        condition_save = step in tprints or step == N_steps - 1 or flag.sum()>0 or step == 0
        if condition_save:
            results['overlaps'].append(w_student @ w_teacher)
            results['preactivations'].append(np.sum(data_init[0] * w_student,axis=-1))
            results['flags'].append(flag)
            results['times'].append(step)
            results['grad'].append(np.linalg.norm(grad,axis=-1))            
    print("Training completed.")


        
    for key in results:
        results[key] = np.array(results[key])
        print(f"{key} shape: {results[key].shape}")
    
    results['params'] = vars(args)
    print("Parameters:", results['params'])
    print(f'{results['flags'].sum(axis=0) =}')
    # Save results using pickle
   
    filename = f"logs/test_d{d}_alpha{alpha}_p{args.p_repeat}_lr{args.lr}_student{student}_teacher{teacher}_k{k}_loss{loss}_k0{k0}.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to {filename}")

if __name__ == "__main__":
    main()
