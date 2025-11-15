import numpy as np
from SGD.dynamics import Trainer
from SGD.data import DataGenerator , Perceptron , RademacherCumulant, SkewedCumulant
from SGD.optimizers import SingleStep, TwiceStep, ExtraGradient
from SGD.utils import initialize_weights , save_data , make_params_dict , save_fig
import argparse
import time
import matplotlib.pyplot as plt
from SGD.plot_config import set_font_sizes , create_fig , apply_general_styles 

def main():
    # Define parameters
    parser = argparse.ArgumentParser(description="Train a student network with SGD.")
    parser.add_argument('--d', type=int, default=100, help='Dimensionality of the data')
    parser.add_argument('--N_walkers', type=int, default=10, help='Number of parallel walkers (default is None for single chain)')
    parser.add_argument('--teacher', type=str, default='He3', help='Teacher activation function')
    parser.add_argument('--k',type=int,default=3,help='Information Exponent')
    parser.add_argument('--student', type=str, default='relu', help='Student activation function')
    parser.add_argument('--loss', type=str, default='corr', help='Loss function')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate in units of 1/d (default is 1/d)')
    parser.add_argument('--dataset_size', type=float, default=0.0, help='Dataset size')
    parser.add_argument('--p_repeat', type=float, default=0.0, help='Probability of repeat')
    parser.add_argument('--alpha', type=float, default=100.0, help='Sample complexity (N_steps/d)')
    parser.add_argument('--mode',type=str,default='online',help='Sampling mode = online or repeat')
    parser.add_argument('--progress',type=str,default='True',help='Show progress bar or not')
    parser.add_argument('--model',type=str,default='skewed',help='Data Model: perceptron , rademacher or skewed')
    parser.add_argument('--optimizer_type',type=str,default='extragradient',help='Optimizer: singlestep, extragradient, twicestep')
    parser.add_argument('--rho',type=float,default=0.1,help='Learning rate for extragradient step')

    args = parser.parse_args()

    print("Parsed arguments:", args)
    # Argument extraction
    d = args.d
    k = args.k
    N_walkers = args.N_walkers
    teacher = args.teacher
    student = args.student
    loss = args.loss
    lr = args.lr 
    dataset_size = args.dataset_size
    p_repeat = args.p_repeat 
    alpha = args.alpha
    mode = args.mode
    progress = args.progress in ['True','1','yes']
    model = args.model
    optimizer_type = args.optimizer_type
    rho = args.rho

    # Fixed names will be set as subfolder name inside ../data/experiment_name/ folder
    # Variable names will go on the name of the file after file_name
    names_fixed = ['alpha','loss','N_walkers','model','mode']
    names_variable = ['d','lr','student','optimizer_type']
    params = make_params_dict(names_fixed,names_variable)

    # Initialize weights
    u_spike, w_initial = initialize_weights(d,N_walkers=N_walkers,m0=0.0,mode='fixed')

    # Initialize data model
    rng = np.random.default_rng()
    if model == 'perceptron':
        data_sampler = Perceptron(dim=d,w_teacher=u_spike,teacher=teacher,rng=rng)
    elif model == 'rademacher':
        data_sampler = RademacherCumulant(dim=d,spike=u_spike, rng=rng)
    elif model == 'skewed':
        data_sampler = SkewedCumulant(dim=d,spike=u_spike, rng=rng)
    else:
        raise ValueError(f"Unknown model type: {model}")
    
    # Initialize data generator
    data_generator = DataGenerator(data_sampler,N_walkers=N_walkers,mode=mode,dataset_size=dataset_size,p_repeat=p_repeat) 

    # Initialize optimizer
    if optimizer_type == 'singlestep':
        optimizer = SingleStep(data_generator, loss, student, lr)
    elif optimizer_type == 'twicestep':
        optimizer = TwiceStep(data_generator, loss, student, lr)
    elif optimizer_type == 'extragradient':
        optimizer = ExtraGradient(data_generator, loss, student, lr, rho)
    else:   
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    # Initialize Trainer
    trainer = Trainer(d,u_spike,optimizer,N_walkers=N_walkers)
        

    # Save data
    N_steps = int(alpha * d)
    # tprints = np.unique(np.logspace(-0.1,np.log10(N_steps),500).astype(int))
    tprints = np.arange(0,N_steps, max(1,N_steps//500))
    data = {'overlap':[],'times':[]}

    # Run evolution
    t0 = time.time()
    print("\nStarting training...")
    for step, (w_student, flag , grad) in enumerate(trainer.evolution(w_initial, N_steps, progress=progress)):
        if step in tprints: # Save some steps
            data['overlap'].append(w_student @ u_spike)
            data['times'].append(step)
        
        # condition_print = step in tprints[::20]
        condition_print = False
        if condition_print : # Print some steps
            print(f"Step {step + 1}/{N_steps} with overlap {data['overlap'][-1]}...")

    dt = time.time() - t0
    print(f"End training... Took = {dt/60:.4} min\n")

    # Optional Steps
    for key in data:
        data[key] = np.array(data[key])    
    data['params'] = vars(args)

    # Save data    
    save_data(data,file_name='evolutions',experiment_name='repetita_iuvant_check',params=params)


    # Plot overlap evolution
    apply_general_styles()
    set_font_sizes('tight')
    fig , ax = create_fig()
    xtk = [d**i for i in range(4)]
    xlb = [rf'$d^{i}$' for i in range(4)]
    colors = plt.get_cmap('Blues')(np.linspace(0.2,1.0,N_walkers))
    for w in range(N_walkers):
        ax.plot(data['times']/d,data['overlap'][:,w],lw=0.8,color=colors[w])
    ax.axhline(-1/np.sqrt(d),ls='--',color='k',lw=1.0)
    # ax.set_xscale('log')
    # ax.set_xticks(xtk,xlb)
    fig.suptitle(rf'$d = {d}$, $\eta_0 = {args.lr:.2}$, $\alpha = {alpha}$ $\sigma = {student}$'+f'\n Model: {model}, Optimizer: {optimizer_type}')

    names = ['d','mode','model','optimizer_type','student']
    params = make_params_dict(names)
    save_fig(fig,'_test_overlap',params=params,base_dir='./logs')


if __name__ == "__main__":
    main()
