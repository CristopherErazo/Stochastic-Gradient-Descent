import numpy as np
from SGD.dynamics import Trainer
from SGD.data import DataGenerator

def main():
    # Define parameters
    d = 2000
    teacher = "tanh"
    student = "tanh"
    loss = "mse"
    lr = 2/d
    w_teacher = np.random.randn(d)
    w_teacher/=np.linalg.norm(w_teacher)
    w_initial = np.random.randn(d)
    w_initial/=np.linalg.norm(w_initial)
    noise = 0.0
    dataset_size = 1
    p_repeat = 0.2
    N_steps = 5000

    # Initialize DataGenerator
    data_generator = DataGenerator(d, teacher, w_teacher, noise=noise, dataset_size=dataset_size, p_repeat=p_repeat,mode='repeat')

    # Extract Data as arrays
    X , Y = data_generator.get_dataset(how='arrays')
    print(f"Data shape: X = {X.shape} , Y = {Y.shape}" )
    print(f'initial norms: ||w_teacher|| = {np.linalg.norm(w_teacher):.4f} , ||w_initial|| = {np.linalg.norm(w_initial):.4f}')
    print(f"overlap = {np.dot(w_teacher, w_initial):.4f}")

    # Initialize Trainer
    trainer = Trainer(d, w_teacher, teacher, student, loss, lr, data_generator)

    # Run evolution
    print("Starting training...")
    for step, (w_student, flag) in enumerate(trainer.evolution(w_initial, N_steps, progress=False)):
        if step % 100 == 0 or step == N_steps - 1:
            print(f"Step {step + 1}/{N_steps}: overlap = {np.dot(w_teacher, w_student):.4f} : flag = {flag} : preactivations = {X @ w_student} ")

    print("Training completed.")

if __name__ == "__main__":
    main()
