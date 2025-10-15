import pytest
import numpy as np
from SGD.dynamics import Trainer
from SGD.data import DataGenerator

def test_trainer_initialization():
    d = 10
    teacher = "tanh"
    student = "relu"
    loss = "mse"
    lr = 0.01
    w_teacher = np.random.rand(d)
    noise = 0.1
    dataset_size = 100
    p_repeat = 0.5

    # Initialize DataGenerator
    data_generator = DataGenerator(d, teacher, w_teacher, noise=noise, dataset_size=dataset_size, p_repeat=p_repeat)

    # Valid Trainer initialization
    trainer = Trainer(d, w_teacher, teacher, student, loss, lr, data_generator)

    assert trainer.d == d
    assert trainer.teacher_fun is not None
    assert trainer.student_fun is not None
    assert trainer.student_deriv is not None
    assert trainer.gradient is not None
    assert trainer.lr == lr
    assert trainer.normalize is True
    assert trainer.spherical is True
    assert trainer.sample_data == data_generator.generate

    # Invalid Trainer initialization cases
    with pytest.raises(ValueError):
        Trainer(d + 1, w_teacher, teacher, student, loss, lr, data_generator)
    with pytest.raises(ValueError):
        Trainer(d, "Invalid vector", teacher, student, loss, lr, data_generator)
    with pytest.raises(ValueError):
        Trainer(d, w_teacher, "invalid_teacher", student, loss, lr, data_generator)
    with pytest.raises(ValueError):
        Trainer(d, w_teacher, teacher, "invalid_student", loss, lr, data_generator)
    with pytest.raises(ValueError):
        Trainer(d, w_teacher, teacher, student, "invalid_loss", lr, data_generator)
    with pytest.raises(ValueError):
        Trainer(d, w_teacher, teacher, student, loss, -0.01, data_generator)

def test_trainer_evolution():
    d = 10
    teacher = "tanh"
    student = "relu"
    loss = "mse"
    lr = 0.01
    w_teacher = np.random.rand(d)
    w_initial = np.random.rand(d)
    noise = 0.1
    dataset_size = 100
    p_repeat = 0.5
    N_steps = 50

    # Initialize DataGenerator
    data_generator = DataGenerator(d, teacher, w_teacher, noise=noise, dataset_size=dataset_size, p_repeat=p_repeat)

    # Initialize Trainer
    trainer = Trainer(d, w_teacher, teacher, student, loss, lr, data_generator)

    # Run evolution
    weights = []
    for w_student in trainer.evolution(w_initial, N_steps, progress=False):
        weights.append(w_student)
        assert w_student.shape == (d,)

    assert len(weights) == N_steps
