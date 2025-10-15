import pytest
import numpy as np
from SGD.data import DataGenerator

def test_data_generator_validation():
    d = 5
    teacher_fun = "tanh"
    w_teacher = np.random.normal(size=d)
    noise = 0.1

    data_gen = DataGenerator(d, teacher_fun, w_teacher, noise=noise)
    x, y = data_gen.generate_gaussian()

    assert x.shape == (d,), "Input x should have shape (d,)"
    assert isinstance(y, float), "Output y should be a float"

    # Test invalid w_teacher (wrong type)
    with pytest.raises(ValueError):
        DataGenerator(d, teacher_fun, "invalid_w_teacher", noise=noise)

    # Test invalid w_teacher (wrong shape)
    with pytest.raises(ValueError):
        DataGenerator(d, teacher_fun, np.random.normal(size=d + 1), noise=noise)

    # Test dataset_size and p_repeat mismatch
    with pytest.raises(ValueError):
        DataGenerator(d, teacher_fun, w_teacher, noise=noise, dataset_size=10, p_repeat=None, mode='repeat')

    with pytest.raises(ValueError):
        DataGenerator(d, teacher_fun, w_teacher, noise=noise, dataset_size=None, p_repeat=0.5, mode='repeat')

    # Test invalid p_repeat value
    with pytest.raises(ValueError):
        DataGenerator(d, teacher_fun, w_teacher, noise=noise, dataset_size=10, p_repeat=1.5, mode='repeat')

    with pytest.raises(ValueError):
        DataGenerator(d, teacher_fun, w_teacher, noise=noise, dataset_size=10, p_repeat=-0.1, mode='repeat')


    # Test invalid mode
    with pytest.raises(ValueError):
        DataGenerator(d, teacher_fun, w_teacher, noise=noise, dataset_size=10, p_repeat=0.5, mode='invalid_mode')
    # Test missing dataset_size and p_repeat in 'repeat' mode
    with pytest.raises(ValueError):
        DataGenerator(d, teacher_fun, w_teacher, noise=noise, mode='repeat')



def test_generate_batch():
    d = 5
    N = 10
    w_teacher = np.random.normal(size=d)
    teacher_fun = "relu"
    noise = 0.1

    data_gen = DataGenerator(d, teacher_fun, w_teacher, noise=noise)
    dataset = data_gen.generate_batch(N)

    assert len(dataset) == N, "Batch size should be N"
    for x, y in dataset:
        assert x.shape == (d,), "Each input x should have shape (d,)"
        assert isinstance(y, float), "Each output y should be a float"


def test_generate_repeating():
    d = 5
    dataset_size = 20
    p_repeat = 0.7
    w_teacher = np.random.normal(size=d)
    teacher_fun = "He2"
    noise = 0.1

    data_gen = DataGenerator(d, teacher_fun, w_teacher, noise=noise, dataset_size=dataset_size, p_repeat=p_repeat, mode='repeat')
    x, y = data_gen.generate_repeating()

    assert x.shape == (d,), "Input x should have shape (d,)"
    assert isinstance(y, float), "Output y should be a float"

    # Check if the dataset was created correctly
    assert len(data_gen.dataset) == dataset_size, "Fixed dataset size should match dataset_size"
    for x_fixed, y_fixed in data_gen.dataset:
        assert x_fixed.shape == (d,), "Each fixed input x should have shape (d,)"
        assert isinstance(y_fixed, float), "Each fixed output y should be a float"