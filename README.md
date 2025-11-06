# SGD — Teacher-Student Training Toolkit

Small toolkit to run teacher-student experiments and study learning dynamics
with stochastic gradient descent (SGD). This project contains data samplers
and a flexible `Trainer` that can run single- or multi-walker SGD on a variety
of synthetic data models.

## What it does

- Synthetic data generators (`Perceptron`, `RademacherCumulant`, `SkewedCumulant`).
- `DataGenerator` in `src/SGD/data.py` orchestrates online / repeated sampling and supports multiple walkers.
- `Trainer` in `src/SGD/dynamics.py` runs SGD, returns intermediate states and supports
	normalization / spherical projection and several activation/loss choices.

## Installation (Linux)

```bash
git clone https://github.com/CristopherErazo/Stochastic-Gradient-Descent.git SGD
cd "SGD"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## Quick examples

1) Minimal sampling:

```python
from SGD.data import SkewedCumulant
sampler = SkewedCumulant(dim=1000, spike=None, snr=5.0)
X, y, _ = sampler.sample(n=100)
```

2) Train a single-chain student:

```python
import numpy as np
from SGD.data import Perceptron, DataGenerator
from SGD.dynamics import Trainer

d = 200
u_spike = np.zeros(d); u_spike[0] = 1.0
perceptron = Perceptron(dim=d, w_teacher=u_spike, teacher='He3')
data_gen = DataGenerator(perceptron, N_walkers=None, mode='online')
trainer = Trainer(d, u_spike, student='relu', loss='corr', lr=0.01, data_generator=data_gen)

w0 = np.random.randn(d)
for step, (w_student, flag, grad) in enumerate(trainer.evolution(w0, N_steps=1000, progress=True)):
		if step % 100 == 0:
				print(step, np.linalg.norm(w_student))
```

## Project layout

- `src/SGD/` — modules: `dynamics.py`, `data.py`, `functions.py`, `utils.py`, etc.
- `notebooks/` — example notebooks and experiments.
- `requirements.txt`, `LICENSE`.

## Notes

- `Trainer.evolution` is a generator that yields `(w_student, flag, grad)` each step — it's convenient for streaming plots and logging.
- `DataGenerator` supports both online sampling and fixed-dataset replay modes; see its constructor for options used in the notebooks.



## License

See `LICENSE` in the repository root.

---


