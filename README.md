# AROPS
AROPS: A Framework of Automated Reaction Optimization with Parallelized Scheduling
## Installation
### Installation Requirements
+ Python >= 3.7
+ PyTorch >= 1.10
+ gpytorch==1.8.1
+ botorch==0.6.0
+ scipy
+ scikit-optimize
### Manual install
You can do a manual install. For a basic install, run:
```bash
git clone https://github.com/pytorch/botorch.git
pip install -r AROPS_requirements.txt
```
## Getting Started
```python
# The optmization example of Case 1 of Benchmark A.
from arops import AROPS

# Create AROPS instance
simulator = AROPS(reactor_number=2, analysis_instrument_number=1, schedule='ARIA-PI',
                  benchmark='Benchmark_A1', analysis_time=10, pi_min=1e-4)

# Run the optimization process
res = simulator.run()

# Results output
print('Optimum: {:.4f}, \nTime: {:.2f} min, \nNumber of Experiments: {:d}'.format(res.opt_obj, res.time, res.n_exps))
print('Optimal conditions:', res.opt_con)
```
## License
AROPS is distributed under an MIT License.
