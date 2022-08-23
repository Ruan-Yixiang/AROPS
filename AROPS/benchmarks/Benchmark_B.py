# Benchmark B
from olympus import Dataset
from olympus import Emulator
from skopt.space import Real, Space
from skopt.utils import normalize_dimensions
import warnings

warnings.filterwarnings('ignore')
emulator = Emulator(model='BayesNeuralNet', dataset='snar')
dataset = Dataset("snar")
space_list = []
for i in dataset.param_space:
    space_list.append(Real(i['low'], i['high'], name=i['name'], transform='normalize'))
space = Space(normalize_dimensions(space_list))  # the design space
goal = dataset.goal  # the optimization goal
t_index = 0  # the time index in the input vector
least_dist = 1e-3  # the Euclidean distance threshold

# Reaction simulator
# input: condition vector, output: objective
def run_exp(con):
    return emulator.run(con)[0][0]
