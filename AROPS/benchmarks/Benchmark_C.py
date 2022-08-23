# Benchmark C
from skopt.utils import normalize_dimensions
from skopt.space import Categorical, Space
import pandas as pd

# the design space
space = Space(normalize_dimensions([Categorical(('1a', '1b', '1c', '1d'), transform='onehot',
                                                name='reactant_1'),
                                    Categorical(('2a', '2b', '2c'), transform='onehot',
                                                name='reactant_2'),
                                    Categorical(('L1.0', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'L10', 'L11',
                                                 'L12'), transform='onehot', name='solvent'),
                                    Categorical(('B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8'), transform='onehot',
                                                name='base'),
                                    Categorical(('S1', 'S2', 'S3', 'S4'), transform='onehot',
                                                name='solvent')]))
goal = 'maximize'
t_index = 1.75  # the fixed reaction time, float
least_dist = 0.01  # the Euclidean distance threshold
data = pd.read_csv('data_Pfizer.csv')


# Reaction simulator
# input: condition vector, output: objective
def run_exp(con):
    y = float(data.loc[(data["Reactant_1"].str.contains(con[0])) & (data["Reactant_2"].str.contains(con[1])) & (
        data["Ligand"].str.contains(con[2])) & (data["Base"].str.contains(con[3])) & (
                           data["Solvent"].str.contains(con[4]))]['Yield'])
    return y
