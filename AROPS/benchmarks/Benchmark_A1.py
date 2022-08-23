# Case 1 of Benchmark A
import numpy as np
from scipy.constants import R
from scipy.integrate import odeint
from skopt.utils import normalize_dimensions
from skopt.space import Real, Categorical, Space

ncv = 3  # number of continuous variable
ndv = 1  # number of categorical variable
nparam = ncv + ndv
ca0 = 0.167  # M
cb0 = 0.250  # M
ar = 3.1e7  # L^(1/2) mol^(−3/2) s^(−1)
ear = 55  # kJ/mol
eas1 = 100  # kJ/mol
as1 = 1e12  # s^(-1)#case 3
eas2 = 50  # kJ/mol
as2 = 3.1e5  # L^(1/2) mol^(−3/2) s^(−1)#case 4
eai = {'a': 0.7, 'b': 0.4, 'c': 0.3, 'd': 0.7, 'e': 0.0, 'f': 2.2, 'g': 3.8, 'h': 7.3}  # kJ/mol
# the design space
space = Space(normalize_dimensions([Real(30, 110, name='T', transform='normalize'),
                                    Real(10, 100, name='tres', prior='log-uniform', transform='normalize'),
                                    Real(0.835, 4.175, name='ccat', prior='log-uniform', transform='normalize'),
                                    Categorical(('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'), transform='onehot',
                                                name='catalyst')]))
goal = 'maximize'  # the optimization goal
t_index = 1  # the time index in the input vector
least_dist = 1e-3  # the Euclidean distance threshold


# Reaction simulator
# input: condition vector, output: objective
def run_exp(con):
    T = con[0]
    tre = con[1]
    ccat = con[2]
    # change units
    T = T + 273.15
    tre = tre * 60
    ccat = ccat / 1000
    ea = eai[con[3]]
    kr = ccat ** 0.5 * ar * np.exp(-(ear + ea) / (T * R / 1000))

    ks2 = as2 * np.exp(-eas2 / (T * R / 1000))  # case 1

    # /1000:R J to kJ; kr:mol^(-1)s^(-1)
    def reaction(w, time):
        a, b, c, d = w
        f1 = -kr * a * b
        f2 = -kr * a * b - ks2 * b * c
        f3 = kr * a * b - ks2 * b * c
        f4 = ks2 * b * c  # case 1
        return [f1, f2, f3, f4]

    tre = tre / 10
    time = np.arange(0, tre, 0.001)
    re = odeint(reaction, (ca0, cb0, 0.0, 0.0), time)
    cr = re[-1, :][2]
    y = cr / ca0  # Reaction product yield R
    return y
