import torch
import numpy as np
import warnings
from transformer import transform, i_transform
from RA_model import work_flow, create_equipment, reaction_update, update_eqiupment
from Bayesian_Optimization import get_next_exeps, ProbabilityOfImprovementXi
from botorch.acquisition import ExpectedImprovement

warnings.filterwarnings('ignore')


# The class of results
class Results:
    def __init__(self, results: list):
        self.opt_obj = results[0]  # the optimal objective, float
        self.opt_con = results[1]  # the optimal conditions, list[list]
        self.time = results[2]  # the required time in optimization process (min), float
        self.n_exps = results[3]  # the required number of experiments in optimization process, int
        self.r_operating_time_ratio = results[4]  # the average operating time ratio of the reactors, float
        self.a_operating_time_ratio = results[5]  # the average operating time ratio of the analyzers, float
        self.x = results[6]  # the conditions of screened experiments
        self.obj = results[7]  # the objectives of screened experiments


class AROPS:
    def __init__(self, reactor_number: int, analysis_instrument_number: int, schedule: str, benchmark: str,
                 analysis_time: float, max_experiments: int = 200, pi_min: float = 0.05, max_break: int = 3,
                 rand_seed=None):
        r"""
        :param reactor_number: the number of reactors, Int
        :param analysis_instrument_number: the number of analyzers, Int
        :param schedule: the scheduling modes, Str
        :param benchmark: the benchmark, Str
        :param analysis_time: the analysis time (min), Float
        :param max_experiments: the maximum number of experimental trials, Int
        :param pi_min: the PI threshold, Float
        :param max_break: the number of proposed unpromising experiments reach a threshold number, Int
        :param rand_seed: the random seed, Int or None
        """
        if reactor_number > 0:
            self.reactor_number = reactor_number
        else:
            raise ValueError('Number of reactors must be a positive integer.')
        if analysis_instrument_number > 0:
            self.analysis_instrument_number = analysis_instrument_number
        else:
            raise ValueError('Number of analysis instruments must be a positive integer.')
        if schedule in ['ARIA', 'ARIA-PI', 'SRIA', 'SRIA-PI', 'SRBA',
                        'SRBA-PI']:
            self.schedule = schedule
        else:
            raise FileNotFoundError('schedule must be SRBA, SRBA-PI, ARIA, ARIA-PI, SRIA or SRIA-PI')
        if benchmark in ['Benchmark_A1', 'Benchmark_A2', 'Benchmark_B', 'Benchmark_C']:
            self.benchmark = benchmark
        else:
            raise FileNotFoundError(
                'You can choose benchmarks in "Benchmark_A1", "Benchmark_A2", "Benchmark_B" and "Benchmark_C"')
        if analysis_time > 0:
            self.analysis_time = analysis_time
        else:
            raise ValueError('Analysis time must be a positive float')
        if max_experiments > 0:
            self.max_experiments = max_experiments
        else:
            raise ValueError('Maximal number of experiments must be a positive integer.')
        if pi_min < 1:
            self.pi_min = pi_min
        else:
            raise ValueError('Minimal PI must be a float less than 1.0.')
        if max_break >= 1:
            self.max_break = max_break
        else:
            raise ValueError('Maximal number of break must be a positive integer.')
        if (type(rand_seed) == int and rand_seed >= 0) or rand_seed is None:
            self.rand_seed = rand_seed
        else:
            raise ValueError('Random seed must be a natural number')
        self.equipment, self.equipment_list = create_equipment(reactor_number, analysis_instrument_number)
        self.train_y = []
        self.goal = None
        self.initialization_time = None
        self.end_time = None

    # Run optimization process in simulation
    def run(self):
        # Read benchmark's information
        if self.benchmark == 'Benchmark_A1':
            from benchmarks.Benchmark_A1 import space, run_exp, goal, t_index, least_dist
        elif self.benchmark == 'Benchmark_A2':
            from benchmarks.Benchmark_A2 import space, run_exp, goal, t_index, least_dist
        elif self.benchmark == 'Benchmark_B':
            from benchmarks.Benchmark_B import space, run_exp, goal, t_index, least_dist
        elif self.benchmark == 'Benchmark_C':
            from benchmarks.Benchmark_C import space, run_exp, goal, t_index, least_dist
        else:
            space, run_exp, goal, t_index, least_dist = None, None, None, None, None
        self.goal = goal

        # Random initialization
        if self.reactor_number > 3:
            initialization_number = self.reactor_number
        elif self.reactor_number == 1:
            initialization_number = 5
        else:
            initialization_number = self.reactor_number * 2
        if self.rand_seed is None:
            x = transform(space.transform(space.rvs(n_samples=initialization_number)), space)
        else:
            x = transform(space.transform(space.rvs(n_samples=initialization_number, random_state=self.rand_seed)),
                          space)
        con_x = space.inverse_transform(i_transform(x, space))
        reaction_list = []
        reaction_list = reaction_update(reaction_list, con_x, self.analysis_time, t_index)
        y_check = 1e6
        n_break = 0
        n_real = 0
        n_cat = 0
        if self.schedule == 'ARIA-PI' or self.schedule == 'SRIA-PI' or 'SRBA-PI':
            pi_sort = True
        else:
            pi_sort = False
        while 1:
            reaction_analyzed = []
            if self.schedule == 'SRBA-PI' or self.schedule == 'SRBA':
                work_flow(self.equipment_list, reaction_list, self.equipment, space, waiting=True)
            else:
                work_flow(self.equipment_list, reaction_list, self.equipment, space)
            for i in reaction_list:
                if i.step == 2:
                    reaction_analyzed.append(i)
            if len(set(reaction_analyzed)) == initialization_number:
                break
        train_x = torch.tensor(x)
        all_x = torch.tensor(x)
        for i in self.equipment['reactor']:
            for k in i.history_instance:
                k.obj = run_exp(k.condition)
        for i in range(initialization_number):
            if goal == 'maximize':
                self.train_y.append(run_exp(space.inverse_transform(i_transform(train_x, space))[i]))
            else:
                self.train_y.append(-1 * run_exp(space.inverse_transform(i_transform(train_x, space))[i]))
        self.train_y = torch.tensor(self.train_y)
        self.train_y = self.train_y.unsqueeze(1)

        # Determine the minimum number of experiments
        for j, i in enumerate(space.dimensions):
            para_type = str(type(i)).split('.')[-1][0:-2]
            if para_type == 'Categorical':
                n_cat += len(i.bounds)
            elif para_type == 'Real':
                n_real += 1
        n_elem = n_real + n_cat
        if space.is_categorical:
            n_min = n_elem + 3
            if initialization_number + 2 * self.reactor_number < n_min:
                n_min = n_min
            else:
                n_min = initialization_number + 2 * self.reactor_number
        else:
            n_min = 1
        self.initialization_time = self.equipment['reactor'][0].clocktime
        n_iter = 0
        gp = None

        # active learning loop
        while 1:
            if n_iter:
                gp = get_next_exeps(space, train_x, self.train_y, 1, return_gp=True,
                                    only_return_gp=True)
                PI = ProbabilityOfImprovementXi(gp, best_f=torch.max(self.train_y), maximize=True)

                # PI discarding mechanism to discard ongoing unpromising experiments
                if self.schedule == 'ARIA-PI' or self.schedule == 'SRIA-PI' or self.schedule == 'SRBA-PI':
                    wait_delete = {}
                    for i in self.equipment['analysis_instrument']:
                        wait_delete[i] = []
                        for j in i.waiting_instance:
                            pi = PI(torch.tensor(transform(space.transform([j.condition]), space)))
                            if pi < self.pi_min:
                                wait_delete[i].append(j)
                    for i, j in wait_delete.items():
                        for k in j:
                            i.waiting_instance.remove(k)
                            i.waiting.remove(k.reaction_number)
                            k.step = 3
                    update_eqiupment(self.equipment_list)
                    for j in self.equipment['reactor']:
                        if j.working and j.now_instance != []:
                            if PI(torch.tensor(
                                    transform(space.transform([j.now_instance.condition]), space))) < self.pi_min:
                                j.now_instance.step = 4
                                j.now_instance.end_time = j.clocktime
                                j.working_time += j.clocktime - j.start_time
                                j.end_time = j.clocktime
                                j.working = False
                                j.now_instance = []
                                j.now = []

            # Fit GP and generate new candidate condition
            reactor_ready = []
            for i in self.equipment['reactor']:
                if not i.working:
                    reactor_ready.append(i)
            n_exp_now = train_x.size()[0]
            num_add = 0
            if self.schedule == 'SRIA-PI' or self.schedule == 'SRIA' or self.schedule == 'SRBA-PI' \
                    or self.schedule == 'SRBA':
                classify = len(reactor_ready) >= self.reactor_number  # SR modes
            else:
                classify = len(reactor_ready) != 0  # AR modes
            if classify and len(reaction_list) < self.max_experiments and n_break < self.max_break:
                candidate, qpi, gp = get_next_exeps(space, train_x, self.train_y, len(self.reactor_number), return_gp=True)
                EI = ExpectedImprovement(gp, best_f=torch.max(self.train_y), maximize=True)
                similar_delete_all = []
                dist_delete = []
                similar_delete = []

                # Remove the similar conditions (to previous condition) in the candidates
                # add n_break
                for i, j in enumerate(candidate):
                    for k in all_x:
                        if torch.dist(j, k, p=2) < least_dist:
                            dist_delete.append(i)
                            similar_delete_all.append(i)
                    for o in train_x:
                        if torch.dist(j, o, p=2) < least_dist:
                            n_break += 1
                            similar_delete.append(i)

                # Remove the similar conditions (to other candidates) in the candidates
                if candidate.size()[0] > 1:
                    tep_x = list(range(candidate.size()[0]))
                    for j, i in enumerate(candidate):
                        tep_x.remove(j)
                        if tep_x:
                            for k in torch.index_select(candidate, 0, torch.tensor(tep_x)):
                                if torch.dist(i, k, p=2) < least_dist:
                                    if EI(i.unsqueeze(0)) <= EI(k.unsqueeze(0)):
                                        dist_delete.append(j)

                # PI discarding mechanism, Remove unpromising candidates
                # PI stopping criterion, add n_break
                if self.schedule == 'ARIA-PI' or self.schedule == 'SRIA-PI' or self.schedule == 'SRBA-PI':
                    for i, j in enumerate(qpi):
                        if j > self.pi_min or self.train_y.size()[0] <= n_min:
                            pass
                        else:
                            if i not in similar_delete:
                                dist_delete.append(i)
                                n_break += 1
                    for i, j in enumerate(qpi):
                        new_con = [space.inverse_transform(i_transform(candidate, space))[i]]
                        if i not in dist_delete:
                            all_x = torch.vstack([all_x, candidate[i]])
                            reaction_list = reaction_update(reaction_list, new_con, self.analysis_time, t_index, j)
                            num_add += 1
                        if num_add >= len(reactor_ready):
                            break

                # Calculate the number of similar conditions (to previous condition) in the candidates, add n_break
                # PI stopping criterion, add n_break
                if self.schedule == 'ARIA' or self.schedule == 'SRIA' or self.schedule == 'SRBA':
                    for i, j in enumerate(qpi):
                        if j < self.pi_min and self.train_y.size()[0] > n_min and i not in similar_delete:
                            n_break += 1
                    for i, j in enumerate(qpi):
                        if i not in dist_delete:
                            new_con = [space.inverse_transform(i_transform(candidate, space))[i]]
                            all_x = torch.vstack([all_x, candidate[i]])
                            reaction_list = reaction_update(reaction_list, new_con, self.analysis_time, t_index, j)
                            num_add += 1
                        if num_add >= len(reactor_ready):
                            break
            else:
                gp = get_next_exeps(space, train_x, self.train_y, len(reactor_ready), return_gp=True,
                                    only_return_gp=True)
            analyzed_add = False
            PI = ProbabilityOfImprovementXi(gp, best_f=torch.max(self.train_y), maximize=True)

            # Run the RA simulator and update the data
            while 1:
                reaction_analyzed = []
                if self.schedule == 'SRBA-PI' or self.schedule == 'SRBA':
                    t = work_flow(self.equipment_list, reaction_list, self.equipment, space, return_t=True, PI=PI,
                                  pi_sort=pi_sort, waiting=True)
                else:
                    t = work_flow(self.equipment_list, reaction_list, self.equipment, space, return_t=True, PI=PI,
                                  pi_sort=pi_sort)
                for i in reaction_list:
                    if i.step == 2:
                        reaction_analyzed.append(i)
                if len(set(reaction_analyzed)) >= n_exp_now + 1:
                    analyzed_add = True
                    break
                elif t == -1e-10:
                    break
            reaction_delete = []
            for i in reaction_list:
                if i.step >= 3:
                    reaction_delete.append(i)
            a_i_analyzed = []
            for i in self.equipment['analysis_instrument']:
                if i.history:
                    if np.abs(i.history[-1][1] + i.history[-1][2] - i.clocktime) < 1e-6:
                        a_i_analyzed.append(i)
            if analyzed_add:
                for i in a_i_analyzed:
                    con_analyzed = torch.tensor(
                        transform(space.transform([i.history_instance[-1].condition]), space))
                    train_x = torch.vstack([train_x, con_analyzed])
                    if goal == 'maximize':
                        new_y = [run_exp(i.history_instance[-1].condition)]
                        i.history_instance[-1].obj = new_y[0]
                    else:
                        new_y = [-1 * run_exp(i.history_instance[-1].condition)]
                        i.history_instance[-1].obj = new_y[0] * -1
                    new_y = torch.tensor(np.array(new_y).reshape(1, 1))
                    self.train_y = torch.vstack([self.train_y, new_y])

            # reset the n_break if new optima arise
            if torch.max(self.train_y) > y_check:
                n_break = 0
            y_check = torch.max(self.train_y)
            n_iter += 1

            # Judge whether stop
            if len(reactor_ready) == self.reactor_number and num_add == 0:
                break
            if self.schedule == 'ARIA-PI' or self.schedule == 'SRIA-PI' or self.schedule == 'SRBA-PI':
                if len(set(reaction_analyzed)) >= len(reaction_list) - len(
                        reaction_delete) and n_break >= self.max_break:
                    break
                if len(reaction_list) >= self.max_experiments and len(set(reaction_analyzed)) >= len(
                        reaction_list) - len(reaction_delete):
                    break
            elif self.schedule == 'SRIA' or self.schedule == 'SRBA':
                if n_break >= self.max_break and len(set(reaction_analyzed)) >= len(reaction_list):
                    break
                if self.max_experiments <= len(reaction_list) <= len(set(reaction_analyzed)):
                    break
            else:
                if self.reactor_number > 1:
                    if len(set(reaction_analyzed)) >= len(reaction_list) and n_break >= self.max_break:
                        break
                    if self.max_experiments <= len(reaction_list) <= len(set(reaction_analyzed)):
                        break
                else:
                    if len(set(reaction_analyzed)) >= self.max_experiments or n_break >= self.max_break:
                        break

        # Output results
        if goal == 'maximize':
            opt_obj = np.array(torch.max(self.train_y))
        else:
            opt_obj = np.array(torch.max(self.train_y)) * -1
        opt_con = []
        time = self.equipment_list[0].clocktime
        self.end_time = time
        a_working_time_rate = 0
        r_working_time_rate = 0
        a_end_time = []
        r_end_time = []
        for i in self.equipment['reactor']:
            r_end_time.append(i.end_time)
            r_working_time_rate += i.working_time
        for i in self.equipment['analysis_instrument']:
            a_end_time.append(i.end_time)
            a_working_time_rate += i.working_time
        r_working_time_rate = (r_working_time_rate / max(r_end_time)) / self.reactor_number
        a_working_time_rate = (a_working_time_rate / max(a_end_time)) / self.analysis_instrument_number
        for i in reaction_list:
            if i.obj:
                if np.abs(i.obj - opt_obj) < 1e-10:
                    i.is_optima = True
                    opt_con.append(i.condition)

        return Results(
            [opt_obj, opt_con, time, len(reaction_list), r_working_time_rate, a_working_time_rate,
             space.inverse_transform(i_transform(train_x, space)), self.train_y.detach().numpy().tolist()])
