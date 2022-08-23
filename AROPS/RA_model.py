# the Reactor-analyzer simulator
import numpy as np
from transformer import transform
import torch


# The class of Analysis Instruments
class AnalysisInstrument:
    def __init__(self, number):
        self.number = number
        self.type_name = 'Analysis Instrument'
        self.working = False
        self.history = []
        self.now = []
        self.waiting = []
        self.clocktime = 0
        self.waiting_instance = []
        self.history_instance = []
        self.now_instance = []
        self.working_time = 0
        self.start_time = 0
        self.end_time = 0

    # add reactions to Analysis Instruments
    def add_task(self, reaction):
        self.waiting.append(reaction.reaction_number)
        self.waiting_instance.append(reaction)

    # run the device
    def run(self):
        self.now = self.waiting[0]
        self.now_instance = self.waiting_instance[0]
        self.now_instance.start_analyze = True
        self.history.append(
            [self.waiting[0], self.clocktime, self.now_instance.analysis_time, self.now_instance.reactor_no])
        del self.waiting[0]
        del self.waiting_instance[0]
        self.start_time = self.clocktime
        self.working = True

    # calculate the remaining running time
    def remaining_time(self):
        if self.now_instance != []:
            if self.working and self.now_instance.start_analyze_update:
                remaining_time = self.now_instance.analysis_time - (self.clocktime - self.start_time)
            else:
                remaining_time = -1e-12
                self.now_instance.start_analyze_update = True
        else:
            remaining_time = -1e-10
        return remaining_time

    # update the state of the devices
    def update_state(self):
        if self.working and self.now_instance != []:
            remaining_time = self.now_instance.analysis_time - (self.clocktime - self.start_time)
            if 1e-5 >= remaining_time >= -1e-11:
                self.working = False
                self.now_instance.step += 1
                self.working_time += self.clocktime - self.start_time
                self.end_time = self.clocktime
                self.history_instance.append(self.now_instance)
                self.now = []
                self.now_instance = []


# The class of Reactor
class Reactor:
    def __init__(self, number):
        self.number = number
        self.type_name = 'Reactor'
        self.history = []
        self.waiting = []
        self.now = []
        self.working = False
        self.clocktime = 0
        self.waiting_instance = []
        self.history_instance = []
        self.now_instance = []
        self.working_time = 0
        self.start_time = 0
        self.end_time = 0

    def add_task(self, reaction):
        self.waiting.append(reaction.reaction_number)
        self.waiting_instance.append(reaction)

    def run(self):
        self.now.append(self.waiting[0])
        self.now_instance = self.waiting_instance[0]
        self.now_instance.reactor_no = self.number
        self.history.append([self.waiting[0], self.clocktime, self.now_instance.reaction_time])
        self.now_instance.start_time = self.clocktime
        self.history_instance.append(self.now_instance)
        del self.waiting[0]
        del self.waiting_instance[0]
        self.start_time = self.clocktime
        self.working = True

    def remaining_time(self):
        if self.working and self.now_instance != []:
            remaining_time = self.now_instance.reaction_time - (self.clocktime - self.start_time)
        else:
            remaining_time = -1e-10
        return remaining_time

    def update_state(self):
        if self.working and self.now_instance != []:
            remaining_time = self.now_instance.reaction_time - (self.clocktime - self.start_time)
            if 1e-5 >= remaining_time >= -1e-11:
                self.now_instance.step += 1
                self.working_time += self.clocktime - self.start_time
                self.end_time = self.clocktime
                self.now_instance.end_time = self.clocktime
                self.now = []
                self.now_instance = []
        if self.working and self.now_instance == [] and (
                self.history_instance[-1].step >= 3 or self.history_instance[-1].start_analyze):
            self.working = False


# The class of reactions
class Reaction:
    def __init__(self, reaction_number, reaction_time, analysis_time):
        self.reaction_number = reaction_number
        self.reaction_time = reaction_time
        self.analysis_time = analysis_time
        self.reactor_no = None
        self.step = 0
        self.start_time = 0
        self.end_time = 0
        self.condition = np.array([])
        self.obj = None
        self.is_optima = False
        self.start_analyze = False
        self.start_analyze_update = False
        self.pi = None


# Synchronize the remaining time of the devices
def eq_termin(equipment_list):
    remaining_time = []
    for i in equipment_list:
        remaining_time_i = i.remaining_time()
        if remaining_time_i >= -1e-11:
            remaining_time.append(remaining_time_i)
    if remaining_time:
        return min(remaining_time)
    else:
        return -1e-10


# Synchronously update the state of devices
def update_eqiupment(equipment_list):
    for i in reversed(equipment_list):
        i.update_state()


# Synchronize the time of the devices
def time_update(t, equipment_list):
    for i in equipment_list:
        i.clocktime = i.clocktime + t


# Run the devices
def equipment_run(equipment_list):
    for i in equipment_list:
        if i.working == False and len(i.waiting) != 0:
            i.run()


# Add reactions to the devices
def reaction_add(reaction, equipment, space, PI=None, pi_sort=False, waiting=False):
    a_i = equipment['analysis_instrument']
    reactor = equipment['reactor']
    reaction_step0 = []
    reaction_step1 = []
    reactor_ready = []
    a_i_list = []
    a_i_dict = {}
    reaction_step1_ready = []
    for i in a_i:
        if not i.working:
            a_i_list.append(i)
    for i in a_i:
        if i.working:
            if i.waiting_instance:
                a_i_dict[i] = i.remaining_time() + len(i.waiting_instance) * i.waiting_instance[0].analysis_time
            else:
                a_i_dict[i] = i.remaining_time()
    a_i_dict = sorted(a_i_dict.items(), key=lambda item: item[1])
    for i in a_i_dict:
        a_i_list.append(i[0])
    for i in reaction:
        if i.step == 0:
            reaction_step0.append(i)
        if i.step == 1:
            reaction_step1.append(i)
    if pi_sort and PI:
        for i, j in enumerate(reaction_step1):
            if i >= len(reaction_step1) - 1:
                break
            if PI(torch.tensor(transform(space.transform([j.condition]), space))) < PI(
                    torch.tensor(transform(space.transform([reaction_step1[i + 1].condition]), space))):
                tem = j
                reaction_step1[i] = reaction_step1[i + 1]
                reaction_step1[i + 1] = tem
    for i in equipment['reactor']:
        if not i.working:
            reactor_ready.append(i)
    num_reactor = len(reactor) - len(reactor_ready)
    for j, i in enumerate(reaction_step1):
        a_i_exist_number = []
        for k in equipment['analysis_instrument']:
            for m in k.waiting:
                a_i_exist_number.append(m)
            a_i_exist_number.append(k.now)
            for n in k.history:
                a_i_exist_number.append(n[0])
        if i.reaction_number not in a_i_exist_number:
            reaction_step1_ready.append(i)
    if waiting:
        if len(reaction_step1_ready) >= num_reactor:
            for j, i in enumerate(reaction_step1_ready):
                a_i_list[j % len(a_i_list)].add_task(i)
    else:
        for j, i in enumerate(reaction_step1_ready):
            a_i_list[j % len(a_i_list)].add_task(i)
    for j, i in enumerate(reaction_step0):
        reactor_exist_number = []
        for k in reactor:
            for o in k.now:
                reactor_exist_number.append(o)
            for o in k.waiting:
                reactor_exist_number.append(o)
            for m in k.history:
                reactor_exist_number.append(m[0])
        if i.reaction_number not in reactor_exist_number and len(reactor_ready) != 0:
            reactor_ready[j % len(reactor_ready)].add_task(i)


# create the simulator's workflow
def work_flow(equipment_list, reaction, equipment, space, PI=None, return_t=False, pi_sort=False, waiting=False):
    reaction_add(reaction, equipment, space, PI, pi_sort, waiting=waiting)
    equipment_run(equipment_list)
    t = eq_termin(equipment_list)
    time_update(t, equipment_list)
    update_eqiupment(equipment_list)
    if return_t:
        return t


# create the devices instruments
def create_equipment(reactor_number, a_i_number):
    equipment = {'reactor': [], 'analysis_instrument': []}
    reactor_dict = {}
    for i in range(reactor_number):
        reactor_dict[i] = Reactor(i)
    equipment['reactor'] = list(reactor_dict.values())
    a_i_dict = {}
    for i in range(a_i_number):
        a_i_dict[i] = AnalysisInstrument(i)
    equipment['analysis_instrument'] = list(a_i_dict.values())
    equipment_list = equipment['reactor'] + equipment['analysis_instrument']
    return equipment, equipment_list


# update the state of reactions
def reaction_update(reaction_list, con_x, analysis_time, t_index, pi=None):
    reaction_dict = {}
    for i, j in enumerate(con_x):
        if type(t_index) == int:
            reaction_dict[i + len(reaction_list)] = Reaction(i + len(reaction_list), j[t_index], analysis_time)
        elif type(t_index) == float:
            reaction_dict[i + len(reaction_list)] = Reaction(i + len(reaction_list), t_index, analysis_time)
        reaction_dict[i + len(reaction_list)].condition = j
        if pi:
            reaction_dict[i + len(reaction_list)].pi = pi
    return reaction_list + list(reaction_dict.values())
