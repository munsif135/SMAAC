import numpy as np
import torch
import itertools

# episode means : This process repeats until the end of the episode, which might happen when a blackout occurs, a given time horizon is met, or some other termination criterion is reached.
CASE_CONFIG = {
    # sub, gen, load, line
    5: (5, 2, 3, 8),
    14: (14, 5, 11, 20), # (14, 6, 11, 20)
    118: (118, 62, 99, 186),
    36: (36, 22, 37, 59)
}


class graphGoalConverter:
    def __init__(self, env, mask, mask_hi, danger, device, rule='c'):
        self.obs_space = env.observation_space 
        self.action_space = env.action_space
        self.mask = mask # 2
        self.mask_hi = mask_hi # 19
        self.danger = danger # 0.9
        self.rule = rule # 'c'
        self.device = device
        self.thermal_limit_under400 = torch.from_numpy(env._thermal_limit_a < 400)
        self.init_obs_converter()
        self.init_action_converter()

    def init_obs_converter(self):        
        self.idx = self.obs_space.shape #gives the size of the each observation eg: observation related _shunt_bus :1, load_q:11 we can get this value by gym_env.observation_space-> load_q': Box(-inf, inf, (11,), float32) the 11 indicates the size of load_q is 11
        self.pp = np.arange(sum(self.idx[:6]),sum(self.idx[:7])) #22 values,  indexes of active power value of each generator (there are 22 gens)
        self.lp = np.arange(sum(self.idx[:9]),sum(self.idx[:10])) # 37 values, indexes of active load value of each load (expressed in MW).(there are 37 loads)
        self.op = np.arange(sum(self.idx[:12]),sum(self.idx[:13])) # 59 values:  indexes of The active power flow at the origin end of each powerline (there are 59 powerlines)
        self.ep = np.arange(sum(self.idx[:16]),sum(self.idx[:17]))  # 59 values:  indexes of The active power flow at the extremity end of each powerline (there are 59 powerlines)
        self.rho = np.arange(sum(self.idx[:20]),sum(self.idx[:21])) # 59 values:  indexes of The capacity of each powerline. It is defined at the observed current flow divided by the thermal limit of each powerline (no unit)
        self.topo = np.arange(sum(self.idx[:23]),sum(self.idx[:24])) # 177 values:  topo_vect(For each object (load, generator, ends of a powerline) it gives on which bus this object is connected)
        self.main = np.arange(sum(self.idx[:26]),sum(self.idx[:27])) # 59 values: indexes related to time_next_maintenance (time of the next planned maintenance)
        self.over = np.arange(sum(self.idx[:22]),sum(self.idx[:23])) # 59 values:timestep_overflow: number of time steps during which the powerline is on overflow (max over all powerlines)
        
        # parse substation info( Collection details related to gens, loads etc connected to easch substation)
        self.subs = [{'e':[], 'o':[], 'g':[], 'l':[]} for _ in range(self.action_space.n_sub)] #action_space.n_sub = number of substations
        for gen_id, sub_id in enumerate(self.action_space.gen_to_subid): # ids of substations where generators are connected
            self.subs[sub_id]['g'].append(gen_id)
        for load_id, sub_id in enumerate(self.action_space.load_to_subid): # ids of substations where loads are connected
            self.subs[sub_id]['l'].append(load_id)
        for or_id, sub_id in enumerate(self.action_space.line_or_to_subid): # ids of substations where line orgins are connected
            self.subs[sub_id]['o'].append(or_id)
        for ex_id, sub_id in enumerate(self.action_space.line_ex_to_subid): # ids of substations where line extremes are connected
            self.subs[sub_id]['e'].append(ex_id)
    
        '''
        results looks like: 
        subs = [{'e': [], 'o': [2], 'g': [], 'l': [0, 1]}, #loads 0 and 1 are connected to substation 0
        {'e': [], 'o': [3, 4, 12], 'g': [0], 'l': [2, 3]}, #orgin lines 3, 4, 12 and loads 2,3 are connected to substation 1
        .
        .
        '''
        # numbers of the lines that are connected to each substation
        self.sub_to_topos = []  # [0]: [0, 1, 2](line 0,1,2 are connected to the 0th substation ), [1]: [3, 4, 5, 6, 7, 8]
        for sub_info in self.subs:
            a = []
            for i in sub_info['e']:
                a.append(self.action_space.line_ex_pos_topo_vect[i])
            for i in sub_info['o']:
                a.append(self.action_space.line_or_pos_topo_vect[i])
            for i in sub_info['g']:
                a.append(self.action_space.gen_pos_topo_vect[i])
            for i in sub_info['l']:
                a.append(self.action_space.load_pos_topo_vect[i])
            self.sub_to_topos.append(torch.LongTensor(a))

        '''
        results looks like: 
        sub_to_topos = tensor([0, 1, 2]), line # 1,2 and 3 are connected to 0th substation.
                            tensor([3, 4, 5, 6, 7, 8]),
                            tensor([ 9, 10, 11]),
                .
                .
        '''

        # split topology over sub_id
        self.sub_to_topo_begin, self.sub_to_topo_end = [], []
        idx = 0
        for num_topo in self.action_space.sub_info: #action_space.sub_info - number of lines that are connected to each substations
            self.sub_to_topo_begin.append(idx)
            idx += num_topo
            self.sub_to_topo_end.append(idx)
        dim_topo = self.idx[-7]
        self.last_topo = np.ones(dim_topo, dtype=np.int32)
        self.n_feature = 5

        '''
        results looks like: 
        last_topo = array([1, 1, 1, 1, 1.....(177 ones)....)]
        sub_to_topo_begin = [0,3,9,12,....] # 9=3+6, 12=3+6+3, this 3,6,3 are number of lines that are connected to each substations
        sub_to_topo_end = [3,9,12,15] # same as sub_to_topo_begin but start with 1st index value
        '''

    def convert_obs(self, o):
        # o.shape : (B, O)
        # output (Batch, Node, Feature)
        length = self.action_space.dim_topo # N : The total number of objects in the powergrid. This is also the dimension of the “topology vector” 
        
        # active power p
        p_ = torch.zeros(o.size(0), length).to(o.device)    # (B, N) o.size(0)=1, length = 177
        # ALL the power values
        #o[...,  self.pp] : gives the  active power value of all 22 generators, other 3 eqns adds 37 power values of the loads, 59 power values of the powerlines orgins and 59 power values of the powerlines extremes
        p_[..., self.action_space.gen_pos_topo_vect] = o[...,  self.pp] # action_space.gen_pos_topo_vect: element id of each generators. element id - overall id number
        p_[..., self.action_space.load_pos_topo_vect] = o[..., self.lp] # action_space.load_pos_topo_vect: element id of each loads
        p_[..., self.action_space.line_or_pos_topo_vect] = o[..., self.op] # action_space.line_or_pos_topo_vect: element id of each line orgins
        p_[..., self.action_space.line_ex_pos_topo_vect] = o[..., self.ep] # action_space.line_ex_pos_topo_vect: element id of each line extreme

        # ALL the rho(The capacity of each powerline. It is defined at the observed current flow divided by the thermal limit of each powerline) values
        # rho (powerline usage ratio)
        rho_ = torch.zeros(o.size(0), length).to(o.device) # o.size(0)=1, length = 177, except for line orgin and extreme indexes other index values will be zero
        rho_[..., self.action_space.line_or_pos_topo_vect] = o[..., self.rho] # action_space.line_or_pos_topo_vect: element id of each line orgins
        rho_[..., self.action_space.line_ex_pos_topo_vect] = o[..., self.rho] # action_space.line_ex_pos_topo_vect: element id of each line extreme

        # whether each line is in danger
        # self.danger = 0.9
        danger_ = torch.zeros(o.size(0), length).to(o.device)
        # thermal_limit_under400 = [True,  True,  True,  True, False,....] checking whether thermal limit of the lines are below 400
        # (o[...,self.rho] >= self.danger-0.05) checking rho>0.85
        danger = ((o[...,self.rho] >= self.danger-0.05) & self.thermal_limit_under400.to(o.device)) | (o[...,self.rho] >= self.danger)
        danger_[..., self.action_space.line_or_pos_topo_vect] = danger.float() #danger.float() = [0., 0., 1., 0., 1.,...] 0 -true, 1-false
        danger_[..., self.action_space.line_ex_pos_topo_vect] = danger.float()      

        # whether overflow occurs in each powerline
        over_ = torch.zeros(o.size(0), length).to(o.device)
        over_[..., self.action_space.line_or_pos_topo_vect] = o[..., self.over]/3
        over_[..., self.action_space.line_ex_pos_topo_vect] = o[..., self.over]/3
        # over_ = [ 0.3333, 0.0000, 0.0000] if number of time steps during which the powerline is on overflow is 1, 0.3333 will be the value

        # whether each powerline is in maintenance
        main_ = torch.zeros(o.size(0), length).to(o.device)
        temp = torch.zeros_like(o[..., self.main])
        temp[o[..., self.main]==0] = 1
        main_[..., self.action_space.line_or_pos_topo_vect] = temp
        main_[..., self.action_space.line_ex_pos_topo_vect] = temp
        
        '''
        main_ = [0., 0., 0., 0., 0.,...] 
        1 at position i it means that the powerline i will be disconnected for maintenance operation at the next time step.
        0 at position i means that powerline i is disconnected from the powergrid for maintenance operation at the current time step.
        -1 at position i means that powerline i will not be disconnected for maintenance reason for this episode.
        k > 1 at position i it means that the powerline i will be disconnected for maintenance operation at in k time steps
        '''


        # current bus assignment
        topo_ = torch.clamp(o[..., self.topo] - 1, -1) # resulting topo_ = tensor([[0., 0., 0., 0., 0.,.....]])

        state = torch.stack([p_, rho_, danger_, over_, main_], dim=2) # B, N, F (1, 177, 5])
        return state, topo_.unsqueeze(-1)
  
    def init_action_converter(self):
        self.sorted_sub = list(range(self.action_space.n_sub))
        self.sub_mask = []  # mask for parsing actionable topology
        self.psubs = []     # actionable substation IDs
        self.masked_sub_to_topo_begin = []
        self.masked_sub_to_topo_end = []
        idx = 0
        for i, num_topo in enumerate(self.action_space.sub_info): #action_space.sub_info - number of lines that are connected to each substations
            if num_topo > self.mask and num_topo < self.mask_hi: #self.mask = 2 self.mask_hi = 19, all the substations which has number of line connections b/w 2 and 19 
                '''sub_to_topo_begin = [0,3,9,12,....] # 9=3+6, 12=3+6+3, this 3,6,3 are number of lines that are connected to each substations
                    sub_to_topo_end = [3,9,12,15] # same as sub_to_topo_begin but start with 1st index value'''
                self.sub_mask.extend(
                    [j for j in range(self.sub_to_topo_begin[i]+1, self.sub_to_topo_end[i])])
                self.psubs.append(i)
                self.masked_sub_to_topo_begin.append(idx)
                idx += num_topo-1
                self.masked_sub_to_topo_end.append(idx)

            else:
                self.masked_sub_to_topo_begin.append(-1)
                self.masked_sub_to_topo_end.append(-1)
        self.n = len(self.sub_mask)

        if self.rule == 'f':
            if self.obs_space.n_sub == 5:
                self.masked_sorted_sub = [4, 0, 1, 3, 2]
            elif self.obs_space.n_sub == 14:
                self.masked_sorted_sub = [13, 5, 0, 12, 9, 6, 10, 1, 11, 3, 4, 7, 2]
            elif self.obs_space.n_sub == 36: # mask = 5
                self.masked_sorted_sub = [9, 33, 29, 7, 21, 1, 4, 23, 16, 26, 35]
                if self.mask == 4:
                    self.masked_sorted_sub = [35, 23, 9, 33, 4, 28, 1, 32, 13, 21, 26, 29, 16, 22, 7, 27]
        else:
            if self.obs_space.n_sub == 5:
                self.masked_sorted_sub = [0, 3, 2, 1, 4]
            elif self.obs_space.n_sub == 14:
                self.masked_sorted_sub = [5, 1, 3, 4, 2, 12, 0, 11, 13, 10, 9, 6, 7]
            elif self.obs_space.n_sub == 36: # mask = 5
                self.masked_sorted_sub = [16, 23, 21, 26, 33, 29, 35, 9, 7, 4, 1]
                if self.mask == 4:
                    self.masked_sorted_sub += [22, 27, 28, 32, 13]

        # powerlines which are not controllable by bus assignment action
        self.lonely_lines = set()
        for i in range(self.obs_space.n_line): #obs_space.n_line number of lines
            # extracting the lines that doesn't have neither orgin nor extreme 
            if (self.obs_space.line_or_to_subid[i] not in self.psubs) \
               and (self.obs_space.line_ex_to_subid[i] not in self.psubs): # psubs : actionable substation IDs
                self.lonely_lines.add(i)
        self.lonely_lines = list(self.lonely_lines) # converting the set to list
        print('Lonely line', len(self.lonely_lines), self.lonely_lines)
        print('Masked sorted topology', len(self.masked_sorted_sub), self.masked_sorted_sub)

    def inspect_act(self, sub_id, goal, topo_vect):
        # Correct illegal action collect original ids
        exs = self.subs[sub_id]['e']
        ors = self.subs[sub_id]['o']
        lines = exs + ors   # [line_id0, line_id1, line_id2, ...]
        
        # minimal prevention of isolation
        line_idx = len(lines) - 1 
        if (goal[:line_idx] == 1).all() * (goal[line_idx:] != 1).any():
            goal = torch.ones_like(goal)
        
        if torch.is_tensor(goal): goal = goal.numpy()
        beg = self.masked_sub_to_topo_begin[sub_id]
        end = self.masked_sub_to_topo_end[sub_id]
        already_same = np.all(goal == topo_vect[self.sub_mask][beg:end])
        return goal, already_same

    def convert_act(self, sub_id, new_topo, obs=None):
        new_topo = [1] + new_topo.tolist()
        act = self.action_space({'set_bus': {'substations_id': [(sub_id, new_topo)]}})
        return act

    def plan_act(self, goal, topo_vect, sub_order_score=None):
        # Compare obs.topo_vect and goal, then parse partial order from whole topological sort
        topo_vect = torch.LongTensor(topo_vect)
        topo_vect = topo_vect[self.sub_mask]
        targets = []
        goal = goal.squeeze(0).cpu() + 1

        if sub_order_score is None:
            sub_order = self.masked_sorted_sub
        else:
            sub_order = [i[0] for i in sorted(list(zip(self.masked_sorted_sub, sub_order_score.tolist())),
                        key=lambda x: -x[1])]

        for sub_id in sub_order:
            beg = self.masked_sub_to_topo_begin[sub_id]
            end = self.masked_sub_to_topo_end[sub_id]
            topo = topo_vect[beg:end]
            new_topo = goal[beg:end]
            if torch.any(new_topo != topo).item():
                targets.append((sub_id, new_topo))

        # Assign sequentially actions from the goal
        plan = [(sub_id, new_topo) for sub_id, new_topo in targets]
        return plan

    def heuristic_order(self, obs, low_actions):
        if len(low_actions) == 0:
            return []
        rhos = []
        for item in low_actions:
            sub_id = item[0]
            lines = self.subs[sub_id]['e'] + self.subs[sub_id]['o']
            rho = obs.rho[lines].copy()
            rho[rho==0] = 3
            rho_max = rho.max()
            rho_mean = rho.mean()
            rhos.append((rho_max, rho_mean))
        order = sorted(zip(low_actions, rhos), key=lambda x: (-x[1][0], -x[1][1]))
        return list(list(zip(*order))[0])
