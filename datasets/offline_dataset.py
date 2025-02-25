import numpy as np
import torch, pdb
import h5py

class ReplayBuffer(object):
    def __init__(self, obs_dim, action_dim, state_dim, n_agents, env_name, data_dir, max_size=int(2e6), device='cuda'):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.n_agents = n_agents
        self.env_name = env_name
        self.data_dir = data_dir

        self.o = np.zeros((max_size, n_agents, obs_dim))
        self.s = np.zeros((max_size, n_agents, state_dim))
        self.a = np.zeros((max_size, n_agents, action_dim))
        self.r = np.zeros((max_size, 1))
        self.mask = np.zeros((max_size, 1))
        self.s_next = np.zeros((max_size, n_agents, obs_dim))
        self.o_next = np.zeros((max_size, n_agents, state_dim))
        self.a_next = np.zeros((max_size, n_agents, action_dim))
        self.device = torch.device(device)

    def sample(self, batch_size):
        o_size = self.o.shape[0]
        ind = np.random.randint(0, o_size, size=batch_size)  
        return (
            torch.FloatTensor(self.o[ind]).to(self.device),
            torch.FloatTensor(self.s[ind]).to(self.device),
            torch.FloatTensor(self.a[ind]).to(self.device),
            torch.FloatTensor(self.r[ind]).to(self.device),  
            torch.FloatTensor(self.mask[ind]).to(self.device),
            torch.FloatTensor(self.s_next[ind]).to(self.device),
            torch.FloatTensor(self.o_next[ind]).to(self.device),
            torch.FloatTensor(self.a_next[ind]).to(self.device)
        )

    def load(self, env_name='HC', obsk=0, offline_ver=0, agent_view_radius=0.5, n_agents=3, n_actions=2):
        pp = False
        print('==========Data loading==========')
        data_file = self.data_dir + self.env_name + '.hdf5'
        data_file2 = None
        # data_file = self.data_dir + 'test.hdf5'
        data_file = './data/'
        if 'aaCh' in env_name:
            if obsk == 0:
                if offline_ver > 2:
                    data_file2 = './data/'
                if offline_ver == 0:
                    data_file = data_file + 'HalfCheetah-v2_obsk0_from_ADER_4804000'
                elif offline_ver == 1:
                    data_file = data_file + 'HalfCheetah-v2_obsk0_from_ADER_1004000'
                elif offline_ver == 2:
                    data_file =  data_file +'HalfCheetah-v2_obsk0_from_ADER_404000'
                elif offline_ver == 3:
                    data_file =  data_file + 'HalfCheetah-v2_obsk0_from_ADER_4804000'
                    data_file2 =  data_file2 + 'HalfCheetah-v2_obsk0_from_ADER_1004000'
                    data_file2 = data_file2 + '_seed_0.hdf5'
                elif offline_ver == 4:
                    data_file =  data_file + 'HalfCheetah-v2_obsk0_from_ADER_4804000'
                    data_file2 =  data_file2 + 'HalfCheetah-v2_obsk0_from_ADER_404000'
                    epoch_total = 1050
                    data_file2 = data_file2 + '_seed_0.hdf5'
                elif offline_ver == 5:
                    data_file =  data_file + 'HalfCheetah-v2_obsk0_from_ADER_1004000'
                    data_file2 =  data_file2 + 'HalfCheetah-v2_obsk0_from_ADER_404000'
                    data_file2 = data_file2 + '_seed_0.hdf5'
                    epoch_total = 1050
            elif obsk == 1:
                if offline_ver > 2:
                    data_file2 = './data/'
                if offline_ver == 0:
                    data_file = data_file + 'HalfCheetah-v2_obsk1_from_ADER2_0'
                elif offline_ver == 1:
                    data_file = data_file + 'HalfCheetah-v2_obsk1_from_ADER2_804000'
                elif offline_ver == 2:
                    data_file =  data_file +'HalfCheetah-v2_obsk1_from_ADER2_604000'
                elif offline_ver == 3:
                    data_file =  data_file + 'HalfCheetah-v2_obsk1_from_ADER2_0'
                    data_file2 =  data_file2 + 'HalfCheetah-v2_obsk1_from_ADER2_804000'
                    data_file2 = data_file2 + '_seed_0.hdf5'
                elif offline_ver == 4:
                    data_file =  data_file + 'HalfCheetah-v2_obsk1_from_ADER2_0'
                    data_file2 =  data_file2 + 'HalfCheetah-v2_obsk1_from_ADER2_604000'
                    data_file2 = data_file2 + '_seed_0.hdf5'
                elif offline_ver == 5:
                    data_file =  data_file + 'HalfCheetah-v2_obsk1_from_ADER2_804000'
                    data_file2 =  data_file2 + 'HalfCheetah-v2_obsk1_from_ADER2_604000'
                    data_file2 = data_file2 + '_seed_0.hdf5'
            data_file = data_file + '_seed_0.hdf5'
        
        elif 'Walker' in env_name:
            if obsk == 0:
                if offline_ver > 2:
                    data_file2 = './data/'
                if offline_ver == 0:
                    data_file = data_file + 'Walker_2X3_obsk0_expert_dataset'
                elif offline_ver == 1:
                    data_file = data_file + 'Walker_2X3_obsk0_medium_dataset'

            data_file = data_file + '.hdf5'
        elif 'pred_prey' in env_name:
            if offline_ver > 2:
                data_file2 = './data/'
            if agent_view_radius == 0.5:
                if offline_ver == 0:
                    data_file = data_file + 'continuous_pred_prey_3a0.5_from_ADER_3000050.hdf5'
                elif offline_ver == 1:
                    data_file = data_file + 'continuous_pred_prey_3a0.5_from_ADER_500050.hdf5'
                elif offline_ver == 2:
                    data_file =  data_file +'continuous_pred_prey_3a0.5_from_ADER_200050.hdf5'
                elif offline_ver == 3:
                    data_file =  data_file + 'continuous_pred_prey_3a0.5_from_ADER_3000050.hdf5'
                    data_file2 =  data_file2 + 'continuous_pred_prey_3a0.5_from_ADER_500050.hdf5'
                elif offline_ver == 4:
                    data_file =  data_file +  'continuous_pred_prey_3a0.5_from_ADER_3000050.hdf5'
                    data_file2 =  data_file2 + 'continuous_pred_prey_3a0.5_from_ADER_200050.hdf5'
                elif offline_ver == 5:
                    data_file =  data_file + 'continuous_pred_prey_3a0.5_from_ADER_500050.hdf5'
                    data_file2 =  data_file2 + 'continuous_pred_prey_3a0.5_from_ADER_200050.hdf5'
                elif agent_view_radius == 0.3:
                    if offline_ver == 0:
                        data_file = data_file + 'continuous_pred_prey_3a0.3_from_ADER_2000050.hdf5'
                    elif offline_ver == 1:
                        data_file = data_file + 'continuous_pred_prey_3a0.3_from_ADER_400050.hdf5'
                    elif offline_ver == 2:
                        data_file =  data_file + 'continuous_pred_prey_3a0.3_from_ADER_2000050.hdf5'
                        data_file2 =  data_file2 + 'continuous_pred_prey_3a0.3_from_ADER_400050.hdf5'
                else:
                    print("DO NOT EXIST")
                    pdb.set_trace()

        elif 'Cheetah' or 'Ant' or 'Hopper' in env_name:
            data_file = self.data_dir + self.env_name + '.hdf5'


        data_file = self.data_dir + self.env_name + '.hdf5'
        if 'pred_prey' in env_name:
            data_file = './data/'
            if offline_ver >= 2:
                data_file2 = './data/'
            if agent_view_radius == 0.5:
                if offline_ver == 0:
                    data_file = data_file + 'continuous_pred_prey_3a0.5_from_ADER_3000050.hdf5'
                elif offline_ver == 1:
                    data_file = data_file + 'continuous_pred_prey_3a0.5_from_ADER_500050.hdf5'
                elif offline_ver == 2:
                    data_file =  data_file +'continuous_pred_prey_3a0.5_from_ADER_200050.hdf5'
                elif offline_ver == 3:
                    data_file =  data_file + 'continuous_pred_prey_3a0.5_from_ADER_3000050.hdf5'
                    data_file2 =  data_file2 + 'continuous_pred_prey_3a0.5_from_ADER_500050.hdf5'
                elif offline_ver == 4:
                    data_file =  data_file +  'continuous_pred_prey_3a0.5_from_ADER_3000050.hdf5'
                    data_file2 =  data_file2 + 'continuous_pred_prey_3a0.5_from_ADER_200050.hdf5'
                elif offline_ver == 5:
                    data_file =  data_file + 'continuous_pred_prey_3a0.5_from_ADER_500050.hdf5'
                    data_file2 =  data_file2 + 'continuous_pred_prey_3a0.5_from_ADER_200050.hdf5'
            elif agent_view_radius == 0.3:
                if offline_ver == 0:
                    data_file = data_file + 'continuous_pred_prey_3a0.3_from_ADER_2000050.hdf5'
                elif offline_ver == 1:
                    data_file = data_file + 'continuous_pred_prey_3a0.3_from_ADER_400050.hdf5'
                elif offline_ver == 2:
                    data_file =  data_file + 'continuous_pred_prey_3a0.3_from_ADER_2000050.hdf5'
                    data_file2 =  data_file2 + 'continuous_pred_prey_3a0.3_from_ADER_400050.hdf5'
        pp = False
        mix = False
        
        # data_file = self.data_dir + self.env_name + '.hdf5'
        print('Loading from:', data_file)
        print('Loading from 2:', data_file2)

        f = h5py.File(data_file, 'r')
        s = np.array(f['s'])
        o = np.array(f['o'])
        a = np.array(f['a'])
        r = np.array(f['r'])
        d = np.array(f['d'])
        f.close()

        print("========= original shape ==============")

        print("s shape : ", s.shape)
        print("o shape : ", o.shape)
        print("a shape : ", a.shape)
        print("r shape : ", r.shape)
        print("d shape : ", d.shape)

        print("=================================================")
        


        if 'pred_prey' in env_name:
            s = s[:int(s.shape[0]/10)]
            o = o[:int(o.shape[0]/10)]
            a = a[:int(a.shape[0]/10)]
            r = r[:int(r.shape[0]/10)]
            d = d[:int(d.shape[0]/10)]

        data_file2 = None

        if data_file2 is not None:
            f = h5py.File(data_file2, 'r')
            s_ = np.array(f['s'])
            o_ = np.array(f['o'])
            a_ = np.array(f['a'])
            r_ = np.array(f['r'])
            d_ = np.array(f['d'])
            f.close()

            if 'pred_prey' in env_name:
                s_ = s_[:int(s_.shape[0]/10)]
                o_ = o_[:int(o_.shape[0]/10)]
                a_ = a_[:int(a_.shape[0]/10)]
                r_ = r_[:int(r_.shape[0]/10)]
                d_ = d_[:int(d_.shape[0]/10)]

            s = np.concatenate([s[:int(s.shape[0]/2)], s_[:int(s.shape[0]/2)]], axis=0)
            o = np.concatenate([o[:int(s.shape[0]/2)], o_[:int(s.shape[0]/2)]], axis=0)
            a = np.concatenate([a[:int(s.shape[0]/2)], a_[:int(s.shape[0]/2)]], axis=0)
            r = np.concatenate([r[:int(s.shape[0]/2)], r_[:int(s.shape[0]/2)]], axis=0)
            d = np.concatenate([d[:int(s.shape[0]/2)], d_[:int(s.shape[0]/2)]], axis=0)
            
        

            # avail_a = np.concatenate([avail_a, avail_a_], axis=0)
            # filled = np.concatenate([filled, filled_], axis=0)
        
        avg_epi_ret_in_dataset = np.sum(r, axis=1).mean()
        max_epi_ret_in_dataset2 = np.sum(r, axis=1).max()

        print("############ avg_epi_ret_in_dataset : ", avg_epi_ret_in_dataset)
        print("############ max_epi_ret_in_dataset : ", max_epi_ret_in_dataset2)


        if mix:            
            print("################################################")
            print("avg epi rew : ", r.sum(axis=1).mean())
            print("################################################")
            s = s.reshape(-1, 1, s.shape[-1])
            s = np.repeat(s, n_agents, axis=1)
            o = o.reshape(-1, n_agents, o.shape[-1])
            a = a.reshape(-1, n_agents, a.shape[-1])
            r = r.reshape(-1, r.shape[-1])
            d[:, -1] = 1
            d = d.reshape(-1, d.shape[-1])

        print("s shape : ", s.shape)
        print("o shape : ", o.shape)
        print("a shape : ", a.shape)
        print("r shape : ", r.shape)
        print("d shape : ", d.shape)
        print("==============================")

        ind = np.where(d==1)[0]

        max_epi_ret_in_dataset = -1000
        for indd in range(len(ind)):
            if indd == 0:
                r_temp = r[:ind[indd]+1].sum()
                d_temp = d[:ind[indd]+1].sum()
            else:
                r_temp = r[ind[indd-1]: ind[indd]].sum()
                d_temp = d[ind[indd-1]: ind[indd]].sum()

            if d_temp != 1:
                pdb.set_trace()
            if r_temp > max_epi_ret_in_dataset:
                max_epi_ret_in_dataset = r_temp

        if 'pred_prey' in env_name:
            max_epi_ret_in_dataset = max_epi_ret_in_dataset2

        if 'Walker' in env_name:
            s = s[:, :, 0]
            # r = r[:, :, 0]
            for j in range(s.shape[0]):
                ind = np.where(d[j]==1)[0][0]
                if j == 0:
                    s_part = np.array(s[j, :ind])
                    o_part = np.array(o[j, :ind])
                    a_part = np.array(a[j, :ind])
                    r_part = np.array(r[j, :ind])
                    d_part = np.array(d[j, :ind])
                else:
                    s_part = np.concatenate([s_part, np.array(s[j, :ind])], axis=0)
                    o_part = np.concatenate([o_part, np.array(o[j, :ind])], axis=0)
                    a_part = np.concatenate([a_part, np.array(a[j, :ind])], axis=0)
                    r_part = np.concatenate([r_part, np.array(r[j, :ind])], axis=0)
                    d_part = np.concatenate([d_part, np.array(d[j, :ind])], axis=0)
            s = s_part
            o = o_part
            a = a_part
            r = r_part
            d = d_part


        # s = s.reshape([-1, 1, s.shape[-1]])
        # s = np.repeat(s, n_agents, axis=1)

        if 'Cheetah' in env_name or 'Ant' in env_name or 'Hopper' in env_name:
            s = s.reshape([-1, n_agents, s.shape[-1]])
        else:
            s = s.reshape([-1, 1, s.shape[-1]])
            s = np.repeat(s, n_agents, axis=1)
        
        o = o.reshape([-1, n_agents, o.shape[-1]])
        a = a.reshape([-1, n_agents, n_actions])
        r = r.reshape([-1, 1])
        d = d.reshape([-1, 1])

        print("s shape : ", s.shape)
        print("o shape : ", o.shape)
        print("a shape : ", a.shape)
        print("r shape : ", r.shape)
        print("d shape : ", d.shape)

        

        
        data_size = s.shape[0]
        nonterminal_steps, = np.where(
            np.logical_and(
                np.logical_not(d[:,0]),
                np.arange(data_size) < data_size - 1))
        print('Found %d non-terminal steps out of a total of %d steps.' % (
            len(nonterminal_steps), data_size))
        
        

        self.o = o[nonterminal_steps]
        self.s = s[nonterminal_steps]
        self.a = a[nonterminal_steps]
        self.r = r[nonterminal_steps].reshape(-1, 1)
        self.mask = 1 - d[nonterminal_steps + 1].reshape(-1, 1)
        self.s_next = s[nonterminal_steps + 1]
        self.o_next = o[nonterminal_steps + 1]
        self.a_next = a[nonterminal_steps + 1]
        self.size = self.o.shape[0]

        nonterminal_steps2 = nonterminal_steps-1
        d1, _ = np.where(self.mask==0)
        for j in range(len(nonterminal_steps2)):
            if nonterminal_steps2[j] < 0:
                nonterminal_steps2[j] = 0
            elif nonterminal_steps2[j] in d1:
                nonterminal_steps2[j] = nonterminal_steps2[j] + 1

        # self.o_prev = o[nonterminal_steps2]
        # self.s_prev = s[nonterminal_steps2]
        # self.a_prev = a[nonterminal_steps2]

        return avg_epi_ret_in_dataset, max_epi_ret_in_dataset
         


        

    def load_old(self):
        print('==========Data loading==========')
        data_file = self.data_dir + self.env_name + '.hdf5'
        # data_file = self.data_dir + 'test.hdf5'
        print('Loading from:', data_file)
        f = h5py.File(data_file, 'r')
        s = np.array(f['s'])
        o = np.array(f['o'])
        a = np.array(f['a'])
        r = np.array(f['r'])
        d = np.array(f['d'])
        f.close()

        data_size = s.shape[0]
        nonterminal_steps, = np.where(
            np.logical_and(
                np.logical_not(d[:,0]),
                np.arange(data_size) < data_size - 1))
        print('Found %d non-terminal steps out of a total of %d steps.' % (
            len(nonterminal_steps), data_size))

        self.o = o[nonterminal_steps]
        self.s = s[nonterminal_steps]
        self.a = a[nonterminal_steps]
        self.r = r[nonterminal_steps].reshape(-1, 1)
        self.mask = 1 - d[nonterminal_steps + 1].reshape(-1, 1)
        self.s_next = s[nonterminal_steps + 1]
        self.o_next = o[nonterminal_steps + 1]
        self.a_next = a[nonterminal_steps + 1]
        self.size = self.o.shape[0]