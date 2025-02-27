import os
import copy, pdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.network import Actor, V_critic, Q_critic, MixNet, QMixer, QMixer_nonmonotonic
from networks.network_comm import Actor_comm#, V_critic, Q_critic, MixNet


class FACMAC(object):
    def __init__(self, observation_spec, action_spec, state_spec,  num_agent, eval_env, config):
        self._gamma = config['gamma']
        self._tau = config['tau']
        self._hidden_sizes = config['hidden_sizes']
        self._mix_hidden_sizes = config['mix_hidden_sizes']
        self._batch_size = config['batch_size']
        self._lr = config['lr']
        self._grad_norm_clip = config['grad_norm_clip']
        self._num_agent = num_agent
        self._device = config['device']
        self._eval_env = eval_env
        self._iteration = 0
        self._optimizers = dict()

        self._comm = config['comm']
        self._dim_msg = config['dim_msg']

        # q-network and mix-network
        self._q_network = Q_critic(observation_spec, action_spec, num_agent, self._hidden_sizes, self._device).to(self._device)
        self._q_target_network = copy.deepcopy(self._q_network)

        self._mixer = config['mixer']
        if config['mixer'] == 'nonmono':
            self._mix_network = QMixer_nonmonotonic(state_spec, action_spec, num_agent, self._mix_hidden_sizes, self._device).to(self._device)
            self._mix_target_network = copy.deepcopy(self._mix_network)
            self._q_param = list(self._q_network.parameters()) + list(self._mix_network.parameters())
        elif config['mixer'] == 'mono':
            self._mix_network = QMixer(state_spec, action_spec, num_agent, self._mix_hidden_sizes,
                                                    self._device).to(self._device)
            self._mix_target_network = copy.deepcopy(self._mix_network)
            self._q_param = list(self._q_network.parameters()) + list(self._mix_network.parameters())
        elif config['mixer'] == 'vdn':
            self._q_param = list(self._q_network.parameters()) #+ list(self._mix_network.parameters())

        self._optimizers['q'] = torch.optim.Adam(self._q_param, self._lr)
        
        # policy-network
        if self._comm == 1:
            print("============= COMMUNICATION ============= ")
            self._policy_network = Actor_comm(observation_spec, action_spec, num_agent, self._hidden_sizes, self._device, self._dim_msg).to(self._device)
        else:
            self._policy_network = Actor(observation_spec, action_spec, num_agent, self._hidden_sizes, self._device).to(self._device)

        self._optimizers['policy'] = torch.optim.Adam(self._policy_network.parameters(), self._lr)
            

    
    def q_loss(self, o_with_a_id, s, o_next_with_a_id, s_next, r, mask, result={}, clipq=0, max_return=1e5):
        q_values = self._q_network(o_with_a_id)
        tar_q_values = self._q_target_network(o_next_with_a_id)

        if self._mixer == 'vdn':
            q_total = q_values.sum(dim=1)
            tar_q_total = tar_q_values.sum(dim=1)
        else:
            q_total = self._mix_network(q_values, s).squeeze(-1)
            tar_q_total = self._mix_target_network(tar_q_values, s_next).squeeze(-1)

        if clipq != 0:
            expected_q_total = r+ self._gamma * mask * torch.clip(tar_q_total.detach(), max=max_return * clipq)            
        else:
            expected_q_total = r+ self._gamma * mask * tar_q_total.detach()
        q_loss = ((q_total - expected_q_total.detach())**2).mean()

        result.update({
            'q_loss': q_loss,
            'expected_q_total': expected_q_total.mean(),
            'q_total': q_total.mean(),
            'q_values1': q_values[:,0,:].mean(),
            'q_values2': q_values[:,1,:].mean(),
        })

        return result

    def policy_loss(self, o_with_a_id, s, a, a_from_policy, pg_norm , pg_norm_bc=1, bc=1,  result={}):
        
        #config['nbc4comm'] 
        q_values = self._q_network(o_with_a_id)
        if self._mixer == 'vdn':
            q_total = q_values.sum(dim=1)
        else:
            q_total = self._mix_network(q_values, s).squeeze(-1)
        policy_loss = -q_total.mean()
        if bc != 0:
            if pg_norm >= 0:
                policy_loss = policy_loss/abs(policy_loss.item())*pg_norm
                bc_loss = ((a-a_from_policy)**2).mean() * pg_norm_bc
            else:
                pdb.set_trace()
                bc_loss = ((a - a_from_policy) ** 2)*abs(pg_norm)
                policy_loss = policy_loss + bc_loss.mean()


        result.update({
            'policy_loss': policy_loss+bc_loss,
        })
        
        return result
    
    def train_step(self, o, s, a, r, mask, s_next, o_next, a_next, rl_coeff, bc_coeff, clipq=False, max_return=1e5, bc=1):
        # Shared network values
        one_hot_agent_id = torch.eye(self._num_agent).expand(o.shape[0], -1, -1).to(self._device)
        o_with_id = torch.cat((o, one_hot_agent_id), dim=-1)
        o_with_a_id = torch.cat((o, a, one_hot_agent_id), dim=-1)

        o_next_with_id = torch.cat((o_next, one_hot_agent_id), dim=-1)
        a_next_from_policy = self._policy_network.get_action(o_next_with_id)
        o_next_with_a_id = torch.cat((o_next, a_next_from_policy, one_hot_agent_id), dim=-1)

        # q_loss

        loss_result = self.q_loss(o_with_a_id, s, o_next_with_a_id, s_next, r, mask, result={}, clipq=clipq, max_return=max_return)

        o_with_id_p = torch.cat((o, one_hot_agent_id), dim=-1)
        a_from_policy = self._policy_network.get_action(o_with_id_p)
        o_with_a_id = torch.cat((o, a_from_policy, one_hot_agent_id), dim=-1)

        loss_result = self.policy_loss(o_with_a_id, s, a, a_from_policy, rl_coeff, bc_coeff, result=loss_result)

        self._optimizers['policy'].zero_grad()
        loss_result['policy_loss'].backward()
        nn.utils.clip_grad_norm_(self._policy_network.parameters(), self._grad_norm_clip)
        self._optimizers['policy'].step()
        

        self._optimizers['q'].zero_grad()
        loss_result['q_loss'].backward()
        nn.utils.clip_grad_norm_(self._q_param, self._grad_norm_clip)
        self._optimizers['q'].step()


        # soft update
        for param, target_param in zip(self._q_network.parameters(), self._q_target_network.parameters()):
            target_param.data.copy_(self._tau * param.data + (1 - self._tau) * target_param.data)
        if self._mixer != 'vdn':
            for param, target_param in zip(self._mix_network.parameters(), self._mix_target_network.parameters()):
                target_param.data.copy_(self._tau * param.data + (1 - self._tau) * target_param.data)

        self._iteration += 1
        return loss_result

    def step(self, o, particle=False):
        # o = torch.from_numpy(o).to(self._device)
        # one_hot_agent_id = torch.eye(self._num_agent).expand(o.shape[0], -1, -1).to(self._device)
        o = torch.from_numpy(o).to(self._device)
        if not particle:
            one_hot_agent_id = torch.eye(self._num_agent).expand(o.shape[0], -1, -1).to(self._device)
        else:
            one_hot_agent_id = torch.eye(self._num_agent).expand(o.shape[0], -1).to(self._device)
        o_with_id = torch.cat((o, one_hot_agent_id), dim=-1)
        action = self._policy_network.get_deterministic_action(o_with_id)

        return action.detach().cpu()