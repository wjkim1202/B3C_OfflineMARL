import torch.nn as nn
import torch, pdb
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal

MEAN_MIN = -9.0
MEAN_MAX = 9.0
LOG_STD_MIN = -5
LOG_STD_MAX = 2
EPS = 1e-7


class Actor_comm(nn.Module):
    def __init__(self, num_state, num_action, num_agent, num_hidden, device, num_msg):
        super(Actor_comm, self).__init__()

        self.device = device
        self.n_msg = num_msg
        self.n_agents = num_agent
        self.fc1 = nn.Linear(num_state + num_agent, num_hidden)
        self.fc2 = nn.Linear(num_hidden+num_msg, num_hidden)

        self.mu_head = nn.Linear(num_hidden, num_action)
        self.sigma_head = nn.Linear(num_hidden, num_action)

        self.fc_key = nn.Linear(num_hidden, num_msg)
        self.msg_cat = MultiHeadAttention(n_head=4, d_model=num_msg, d_k=num_msg, d_v=num_msg)

    def get_param_groups(self):
        msg_params = list(self.fc_key.parameters()) + list(self.msg_cat.parameters()) 
        policy_params = list(self.fc1.parameters()) + list(self.fc2.parameters()) + list(self.sigma_head.parameters())  + list(self.mu_head.parameters())
        return msg_params, policy_params

    def forward(self, x, prev_msg=None, only_msg=False):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float).to(self.device)
        x = F.relu(self.fc1(x))

        key = self.fc_key(x).reshape(-1, self.n_agents, self.n_msg)
        encoded_msgs_, _ = self.msg_cat(key, key, key)
        encoded_msgs_ = encoded_msgs_.reshape(-1, self.n_msgs)
        
        x = torch.cat([x, encoded_msgs_], dim=-1)
        x = F.relu(self.fc2(x))

        mu = self.mu_head(x)
        log_sigma = self.sigma_head(x)

        log_sigma = torch.clamp(log_sigma, LOG_STD_MIN, LOG_STD_MAX)
        sigma = torch.exp(log_sigma)

        a_distribution = Normal(mu, sigma)
        action = a_distribution.rsample()

        logp_pi = a_distribution.log_prob(action).sum(axis=-1)
        logp_pi -= (2 * (np.log(2) - action - F.softplus(-2 * action))).sum(axis=-1)

        action = torch.tanh(action)
        return action, logp_pi, a_distribution

    def get_log_density(self, x, y):
        x = F.relu(self.fc1(x))

        key = self.fc_key(x).reshape(-1, self.n_agents, self.n_msg)
        encoded_msgs_, _ = self.msg_cat(key, key, key)
        encoded_msgs_ = encoded_msgs_.reshape(-1, self.n_agents, self.n_msg)
        x = torch.cat([x, encoded_msgs_], dim=-1)
        x = F.relu(self.fc2(x))

        mu = self.mu_head(x)
        log_sigma = self.sigma_head(x)

        y = torch.clamp(y, -1. + EPS, 1. - EPS)
        y = torch.atanh(y)

        mu = torch.clamp(mu, MEAN_MIN, MEAN_MAX)
        log_sigma = torch.clamp(log_sigma, LOG_STD_MIN, LOG_STD_MAX)
        sigma = torch.exp(log_sigma)

        a_distribution = Normal(mu, sigma)
        logp_pi = a_distribution.log_prob(y).sum(axis=-1)
        logp_pi -= (2 * (np.log(2) - y - F.softplus(-2 * y))).sum(axis=-1)
        return logp_pi

    def get_action(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float).to(self.device)
        x = F.relu(self.fc1(x))

        key = self.fc_key(x).reshape(-1, self.n_agents, self.n_msg)
        encoded_msgs_, _ = self.msg_cat(key, key, key)
        encoded_msgs_ = encoded_msgs_.reshape(-1, self.n_agents, self.n_msg)
        
        x = torch.cat([x, encoded_msgs_], dim=-1)
        x = F.relu(self.fc2(x))

        mu = self.mu_head(x)
        log_sigma = self.sigma_head(x)

        log_sigma = torch.clamp(log_sigma, LOG_STD_MIN, LOG_STD_MAX)
        sigma = torch.exp(log_sigma)

        a_distribution = Normal(mu, sigma)
        action = a_distribution.rsample()
        action = torch.tanh(action)
        return action
    


    def get_deterministic_action(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float).to(self.device)
        x = F.relu(self.fc1(x))
        # x = x.reshape(-1, self.n_agents, x.shape[-1])

        key = self.fc_key(x).reshape(-1, self.n_agents, self.n_msg)
        encoded_msgs_, _ = self.msg_cat(key, key, key)
        encoded_msgs_ = encoded_msgs_.reshape(-1, self.n_agents, self.n_msg)

        if len(x.shape) == 2:
            encoded_msgs_ = encoded_msgs_.squeeze(0)
        x = torch.cat([x, encoded_msgs_], dim=-1)
        x = F.relu(self.fc2(x))

        mu = self.mu_head(x)
        mu = torch.clamp(mu, MEAN_MIN, MEAN_MAX)
        mu = torch.tanh(mu)

        # msg = torch.tanh(self.msg_head(x))

        return mu#, msg


class V_critic(nn.Module):
    def __init__(self, num_state, num_agent, num_hidden, device):
        super(V_critic, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(num_state + num_agent, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.state_value = nn.Linear(num_hidden, 1)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float).to(self.device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        v = self.state_value(x)
        return v
    
    def v(self, obs, agent_id):
        x = torch.cat([obs, agent_id], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.state_value(x)
        return q


class Q_critic(nn.Module):
    def __init__(self, num_state, num_action, num_agent, num_hidden, device):
        super(Q_critic, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(num_state + num_action + num_agent, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.state_value = nn.Linear(num_hidden, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.state_value(x)
        return q

    def q(self, obs, action, agent_id):
        x = torch.cat([obs, action, agent_id], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.state_value(x)
        return q


class Q_critic_comm(nn.Module):
    def __init__(self, num_state, num_action, num_msg,  num_agent, num_hidden, device):
        super(Q_critic_comm, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(num_state + num_action + num_agent, num_hidden)
        self.fc2 = nn.Linear(num_hidden + num_msg*num_agent, num_hidden)
        self.state_value = nn.Linear(num_hidden, 1)

    def forward(self, x, msg):
        x = F.relu(self.fc1(x))
        
        x = torch.cat([x, msg], dim=-1)

        x = F.relu(self.fc2(x))
        q = self.state_value(x)
        return q

    def q(self, obs, action, msg, agent_id):
        x = torch.cat([obs, action, agent_id], dim=-1)
        x = F.relu(self.fc1(x))

        x = torch.cat([x, msg])

        x = F.relu(self.fc2(x))
        q = self.state_value(x)
        return q


class MixNet(nn.Module):
    def __init__(self, num_state, num_action, num_agent, num_hidden, device):
        super(MixNet, self).__init__()
        self.device = device
        self.state_shape = num_state * num_agent  # concat state from agents
        self.state_shape = 17  # concat state from agents
        self.n_agents = num_agent
        self.hyper_hidden_dim = num_hidden
        self.num_action = num_action

        self.f_v = nn.Linear(self.state_shape, num_hidden)
        self.w_v = nn.Linear(num_hidden, num_agent)
        self.b_v = nn.Linear(num_hidden, 1)

    def forward(self, states):
        batch_size = states.size(0)
        ####################
        states = states[:, 0, :]
        ####################

        # states = torch.cat([states[:, j, :] for j in range(self.n_agents)], dim=-1)
        # states = states.reshape(-1, self.state_shape)
        x = self.f_v(states)
        w = self.w_v(F.relu(x)).reshape(batch_size, self.n_agents, 1)
        b = self.b_v(F.relu(x)).reshape(batch_size, 1, 1)
        
        return torch.abs(w), b


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        # self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        
        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        
        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        
        print("q shape 1; ", q.shape)
        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        # q = self.dropout(self.fc(q))
        print("q shape 2; ", q.shape)
        q = self.fc(q)
        print("q shape 3; ", q.shape)
        q += residual

        q = self.layer_norm(q)
        print("q shape 4; ", q.shape)
        pdb.set_trace()
        return q, attn

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        # self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        # attn = self.dropout(F.softmax(attn, dim=-1))
        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, v)

        return output, attn
    
