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


class Actor(nn.Module):
    def __init__(self, num_state, num_action, num_agent, num_hidden, device):
        super(Actor, self).__init__()

        self.device = device
        self.fc1 = nn.Linear(num_state + num_agent, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.mu_head = nn.Linear(num_hidden, num_action)
        self.sigma_head = nn.Linear(num_hidden, num_action)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float).to(self.device)
        x = F.relu(self.fc1(x))
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
        x = F.relu(self.fc2(x))
        mu = self.mu_head(x)
        mu = torch.clamp(mu, MEAN_MIN, MEAN_MAX)
        mu = torch.tanh(mu)
        return mu


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


class QMixer(nn.Module):
    def __init__(self, num_state, num_action, num_agent, num_hidden, device):
        super(QMixer, self).__init__()
        
        self.args = device
        self.n_agents = num_agent
        self.state_dim = num_state
        self.embed_dim = 32 # self.args.mixing_embed_dim
        hypernet_embed = 64 #self.args.hypernet_embed

        self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                        nn.ReLU(),
                                        nn.Linear(hypernet_embed, self.embed_dim * self.n_agents))
        self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                        nn.ReLU(),
                                        nn.Linear(hypernet_embed, self.embed_dim))

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, states):
        states = states[:, 0]

        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)
        # First layer
        w1 = torch.abs(self.hyper_w_1(states))

        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)

        
        # Second layer
        w_final = torch.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)
        # Compute final output
        y = torch.bmm(hidden, w_final) + v
        # Reshape and return        
        q_tot = y.view(bs, -1, 1)
        return q_tot


class QMixer_nonmonotonic(nn.Module):
    def __init__(self, num_state, num_action, num_agent, num_hidden, device):
        super(QMixer_nonmonotonic, self).__init__()
        
        self.args = device
        self.n_agents = num_agent
        self.state_dim = num_state
        self.embed_dim = 32 # self.args.mixing_embed_dim
        hypernet_embed = 64 #self.args.hypernet_embed

        self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                        nn.ReLU(),
                                        nn.Linear(hypernet_embed, self.embed_dim * self.n_agents))
        self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                        nn.ReLU(),
                                        nn.Linear(hypernet_embed, self.embed_dim))

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, states):
        states = states[:, 0]

        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)

        
        agent_qs = agent_qs.view(-1, 1, self.n_agents)
        # First layer
        w1 = self.hyper_w_1(states)

        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)

        
        # Second layer
        w_final = self.hyper_w_final(states)
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)
        # Compute final output
        y = torch.bmm(hidden, w_final) + v
        # Reshape and return        
        q_tot = y.view(bs, -1, 1)
        return q_tot


class MixNet(nn.Module):
    def __init__(self, num_state, num_action, num_agent, num_hidden, device):
        super(MixNet, self).__init__()
        self.device = device
        self.state_shape = num_state * num_agent  # concat state from agents

        self.n_agents = num_agent
        self.hyper_hidden_dim = num_hidden
        self.num_action = num_action

        self.f_v = nn.Linear(self.state_shape, num_hidden)
        self.w_v = nn.Linear(num_hidden, num_agent)
        self.b_v = nn.Linear(num_hidden, 1)

    def forward(self, states):
        batch_size = states.size(0)
        states = torch.cat([states[:, j, :] for j in range(self.n_agents)], dim=-1)
        states = states.reshape(-1, self.state_shape)
        x = self.f_v(states)
        w = self.w_v(F.relu(x)).reshape(batch_size, self.n_agents, 1)
        b = self.b_v(F.relu(x)).reshape(batch_size, 1, 1)
        
        return torch.abs(w), b

