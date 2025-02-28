import os
import torch, pdb
import numpy as np

from envs.ma_mujoco.multiagent_mujoco.mujoco_multi import MujocoMulti
from envs.env_wrappers import ShareDummyVecEnv
from utils.logger import setup_logger_kwargs, Logger
from utils.util import evaluate
from datasets.offline_dataset import ReplayBuffer
from algos.OMIGA import OMIGA
from algos.FACMAC import FACMAC


import wandb
from tqdm import tqdm


#####
from functools import partial
from envs.particle import Particle
from envs.multiagentenv import MultiAgentEnv
####


def env_fn(env, **kwargs) -> MultiAgentEnv:
    # env_args = kwargs.get("env_args", {})
    return env(**kwargs)

def make_train_env(config):
    def get_env_fn(rank):
        def init_env():
            if config['env_name'] == "mujoco":
                env_args = {"scenario": config['scenario'],
                            "agent_conf": config['agent_conf'],
                            "agent_obsk": config['agent_obsk'],
                            "full_observability": config['full_observability'],
                            "episode_limit": 1000}
                env = MujocoMulti(env_args=env_args)
            elif config['env_name'] == 'particle':
                env_args = {"scenario_name": config['scenario'],
                            "benchmark": False,
                            "state_mode": "all",
                            "agent_view_radius": 0.5,
                            "score_function": "min",
                            "partial_obs": True,
                            "episode_limit": 50}
                args = {}
                args["env_args"] = env_args
                env = partial(env_fn, env=Particle)      

            else:
                print("Can not support the " + config['env_name'] + "environment.")
                raise NotImplementedError
            env.seed(config['seed'])
            return env

        return init_env
    return ShareDummyVecEnv([get_env_fn(0)])


def make_eval_env(config):
    def get_env_fn(rank):
        def init_env():
            if config['env_name'] == "mujoco":
                env_args = {"scenario": config['scenario'],
                            "agent_conf": config['agent_conf'],
                            "agent_obsk": config['agent_obsk'],
                            "full_observability": config['full_observability'],
                            "episode_limit": 1000}
                env = MujocoMulti(env_args=env_args)
            else:
                print("Can not support the " + config['env_name'] + "environment.")
                raise NotImplementedError
            env.seed(config['seed'])
            return env

        return init_env
    return ShareDummyVecEnv([get_env_fn(0)])


def run(config):

    entity_name = ""
    project_name = "B3C"

    if config['algo'] == 'OMIGA':
        algorithm_name = "OMIGA" 
        algorithm_name = algorithm_name + '_a' + str(config['alpha'])
    elif config['algo'] == 'FACMAC':
        algorithm_name = "FACMAC"
        algorithm_name = algorithm_name + '_' + str(config['mixer']) + '_rl' + str(config['coeff_rl']) + '_bc' + str(config['coeff_bc']) + '_clip' + str(config['clipq'])

    if config['comm'] == 1:
        algorithm_name = algorithm_name + '_comm' + str(config['dim_msg'])

    
    if config['full_observability'] == 1:
        env_name = 'fo_' + str(config['scenario']) +'-' + str(config['agent_conf'])
    else:
        env_name = 'po_' + str(config['scenario']) +'-' + str(config['agent_conf']) + '-obsk' + str(config['agent_obsk'])
    env_name = env_name + '_dataset' + str(config['offline_ver'])

        
    if config['wandb']:
        wandb.init(config=args,
                    project=project_name+ '_' + env_name,
                    entity=entity_name,
                    name= algorithm_name,
                    group=algorithm_name,
                    job_type="training",
                    reinit=True)
    

    # Seeding
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed']) 
    if config['env_name'] == 'mujoco':
        env = make_train_env(config)
        eval_env = make_eval_env(config)
        state_dim = env.share_observation_space[0].shape[0]
        if config['full_observability'] == 1:
            state_dim =  env.observation_space[0].shape[0]
        obs_dim = env.observation_space[0].shape[0]
        action_dim = env.action_space[0].shape[0]
        n_agents = len(env.observation_space)

    elif config['env_name'] == 'particle':
        env_args = {"scenario_name": config['scenario'],
                                "benchmark": False,
                                "state_mode": "all",
                                "agent_view_radius": config['agent_view_radius'],
                                "score_function": "min",
                                "partial_obs": True,
                                "episode_limit": 50}
        env = partial(env_fn, env=Particle)(env_args=env_args)
        eval_env = partial(env_fn, env=Particle)(env_args=env_args)
        obs_dim = env.get_obs_size()
        action_dim = 2
        n_agents = 3
        state_dim = obs_dim * n_agents

    print('state_dim:', state_dim, 'action_dim:', action_dim, 'num_agents:', n_agents)

    logger_kwargs = setup_logger_kwargs(env_name, config['seed'])
    logger = Logger(**logger_kwargs)
    logger.save_config(config)

    print("====== obs_dim : ", obs_dim)
    print("====== action_dim : ", action_dim)
    print("====== state_dim : ", state_dim)
    
    # Datasets
    offline_dataset = ReplayBuffer(obs_dim, action_dim, state_dim, n_agents, env_name, config['data_dir'], device=config['device'])    
    avg_epi_ret_in_dataset, max_epi_ret_in_dataset = offline_dataset.load(env_name= env_name, agent_view_radius = config['agent_view_radius'], offline_ver=config['offline_ver'], obsk=config['agent_obsk'], n_agents=n_agents, n_actions=action_dim)

    config['avg_epi_ret_in_dataset'] = avg_epi_ret_in_dataset
    config['max_epi_ret_in_dataset'] = max_epi_ret_in_dataset

    result_logs = {}

    def _eval_and_log(train_result, config):
        print("==========================================")
        print("env_name : ", env_name)
        print("max return : ", config['max_epi_ret_in_dataset'])
        print("avg return : ", config['avg_epi_ret_in_dataset'])
        train_result = {k: v.detach().cpu().numpy() for k, v in train_result.items()}
        print('\n==========Policy testing==========')
        # evaluation via real-env rollout
        ep_r = evaluate(agent, eval_env, config['env_name'])

        train_result.update({'ep_r': ep_r})
        result_log = {'log': train_result, 'step': iteration}
        result_logs[str(iteration)] = result_log

        for k, v in sorted(train_result.items()):
            print(f'- {k:23s}:{v:15.10f}')
        print(f'iteration={iteration}')
        print('\n==========Policy training==========', flush=True)

        return train_result
     
    # Agent
    if config['algo'] == 'OMIGA':
        agent = OMIGA(obs_dim, action_dim, state_dim, n_agents, eval_env, config)
    elif config['algo'] == 'FACMAC':
        agent = FACMAC(obs_dim, action_dim, state_dim, n_agents, eval_env, config)

    # Train
    print('\n==========Start training==========')

    for iteration in tqdm(range(0, config['total_iterations']), ncols=70, desc=config['algo'], initial=1, total=config['total_iterations'], ascii=True, disable=os.environ.get("DISABLE_TQDM", False)):
        o, s, a, r, mask, s_next, o_next, a_next = offline_dataset.sample(config['batch_size'])        
        if config['algo'] == 'OMIGA':
            train_result = agent.train_step(o, s, a, r, mask, s_next, o_next, a_next)
        else:
            train_result = agent.train_step(o, s, a, r, mask, s_next, o_next, a_next, config['coeff_rl'],
                                            config['clipq'], config['max_epi_ret_in_dataset'], bc=config['bc'])

        if iteration % config['log_iterations'] == 0:
            train_result = _eval_and_log(train_result, config)
            if config['wandb'] == True:
                wandb.log(train_result)

    # Save results
    logger.save_result_logs(result_logs)

    env.close()
    eval_env.close()

if __name__ == "__main__":
    from configs.config import get_parser
    args = get_parser().parse_args() 
    run(vars(args))