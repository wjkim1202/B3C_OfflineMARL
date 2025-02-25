import argparse

def boolean(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='mujoco')  # 'particle' 'mujoco'
    parser.add_argument('--scenario', type=str, default='continuous_pred_prey_3a', help="Which mujoco task to run on")
    # HalfCheetah-v2  Walker2d-v2 continuous_pred_prey_3a
    parser.add_argument('--agent_conf', type=str, default='6x1')
    # 6x1
    parser.add_argument('--agent_obsk', type=int, default=0)
    parser.add_argument('--offline_ver', type=int, default=0)
    parser.add_argument('--comm', type=int, default=0)
    parser.add_argument('--clipq', type=float, default=1)
    
    parser.add_argument('--dim_msg', type=int, default=8)
    
    parser.add_argument('--data_type', type=str, default='expert')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--algo', default='FACMAC', type=str)  # OMIGA  FACMAC
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--wandb', default=True, type=boolean)
    parser.add_argument('--data_dir', default='./data/', type=str)
    
    parser.add_argument('--total_iterations', default=int(1e6), type=int)
    parser.add_argument('--log_iterations', default=int(5e4), type=int)

    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--tau', default=0.005, type=float)
    parser.add_argument('--hidden_sizes', default=256)
    parser.add_argument('--mix_hidden_sizes', default=64)
    parser.add_argument('--vae_hidden_sizes', default=64)   
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=5e-4, type=float)
    parser.add_argument('--grad_norm_clip', default=1.0, type=float)
    parser.add_argument('--pg_norm', default=1.0, type=float)

    parser.add_argument('--nbc4comm', default=0, type=int)
    


    ######## for particle ###########
    parser.add_argument('--state_mode', type=str, default='all')  # 'particle'
    parser.add_argument('--score_function', type=str, default='min')  # 'particle'
    parser.add_argument('--agent_view_radius', default=0.5, type=float)
    parser.add_argument('--partial_obs', default=True, type=boolean)
    parser.add_argument('--benchmark', default=False, type=boolean)
    parser.add_argument('--episode_limit', type=int, default=50)




    ################################
    parser.add_argument('--mixer', type=str, default='nonmono')  # 'nonmono'  'mono'
    parser.add_argument('--bc', type=float, default=1)
    parser.add_argument('--alpha', default=10, type=float)


    

    return parser
