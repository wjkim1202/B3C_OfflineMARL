

############### Partially observable Mujoco envs
############### --scenario 'HalfCheetah-v2' --agent_conf '6x1' --agent_obsk 0 --full_observability 0 --offline_ver 0/1/2/3/4/5 (e/m1/m2/e-m1/e-m2/m1-m2) 
############### --scenario 'HalfCheetah-v2' --agent_conf '6x1' --agent_obsk 1 --full_observability 0 --offline_ver 0/1/2/3/4/5 (e/m1/m2/e-m1/e-m2/m1-m2) 
############### --scenario 'manyagent_swimmer' --agent_conf '5x2' --agent_obsk 0 --full_observability 0 --offline_ver 0/1/2/3 (e/m1/m2/e-m1/e-m2/m1-m2)  

python run_mujoco.py --seed 0 --coeff_rl 1 --clipq 1 --bc 1 --mixer 'nonmono' \
    --offline_ver 5 --scenario 'HalfCheetah-v2' --agent_conf '6x1' --algo 'FACMAC' --agent_obsk 0 --full_observability 0
# python run_mujoco.py --seed 0 --coeff_rl 1 --clipq 1 --bc 1 --mixer 'nonmono' \
#    --offline_ver 5 --scenario 'HalfCheetah-v2' --agent_conf '6x1' --algo 'FACMAC' --agent_obsk 1 --full_observability 0



############### Fully observable Mujoco envs
############### --scenario 'HalfCheetah-v2' --agent_conf '6x1' --full_observability 1 --offline_ver 0/1/2/3 (expert/medium-expert/medium/medium-replay)
############### --scenario 'Hopper-v2' --agent_conf '3x1' --full_observability 1 --offline_ver 0/1/2/3 (expert/medium-expert/medium/medium-replay)
############### --scenario 'Ant-v2' --agent_conf '2x4' --full_observability 1 --offline_ver 0/1/2/3 (expert/medium-expert/medium/medium-replay)

# python run_mujoco.py --seed 0 --coeff_rl 1 --clipq 1 --bc 1 --mixer 'nonmono' \
#     --offline_ver 2 --scenario 'HalfCheetah-v2' --agent_conf '6x1' --algo 'FACMAC' --agent_obsk -100 --full_observability 1
# python run_mujoco.py --seed 0 --coeff_rl 1 --clipq 1 --bc 1 --mixer 'nonmono' \
#     --offline_ver 2 --scenario 'Hopper-v2' --agent_conf '3x1' --algo 'FACMAC' --agent_obsk -100 --full_observability 1
# python run_mujoco.py --seed 0 --coeff_rl 1 --clipq 1 --bc 1 --mixer 'nonmono' \
#     --offline_ver 2 --scenario 'Ant-v2' --agent_conf '2x4' --algo 'FACMAC' --agent_obsk -100 --full_observability 1

