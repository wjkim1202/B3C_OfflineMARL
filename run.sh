
#python run_mujoco.py --scenario 'Hopper-v2' --agent_conf '3x1' --data_type 'expert' --algo 'OMIGA'
#python run_mujoco.py --scenario 'Hopper-v2' --agent_conf '3x1' --data_type 'expert' --algo 'FACMAC' --clipq 1 --comm 0
#python run_mujoco.py --scenario 'Hopper-v2' --agent_conf '3x1' --data_type 'expert' --algo 'FACMAC' --clipq 1 --comm 1
#
#python run_mujoco.py --scenario 'Hopper-v2' --agent_conf '3x1' --data_type 'medium' --algo 'OMIGA'
#python run_mujoco.py --scenario 'Hopper-v2' --agent_conf '3x1' --data_type 'medium' --algo 'FACMAC' --clipq 1 --comm 0
#python run_mujoco.py --scenario 'Hopper-v2' --agent_conf '3x1' --data_type 'medium' --algo 'FACMAC' --clipq 1 --comm 1
#
#python run_mujoco.py --scenario 'Hopper-v2' --agent_conf '3x1' --data_type 'medium-expert' --algo 'OMIGA'
#python run_mujoco.py --scenario 'Hopper-v2' --agent_conf '3x1' --data_type 'medium-expert' --algo 'FACMAC' --clipq 1 --comm 0
#python run_mujoco.py --scenario 'Hopper-v2' --agent_conf '3x1' --data_type 'medium-expert' --algo 'FACMAC' --clipq 1 --comm 1
#
# python run_mujoco.py --scenario 'Hopper-v2' --agent_conf '3x1' --data_type 'medium-replay' --algo 'OMIGA'
#python run_mujoco.py --scenario 'Hopper-v2' --agent_conf '3x1' --data_type 'medium-replay' --algo 'FACMAC' --clipq 1 --comm 0
#python run_mujoco.py --scenario 'Hopper-v2' --agent_conf '3x1' --data_type 'medium-replay' --algo 'FACMAC' --clipq 1 --comm 1



#python run_mujoco.py --scenario 'Ant-v2' --agent_conf '2x4' --data_type 'expert' --algo 'OMIGA' --seed 0
#python run_mujoco.py --scenario 'Ant-v2' --agent_conf '2x4' --data_type 'expert' --algo 'FACMAC' --clipq 1 --comm 0 --seed 0
#python run_mujoco.py --scenario 'Ant-v2' --agent_conf '2x4' --data_type 'expert' --algo 'FACMAC' --clipq 1 --comm 1 --seed 0
#
#python run_mujoco.py --scenario 'Ant-v2' --agent_conf '2x4' --data_type 'medium-expert' --algo 'OMIGA'
#python run_mujoco.py --scenario 'Ant-v2' --agent_conf '2x4' --data_type 'medium-expert' --algo 'FACMAC' --clipq 1 --comm 0 --seed 0
#python run_mujoco.py --scenario 'Ant-v2' --agent_conf '2x4' --data_type 'medium-expert' --algo 'FACMAC' --clipq 1 --comm 1 --seed 0
#

#python run_mujoco.py --scenario 'Ant-v2' --agent_conf '2x4' --data_type 'medium' --algo 'OMIGA' --seed 0
#python run_mujoco.py --scenario 'Ant-v2' --agent_conf '2x4' --data_type 'medium' --algo 'FACMAC' --clipq 1 --comm 0 --seed 0
#python run_mujoco.py --scenario 'Ant-v2' --agent_conf '2x4' --data_type 'medium' --algo 'FACMAC' --clipq 1 --comm 1 --seed 0


python run_mujoco.py --scenario 'Ant-v2' --agent_conf '2x4' --data_type 'medium-replay' --algo 'OMIGA' --seed 0
python run_mujoco.py --scenario 'Ant-v2' --agent_conf '2x4' --data_type 'medium-replay' --algo 'FACMAC' --clipq 1 --comm 0 --seed 0
python run_mujoco.py --scenario 'Ant-v2' --agent_conf '2x4' --data_type 'medium-replay' --algo 'FACMAC' --clipq 1 --comm 1 --seed 0



# expert    medium-expert    medium   medium-replay




# python run_mujoco.py --comm 0 --offline_ver 3 --algo 'FACMAC' --clipq 1
# python run_mujoco.py --comm 1 --offline_ver 3 --algo 'FACMAC' --clipq 1
# python run_mujoco.py --comm 0 --offline_ver 4 --algo 'FACMAC' --clipq 1
# python run_mujoco.py --comm 1 --offline_ver 4 --algo 'FACMAC' --clipq 1
# python run_mujoco.py --comm 0 --offline_ver 2 --algo 'FACMAC' --clipq 1
# python run_mujoco.py --comm 1 --offline_ver 2 --algo 'FACMAC' --clipq 1