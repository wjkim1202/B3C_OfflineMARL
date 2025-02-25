
python run_mujoco.py --seed 2 --pg_norm 1 --clipq 2 --bc 1 --comm 0 --mixer 'nonmono' \
    --data_type 'expert' --scenario 'Ant-v2' --agent_conf '2x4' --algo 'FACMAC'

python run_mujoco.py --seed 2 --pg_norm 0.5 --clipq 2 --bc 1 --comm 0 --mixer 'nonmono' \
    --data_type 'medium-expert' --scenario 'Ant-v2' --agent_conf '2x4' --algo 'FACMAC'

python run_mujoco.py --seed 2 --pg_norm 1 --clipq 2 --bc 1 --comm 0 --mixer 'nonmono' \
    --data_type 'medium' --scenario 'Ant-v2' --agent_conf '2x4' --algo 'FACMAC'

python run_mujoco.py --seed 2 --pg_norm 1 --clipq 2 --bc 1 --comm 0 --mixer 'nonmono' \
    --data_type 'medium-replay' --scenario 'Ant-v2' --agent_conf '2x4' --algo 'FACMAC'


