
python run_mujoco.py --seed 2 --pg_norm 16 --clipq 0.1 --bc 1 --comm 0 --mixer 'nonmono' \
    --data_type 'expert' --scenario 'Hopper-v2' --agent_conf '3x1' --algo 'FACMAC'

python run_mujoco.py --seed 2 --pg_norm 1 --clipq 0.1 --bc 1 --comm 0 --mixer 'nonmono' \
    --data_type 'medium-expert' --scenario 'Hopper-v2' --agent_conf '3x1' --algo 'FACMAC'

python run_mujoco.py --seed 2 --pg_norm 8 --clipq 0.1 --bc 1 --comm 0 --mixer 'nonmono' \
    --data_type 'medium' --scenario 'Hopper-v2' --agent_conf '3x1' --algo 'FACMAC'

python run_mujoco.py --seed 2 --pg_norm 0.25 --clipq 2 --bc 1 --comm 0 --mixer 'nonmono' \
    --data_type 'medium-replay' --scenario 'Hopper-v2' --agent_conf '3x1' --algo 'FACMAC'


