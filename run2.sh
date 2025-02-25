
# python run_mujoco.py --seed 0 --comm 1 --mixer 'nonmono' \
#     --data_type 'expert' --scenario 'Hopper-v2' --agent_conf '3x1' --algo 'OMIGA' --nbc4comm 1


python run_mujoco.py --seed 0 --pg_norm 0.5 --clipq 1 --bc 1 --comm 1 --mixer 'nonmono' \
    --data_type 'medium-expert' --scenario 'Hopper-v2' --agent_conf '3x1' --algo 'FACMAC' --nbc4comm 1

python run_mujoco.py --seed 1 --pg_norm 0.5 --clipq 1 --bc 1 --comm 1 --mixer 'nonmono' \
    --data_type 'medium-expert' --scenario 'Hopper-v2' --agent_conf '3x1' --algo 'FACMAC' --nbc4comm 1

python run_mujoco.py --seed 2 --pg_norm 0.5 --clipq 1 --bc 1 --comm 1 --mixer 'nonmono' \
    --data_type 'medium-expert' --scenario 'Hopper-v2' --agent_conf '3x1' --algo 'FACMAC' --nbc4comm 1
