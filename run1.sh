

python run_mujoco.py --seed 0 --pg_norm 1 --clipq 10 --bc 1 --comm 1 --mixer 'nonmono' \
    --offline_ver 2 --scenario 'continuous_pred_prey_3a' --env_name 'particle' --agent_view_radius 0.5 --algo 'FACMAC' --nbc4comm 1
