# B3C: A Minimalist Approach to Offline Multi-Agent Reinforcement Learning

This repository is the implementation of "[B3C: A Minimalist Approach to Offline Multi-Agent Reinforcement Learning](https://arxiv.org/pdf/2501.18138)". 


This repository is based on the implementations of [OMIGA](https://github.com/ZhengYinan-AIR/OMIGA) for multi-agent Mujoco environments and [CFCQL](https://github.com/thu-rllab/CFCQL) for multi-agent particle environments.


## How to run
``` Bash
python run_mujoco.py --seed 0 --coeff_rl 1 --clipq 1 --bc 1 --mixer 'nonmono' --offline_ver 2 --scenario 'HalfCheetah-v2' --agent_conf '6x1' --algo 'FACMAC' --agent_obsk -100 --full_observability 1
```


