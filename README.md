# variance reduction with VR-SCP

This repo contains the code of the paper "Efficiently Escaping Saddle Points for Non-Convex Policy Optimization" based on the garage code [repository](https://github.com/rlworkgroup/garage) verion 2020.9.0

[Garage docs](https://garage.readthedocs.io/en/latest/)


### Main Required libraries
 `gym 0.17.2` and `MuJoCo 2.0.2.0`
 
 We are using mujoco 2.0.2.0 which can be installed from [here](https://github.com/openai/mujoco-py).

## Installation
- Download the repo
- Activate the enviroment:
`conda activate myenv`
- Follow garage installation guide for developers and finally go to the garage directory and type(note that you should install our modified garage library):
`pip install -e '.[all,dev]'`
- Finally install the rest of requirements: `pip install -r requirements.txt`

* * *

### Examples
Find our experiments in the folder "examples"
and then run the python file you want by `python [example_name].py`

### Main algorithm:
`vr_scrn.py`: the implementation of our algorithm that uses our optimizer in garage/src/garage/torch/algos/.


`VRSCRNOptimizer.py`: the implementation of our optimizer in garage/src/garage/torch/optimizers/. 


`SoftmaxMLPPolicy.py`: the implementation of our softmax policy for discreet action space in garage/src/garage/torch/policies/. 

