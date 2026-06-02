# SKJ-SurRoL-Development
For SurRoL Development 

### 1. Install SurRoL Environment with 
`pip install -e .`
you may see something like package installation failure, check if the surrol is successfully installed with: `pip show surrol`
if it is successfully installed, move on to next step.
### 2. Install specific modules:
`pip install pybullet==3.2.7 gym==0.23.1 numpy==1.24.1`
### 3. Install detr for imitation learning policy
`cd rl/act-main-3`
`cd detr`
`pip install -e .`
check if it is successfully installed with pip show detr
### 4. run the data collection to test if the env has been successfully installed.
`cd surrol/data`
`python data_generation_foveal.py --env BiPegTransfer-v4`
you can also search for other tasks in the surrol/gym/__init__.py 

It is expected to encounter some errors or failures during this setup procedure, you can seek for my help if you are stuck for a long time. 

Tips: Try to use AI tools for debugging.
