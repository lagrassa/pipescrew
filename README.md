#  Plan or parameterized learning

## Overview

planorparam is a experimental library for autonomously deciding whether to use a planner, 
update a model, or learn a model free policy. All Python code is included in `scripts`
* `models` -  models of object and robot
* `scripts/agent` - Autonomous reasoning about planning and learning in simulation 
* `scripts/env` - Test environments
* `scripts/modelfree` -  Library and wrappers for model-free learning
* `scripts/models` - models saved from learning
* `scripts/old` - deprecated code, ignore
* `scripts/planning` - conCERRT implementation, in the future more home-implemented planning algorithms
* `scripts/plotting` - plotting utilities
* `scripts/real_robot` - catch-all for any code that will run on the real robot
* `scripts/test_module` - tests, incomplete but also provides some example code
* `scripts/utils` - library functions 




## Building
Install using `pip install -e .` at the `scripts` folder. 

In order to run the planners, clone the `motion_planners` repo from https://github.com/caelan/motion-planners and add it to your Python path. 
There are some hidden dependencies, though most should be listed in setup.py. If you find any issues, please use the Github issues feature. 

## Running
To run the peg insert environment, run `peginsertenv.py` from the `scripts/env` directory. The `__main__` method shows example usage. `PegInsertEnv` follows the OpenAI gym conventions. 
