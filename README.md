#  Plan or parameterized learning

## Overview

planorparam is a experimental library for autonomously deciding whether to use a planner, 
update a model, or learn a model free policy. All Python code is included in `planorparam`
* `models` -  models of object and robot
* `planorparam/agent` - Autonomous reasoning about planning and learning in simulation 
* `planorparam/env` - Test environments
* `planorparam/modelfree` -  Library and wrappers for model-free learning
* `planorparam/models` - models saved from learning
* `planorparam/old` - deprecated code, ignore
* `planorparam/planning` - conCERRT implementation, in the future more home-implemented planning algorithms
* `planorparam/plotting` - plotting utilities
* `planorparam/real_robot` - catch-all for any code that will run on the real robot
* `planorparam/test_module` - tests, incomplete but also provides some example code
* `planorparam/utils` - library functions 




## Building
Install using `pip install -e .` at the `planorparam` folder. 

In order to run the planners, clone the `motion_planners` repo from https://github.com/caelan/motion-planners and add it to your Python path. 
There are some hidden dependencies, though most should be listed in setup.py. If you find any issues, please use the Github issues feature. 

## Running
To run the peg insert environment, run `peginsertenv.py` from the `planorparam/env` directory. The `__main__` method shows example usage. `PegInsertEnv` follows the OpenAI gym conventions. 

To train the autoencoder, collect data using `realpeginsert.py` with the training=True flag, and then run `planorparam/test_module/test_realautoencoder.py`
To run the imitation learning training, run 
`planorparam/test_module/test_behaviour_cloning` from the `planorparam/real_robot` directory. The autoencoder needs to be trained first in order for the behaviour cloning to work. 
