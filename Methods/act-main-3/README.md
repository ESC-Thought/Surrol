### Install the training dependencies
`cd /rl/act-main-3/detr/`
`pip install -e .`

### Setup the configs
1. In constant, replace the `DATA_DIR` to your data folder.
2. If your dataset consists of waypoints, replace the `waypoints` with the waypoint numbers in your database.
3. In the train.sh, replace the task name with your task name. 

### Run the training
under /rl/act-main-3/, run `bash train.sh` to start training. 

### Notes
You may encounter some errors like missing some modules (i.e., h5py). Just `pip install xxx` will be fine. 

### Eval
the eval code is located in `/rl/eval.py`, make sure you replace the model path: self.model_path['whole'] to your model path. And you can specify the folder to save the rollouts with: self.video_dir.
