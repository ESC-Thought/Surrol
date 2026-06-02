class args:
    def __init__(self):
        self.num_epochs = 30
        self.ckpt_dir = '/research/d1/gds/kjshi/IROS_SurRoL/rl/act-main-3/experiments/sub_policies/grasp/20250227_1442_bi_peg_transfer_grasp_sub_nodepth'
        self.episode_len = 200
        self.state_dim = 14
        self.lr = 1e-5
        self.policy_class = 'ACT'
        self.task_name = 'bi_peg_transfer'
        self.onscreen_render = True
        self.seed = 1024
        self.temporal_agg = True
        self.camera_names = ['rgb1']
        self.images_input = None#['mask']
        self.real_robot = False
        self.chunk_size = 5
        self.kl_weight = 10
        self.batch_size = 16
        self.hidden_dim = 512
        self.dim_feedforward = 3200