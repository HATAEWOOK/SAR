import torch

class Config(object):
    def __init__(self):
        self.manual_seed = 77
        self.trial_num = 1
        self.pre = 'SAR'
        self.dataset = 'FreiHAND'
        self.dataset_path = '/mnt/workplace/dataset/freiHand_dataset'
        self.split = 0.05
        self.device = "cuda"
        self.use_multigpu = True
        # network
        self.backbone = 'resnet34'
        self.num_stage = 2
        self.num_FMs = 8
        self.feature_size = 64
        self.heatmap_size = 32
        self.num_vert = 778
        self.num_joint = 21
        # training
        self.batch_size = 64
        self.lr = 1e3
        self.total_epoch = 50
        input_img_shape = (256, 256)
        self.depth_box = 0.3
        self.num_worker = 20
        # -------------
        self.save_epoch = 1
        self.eval_interval = 1
        self.print_iter = 10
        self.print_epoch = 100
        self.num_epoch_to_eval = 80
        for i in range(self.num_stage):
            self.loss_queries = ['coord_%d'%i, 'normal_%d'%i, 'edge_%d'%i]
            self.loss_weight = None
        # -------------
        self.pretrained_net = None
        self.checkpoint = None # put the path of the trained model's weights here
        self.continue_train = False
        self.vis = False
        # -------------
        self.experiment_name = self.pre + '_{}'.format(self.backbone) + '_Stage{}'.format(self.num_stage) + '_Batch{}'.format(self.batch_size) + \
                        '_lr{}'.format(self.lr) + '_Size{}'.format(input_img_shape[0]) + '_Epochs{}'.format(self.total_epoch)

cfg = Config()
