import os

from utils.utils import make_dir

class HP():
    def __init__(self, model, name='temp', device='cuda', seed = None):
        self.name = name
        self.device = device
        self.multi_gpu = False
        self.model = model
        self.same_per_sample = True
        self.epoch_load = True

        self.position_embedding = False
        self.seed = seed if seed is not None else 42
        self.epochs = 200
        self.batch_size = 4
        self.optimizer_lr = 1e-4
        self.scheduler_step = 10
        self.scheduler_gamma = 0.8
        self.monitoring_cycle = 1
        self.save_cycle = 50
        self.bce_loss_weight = 0.5
        self.dice_loss_weight = 0
        self.focal_loss_weight = 0
        self.gul_loss_weight = 0.5
        self.boundary_loss_weight = 0

        self.batch_size_inference = 12
        self.epoch_load = None
        
        self.path_main = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.path_data          = f'{self.path_main}/data'
        self.path_Res           = f'{self.path_main}/res'
        self.path_res           = f'{self.path_main}/res/{self.name}'
        self.path_model         = f'{self.path_main}/res/{self.name}/model'
        self.path_inference     = f'{self.path_main}/res/{self.name}/inference'
        self.path_evaluation    = f'{self.path_main}/res/{self.name}/evaluation'
        self.path_fig           = f'{self.path_main}/res/{self.name}/fig'
        self.path_temp          = f'{self.path_main}/res/{self.name}/temp'
        
        make_dir(self.path_Res)
        # make_dir(f'{self.path_Res}/Fig')
        make_dir(self.path_res)
        make_dir(self.path_model)
        make_dir(self.path_inference)
        make_dir(self.path_evaluation)
        make_dir(self.path_fig)
        make_dir(self.path_temp)
        
hp = HP(name='temp', model = None)

        
