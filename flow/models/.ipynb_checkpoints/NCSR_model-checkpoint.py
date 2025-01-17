# Copyright (c) 2020 Huawei Technologies Co., Ltd.
# Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International) (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#
# The code is released for academic research use only. For commercial use, please contact Huawei Technologies Co., Ltd.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file contains content licensed by https://github.com/xinntao/BasicSR/blob/master/LICENSE/LICENSE

import logging
from collections import OrderedDict
from utils.util import get_resume_paths, opt_get

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
import sys
sys.path.append('../NTIRE21_Learning_SR_Space/')
from imresize import imresize
import random


logger = logging.getLogger('base')


class NCSRModel(BaseModel):
    def __init__(self, opt, step):
        super(NCSRModel, self).__init__(opt)
        self.opt = opt
        
        self.divide_freq = opt['divide_freq']
        self.heats = opt['val']['heats']
        self.n_sample = opt['val']['n_sample']
        self.hr_size = opt_get(opt, ['datasets', 'train', 'center_crop_hr_size'])
        self.hr_size = 160 if self.hr_size is None else self.hr_size
        self.lr_size = self.hr_size // opt['scale']
        self.max_std = opt['std']
        self.LRnoise = opt['LRnoise']
        self.std_weight = 1.
        self.mode = opt['mode']
        self.scale = opt['scale']
        self.p = opt['prob']
        self.std_channels = opt_get(opt, ['network_G', 'flow', 'std_channels'])
        
        self.std_in = torch.zeros(1, 1, 1, 1).to('cuda')
        self.eps = None
        
        opt['dist'] = False
        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
            
        train_opt = opt['train']

        # define network and load pretrained models
        self.netG = networks.define_Flow(opt, step).to(self.device)
        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = DataParallel(self.netG)
        # print network
        #self.print_network()

        if opt_get(opt, ['path', 'resume_state'], 1) is not None:
            self.load()
        else:
            print("WARNING: skipping initial loading, due to resume_state None")

        if self.is_train:
            self.netG.train()

            self.init_optimizer_and_scheduler(train_opt)
            self.log_dict = OrderedDict()

    def to(self, device):
        self.device = device
        self.netG.to(device)

    def init_optimizer_and_scheduler(self, train_opt):
        # optimizers
        self.optimizers = []
        wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
        optim_params_RRDB = []
        optim_params_other = []
        for k, v in self.netG.named_parameters():  # can optimize for a part of the model
            #print(k, v.requires_grad)
            if v.requires_grad:
                if '.RRDB.' in k:
                    optim_params_RRDB.append(v)
                    #print('opt', k)
                else:
                    optim_params_other.append(v)
                #if self.rank <= 0:
                #    logger.warning('Params [{:s}] will not optimize.'.format(k))

        print('rrdb params', len(optim_params_RRDB))

        self.optimizer_G = torch.optim.Adam(
            [
                {"params": optim_params_other, "lr": train_opt['lr_G'], 'beta1': train_opt['beta1'],
                 'beta2': train_opt['beta2'], 'weight_decay': wd_G},
                {"params": optim_params_RRDB, "lr": train_opt.get('lr_RRDB', train_opt['lr_G']),
                 'beta1': train_opt['beta1'],
                 'beta2': train_opt['beta2'], 'weight_decay': wd_G}
            ],
        )
        print(self.optimizer_G)
        self.optimizers.append(self.optimizer_G)
        # schedulers
        if train_opt['lr_scheme'] == 'MultiStepLR':
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                     restarts=train_opt['restarts'],
                                                     weights=train_opt['restart_weights'],
                                                     gamma=train_opt['lr_gamma'],
                                                     clear_state=train_opt['clear_state'],
                                                     lr_steps_invese=train_opt.get('lr_steps_inverse', [])))
        elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.CosineAnnealingLR_Restart(
                        optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                        restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
        else:
            raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

    def add_optimizer_and_scheduler_RRDB(self, train_opt):
        # optimizers
        assert len(self.optimizers) == 1, self.optimizers
        assert len(self.optimizer_G.param_groups[1]['params']) == 0, self.optimizer_G.param_groups[1]
        for k, v in self.netG.named_parameters():  # can optimize for a part of the model
            if v.requires_grad:
                if '.RRDB.' in k:\
                    self.optimizer_G.param_groups[1]['params'].append(v)
        assert len(self.optimizer_G.param_groups[1]['params']) > 0

    def feed_data(self, data, need_GT=True, val_mode=False):
        std_min = 0.0
        std_max = self.max_std
        self.var_L = data['LQ'].to(self.device)  # LQ  
        self.eps = None
        self.prob = random.random()
        if need_GT:
            self.real_H = data['GT'].to(self.device)  # GT
            std = (std_max - std_min) * torch.rand_like(self.real_H[:,0,0,0]).view(-1,1,1,1) + std_min
            self.eps = torch.randn_like(self.real_H) * std
            self.std_in = std  #/ std_max * self.std_weight
            #if self.p <= self.prob:
            #    self.eps = torch.randn_like(self.real_H) * 0
            #    self.std_in = torch.zeros(self.var_L.size(0), 1, 1, 1).to(self.var_L)            
            if self.LRnoise: #and self.p > self.prob:
                LR_eps = []
                for i in range(len(self.eps)):
                    LR_eps.append(torch.tensor(imresize(self.eps[i].cpu().permute(1,2,0), 1/self.scale), dtype=torch.float32).permute(2,0,1).to(self.device))
                LReps = torch.stack(LR_eps)
                self.var_L = self.var_L + LReps   
        else:
            self.std_in = torch.zeros(self.var_L.size(0), 1, 1, 1).to(self.var_L)
        if val_mode == True:
            self.std_in = torch.zeros(self.var_L.size(0), 1, 1, 1).to(self.var_L)
        
            

    def optimize_parameters(self, step):

        train_RRDB_delay = opt_get(self.opt, ['network_G', 'train_RRDB_delay'])
        if train_RRDB_delay is not None and step > int(train_RRDB_delay * self.opt['train']['niter']) \
                and not self.netG.module.RRDB_training:
            if self.netG.module.set_rrdb_training(True):
                self.add_optimizer_and_scheduler_RRDB(self.opt['train'])

        # self.print_rrdb_state()

        self.netG.train()
        self.log_dict = OrderedDict()
        self.optimizer_G.zero_grad()     
                        
        if self.std_channels == 3 and self.eps is not None:
            std = self.eps
        else:
            std = self.std_in
            
        if self.mode == 'softflow':
            gt = self.real_H + self.eps
        else:
            gt = self.real_H  
            std = None
        
        losses = {}
        weight_fl = opt_get(self.opt, ['train', 'weight_fl'])
        weight_fl = 1 if weight_fl is None else weight_fl      
        if weight_fl > 0:
            z, nll, y_logits = self.netG(gt=gt, lr=self.var_L, reverse=False, std=std)
            nll_loss = torch.mean(nll)
            losses['nll_loss'] = nll_loss * weight_fl
            
        weight_l1 = opt_get(self.opt, ['train', 'weight_l1']) or 0
        if weight_l1 > 0:
            z = self.get_z(heat=0, seed=None, batch_size=self.var_L.shape[0], lr_shape=self.var_L.shape)
            sr, logdet = self.netG(lr=self.var_L, z=z, eps_std=0, reverse=True, reverse_with_grad=True, std=std)
            l1_loss = (sr - self.real_H).abs().mean()
            losses['l1_loss'] = l1_loss * weight_l1
        
        total_loss = sum(losses.values())
        #print(losses)
        total_loss.backward()
        self.optimizer_G.step()

        mean = total_loss.item()
        return mean

    def print_rrdb_state(self):
        for name, param in self.netG.module.named_parameters():
            if "RRDB.conv_first.weight" in name:
                print(name, param.requires_grad, param.data.abs().sum())
        print('params', [len(p['params']) for p in self.optimizer_G.param_groups])

    def test(self):
        self.netG.eval()
        self.fake_H = {}
        for heat in self.heats:
            for i in range(self.n_sample):
                z = self.get_z(heat, seed=None, batch_size=self.var_L.shape[0], lr_shape=self.var_L.shape)
                with torch.no_grad():
                    self.fake_H[(heat, i)], logdet = self.netG(lr=self.var_L, z=z, eps_std=heat, reverse=True, std=self.std_in)
        with torch.no_grad():
            _, nll, _ = self.netG(gt=self.real_H, lr=self.var_L, reverse=False, std=self.std_in)
        self.netG.train()
        return nll.mean().item()

    def get_encode_nll(self, lq, gt):
        self.netG.eval()
        with torch.no_grad():
            _, nll, _ = self.netG(gt=gt, lr=lq, reverse=False, std=self.std_in)
        self.netG.train()
        return nll.mean().item()

    def get_sr(self, lq, heat=None, seed=None, z=None, epses=None):
        return self.get_sr_with_z(lq, heat, seed, z, epses)[0]

    def get_encode_z(self, lq, gt, epses=None, add_gt_noise=True):
        self.netG.eval()
        with torch.no_grad():
            z, _, _ = self.netG(gt=gt, lr=lq, reverse=False, epses=epses, add_gt_noise=add_gt_noise, std=self.std_in)
        self.netG.train()
        return z

    def get_encode_z_and_nll(self, lq, gt, epses=None, add_gt_noise=True):
        self.netG.eval()
        with torch.no_grad():
            z, nll, _ = self.netG(gt=gt, lr=lq, reverse=False, epses=epses, add_gt_noise=add_gt_noise, std=self.std_in)
        self.netG.train()
        return z, nll

    def get_sr_with_z(self, lq, heat=None, seed=None, z=None, epses=None):
        self.netG.eval()

        z = self.get_z(heat, seed, batch_size=lq.shape[0], lr_shape=lq.shape).to(lq.device) if z is None and epses is None else z
        #std = (self.max_std) * torch.randn([1,1,1,1]).view(-1,1,1,1).to('cuda')
        #self.std_in = std / self.max_std * self.std_weight
        #print(z.device)
        #print(lq.device)
        #print(self.std_in.device)
        with torch.no_grad():
            sr, logdet = self.netG(lr=lq, z=z, eps_std=heat, reverse=True, epses=epses, std=self.std_in)
        self.netG.train()
        return sr, z

    def get_z(self, heat, seed=None, batch_size=1, lr_shape=None):
        if seed: torch.manual_seed(seed)
        if opt_get(self.opt, ['network_G', 'flow', 'split', 'enable']):
            C = self.netG.module.flowUpsamplerNet.C
            H = int(self.opt['scale'] * lr_shape[2] // self.netG.module.flowUpsamplerNet.scaleH)
            W = int(self.opt['scale'] * lr_shape[3] // self.netG.module.flowUpsamplerNet.scaleW)
            z = torch.normal(mean=0, std=heat, size=(batch_size, C, H, W)) if heat > 0 else torch.zeros(
                (batch_size, C, H, W))
        else:
            L = opt_get(self.opt, ['network_G', 'flow', 'L']) or 3
            fac = 2 ** (L - 3)
            z_size = int(self.lr_size // (2 ** (L - 3)))
            
            z = torch.normal(mean=0, std=heat, size=(batch_size, 3 * 8 * 8 * fac * fac, z_size, z_size))
        return z

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        def upsample(lr, scale=4):
            return torch.nn.functional.upsample(lr, scale_factor=scale, mode='bicubic')
        
        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L.detach()[0].float().cpu()

        if self.divide_freq:
            var_L_upsample = upsample(self.var_L, self.opt['scale'])
        else:
            var_L_upsample = torch.zeros_like(self.real_H)

        for heat in self.heats:
            for i in range(self.n_sample):
                out_dict[('SR', heat, i)] = (self.fake_H[(heat, i)] + var_L_upsample).detach()[0].float().cpu()

        if need_GT:
            out_dict['GT'] = (self.real_H + var_L_upsample).detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        _, get_resume_model_path = get_resume_paths(self.opt)
        if get_resume_model_path is not None:
            self.load_network(get_resume_model_path, self.netG, strict=True, submodule=None)
            return

        load_path_G = self.opt['path']['pretrain_model_G']
        load_submodule = self.opt['path']['load_submodule'] if 'load_submodule' in self.opt['path'].keys() else 'RRDB'
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path'].get('strict_load', True),
                              submodule=load_submodule)

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)