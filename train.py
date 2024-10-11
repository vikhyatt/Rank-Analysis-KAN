import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
import warmup_scheduler
#from lion_pytorch import Lion
import numpy as np
import os
from tqdm import tqdm
from torch.linalg import matrix_rank
from utils import rand_bbox
import matplotlib.pyplot as plt
from torch.linalg import svdvals
import copy


class Trainer(object):
    def __init__(self, model, args):
        self.model_configs = args.model_configs
        self.checkpoint_epoch = args.checkpoint_epoch
        wandb.config.update(args)
        self.device = args.device
        self.clip_grad = args.clip_grad
        self.cutmix_beta = args.cutmix_beta
        self.cutmix_prob = args.cutmix_prob
        self.num_workers = args.num_workers
        self.model = model
        self.optim_name = 'none'
        self.min_lr = args.min_lr
        self.amp_train = True
        
        if args.optimizer=='sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
        elif args.optimizer=='adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)

        elif args.optimizer == 'lion':
            self.optimizer = Lion(self.model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)

        elif args.optimizer == 'LBFGS':
            self.optim_name = 'lbfgs'
            self.optimizer = optim.LBFGS(self.model.parameters(), lr=args.lr)
            
        else:
            raise ValueError(f"No such optimizer: {self.optimizer}")
        
        self.is_scheduler = args.scheduler
        
        if args.scheduler=='step':
            self.base_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[args.epochs//2, 3*args.epochs//4], gamma=args.gamma)
        elif args.scheduler=='cosine':
            self.base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.epochs, eta_min=args.min_lr)
            
        elif args.scheduler=='none':
            self.base_scheduler = None
            
        else:
            raise ValueError(f"No such scheduler: {self.scheduler}")


        if args.warmup_epoch:
            if args.scheduler != 'none':
                self.scheduler = warmup_scheduler.GradualWarmupScheduler(self.optimizer, multiplier=1., total_epoch=args.warmup_epoch, after_scheduler=self.base_scheduler)
        else:
            if args.scheduler != 'none':
                self.scheduler = self.base_scheduler

        if self.amp_train:
            self.scaler = torch.amp.GradScaler('cuda')

        self.epochs = args.epochs
        self.criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

        self.num_steps = 0
        self.epoch_loss, self.epoch_corr, self.epoch_acc = 0., 0., 0.
    
    def _train_one_step(self, batch):
        self.model.train()
        img, label = batch
        self.num_steps += 1
        img, label = img.to(self.device), label.to(self.device)
        
        self.optimizer.zero_grad()
        r = np.random.rand(1)
        if self.cutmix_beta > 0 and r < self.cutmix_prob:
            # generate mixed sample
            lam = np.random.beta(self.cutmix_beta, self.cutmix_beta)
            rand_index = torch.randperm(img.size(0)).to(self.device)
            target_a = label
            target_b = label[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(img.size(), lam)
            img[:, :, bbx1:bbx2, bby1:bby2] = img[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (img.size()[-1] * img.size()[-2]))
            # compute output
            if self.amp_train:
                with torch.amp.autocast('cuda',dtype = torch.bfloat16):
                    out = self.model(img)
                    loss = self.criterion(out, target_a) * lam + self.criterion(out, target_b) * (1. - lam)
            else:
                out = self.model(img)
                loss = self.criterion(out, target_a) * lam + self.criterion(out, target_b) * (1. - lam)
        else:
            # compute output
            if self.amp_train:
                with torch.amp.autocast('cuda',dtype = torch.bfloat16):
                    out = self.model(img)
                    loss = self.criterion(out, label)
            else:
                out = self.model(img)
                loss = self.criterion(out, label)

        if self.amp_train:
            self.scaler.scale(loss).backward()
            if self.clip_grad:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
                
            self.scaler.step(self.optimizer)
            self.scaler.update()

        else:
            loss.backward()
            if self.clip_grad:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
            self.optimizer.step()

        acc = out.argmax(dim=-1).eq(label).sum(-1)/img.size(0)
        wandb.log({
            'loss':loss,
            'acc':acc
        }, step=self.num_steps)


    # @torch.no_grad
    def _test_one_step(self, batch):
        self.model.eval()
        img, label = batch
        img, label = img.to(self.device), label.to(self.device)

        with torch.no_grad():
            out = self.model(img)
            loss = self.criterion(out, label)

        self.epoch_loss += loss * img.size(0)
        self.epoch_corr += out.argmax(dim=-1).eq(label).sum(-1)


    def compute_svd(self, epoch, test_dl, init = 'default'):
        activations = {}
        def hook_fn(module, input, output):
            activations[module] = output

        #clone_model = copy.deepcopy(self.model)
        #clone_model = clone_model.to(torch.device('cpu'))
        if torch.cuda.device_count() > 1:
            clone_model = self.model
            num_layers = len(clone_model.module.mixer_layers)
            handles = []
            for layer_id in range(num_layers):
                handle1 = clone_model.module.mixer_layers[layer_id].kan1.fc1.register_forward_hook(hook_fn)
                handle2 = clone_model.module.mixer_layers[layer_id].kan1.fc2.register_forward_hook(hook_fn)
                handle3 = clone_model.module.mixer_layers[layer_id].kan2.fc1.register_forward_hook(hook_fn)
                handle4 = clone_model.module.mixer_layers[layer_id].kan2.fc2.register_forward_hook(hook_fn)
                handles.append([handle1,handle2, handle3, handle4 ])
    
            act_per_batch = []
            #torch.set_num_threads(self.num_workers)
            for i, batch in tqdm(enumerate(test_dl)):
                if i == 10:
                    break
                img, label = batch
                img, label = img.to(self.device), label.to(self.device)
                
                out = clone_model(img)
                if i <= 10:
                    acts = []
                    for layer_id in range(num_layers):
                        act1 = activations[clone_model.module.mixer_layers[layer_id].kan1.fc1].cpu().detach().numpy()
                        act2 = activations[clone_model.module.mixer_layers[layer_id].kan1.fc2].cpu().detach().numpy()
                        act3 = activations[clone_model.module.mixer_layers[layer_id].kan2.fc1].cpu().detach().numpy()
                        act4 = activations[clone_model.module.mixer_layers[layer_id].kan2.fc2].cpu().detach().numpy()
                        acts.append([act1, act2, act3, act4])
                
                    act_per_batch.append(acts)
            
                else:
                    for layer_id in range(num_layers):
                        act1 = activations[clone_model.module.mixer_layers[layer_id].kan1.fc1].cpu().detach().numpy()
                        act2 = activations[clone_model.module.mixer_layers[layer_id].kan1.fc2].cpu().detach().numpy()
                        act3 = activations[clone_model.module.mixer_layers[layer_id].kan2.fc1].cpu().detach().numpy()
                        act4 = activations[clone_model.module.mixer_layers[layer_id].kan2.fc2].cpu().detach().numpy()
                        act_per_batch[0][layer_id][0] = np.concatenate((act_per_batch[0][layer_id][0], act1), axis=0)
                        act_per_batch[0][layer_id][1] = np.concatenate((act_per_batch[0][layer_id][1], act2), axis=0)
                        act_per_batch[0][layer_id][2] = np.concatenate((act_per_batch[0][layer_id][2], act3), axis=0)
                        act_per_batch[0][layer_id][3] = np.concatenate((act_per_batch[0][layer_id][3], act4), axis=0)
        else:
            clone_model = self.model
            num_layers = len(clone_model.mixer_layers)
            handles = []
            for layer_id in range(num_layers):
                handle1 = clone_model.mixer_layers[layer_id].kan1.fc1.register_forward_hook(hook_fn)
                handle2 = clone_model.mixer_layers[layer_id].kan1.fc2.register_forward_hook(hook_fn)
                handle3 = clone_model.mixer_layers[layer_id].kan2.fc1.register_forward_hook(hook_fn)
                handle4 = clone_model.mixer_layers[layer_id].kan2.fc2.register_forward_hook(hook_fn)
                handles.append([handle1,handle2, handle3, handle4 ])
    
            act_per_batch = []
            #torch.set_num_threads(self.num_workers)
            for i, batch in tqdm(enumerate(test_dl)):
                if i == 10:
                    break
                img, label = batch
                img, label = img.to(self.device), label.to(self.device)
                
                out = clone_model(img)
                if i <= 10:
                    acts = []
                    for layer_id in range(num_layers):
                        act1 = activations[clone_model.mixer_layers[layer_id].kan1.fc1].cpu().detach().numpy()
                        act2 = activations[clone_model.mixer_layers[layer_id].kan1.fc2].cpu().detach().numpy()
                        act3 = activations[clone_model.mixer_layers[layer_id].kan2.fc1].cpu().detach().numpy()
                        act4 = activations[clone_model.mixer_layers[layer_id].kan2.fc2].cpu().detach().numpy()
                        acts.append([act1, act2, act3, act4])
                
                    act_per_batch.append(acts)
            
                else:
                    for layer_id in range(num_layers):
                        act1 = activations[clone_model.mixer_layers[layer_id].kan1.fc1].cpu().detach().numpy()
                        act2 = activations[clone_model.mixer_layers[layer_id].kan1.fc2].cpu().detach().numpy()
                        act3 = activations[clone_model.mixer_layers[layer_id].kan2.fc1].cpu().detach().numpy()
                        act4 = activations[clone_model.mixer_layers[layer_id].kan2.fc2].cpu().detach().numpy()
                        act_per_batch[0][layer_id][0] = np.concatenate((act_per_batch[0][layer_id][0], act1), axis=0)
                        act_per_batch[0][layer_id][1] = np.concatenate((act_per_batch[0][layer_id][1], act2), axis=0)
                        act_per_batch[0][layer_id][2] = np.concatenate((act_per_batch[0][layer_id][2], act3), axis=0)
                        act_per_batch[0][layer_id][3] = np.concatenate((act_per_batch[0][layer_id][3], act4), axis=0)
                
        final_act = []
        for layer_id in tqdm(range(num_layers)):
            final_act.append([])
            final_act[-1].append(np.concatenate([act_per_batch[i][layer_id][0] for i in range(10)], axis=0))
            final_act[-1].append(np.concatenate([act_per_batch[i][layer_id][1] for i in range(10)], axis=0))
            final_act[-1].append(np.concatenate([act_per_batch[i][layer_id][2] for i in range(10)], axis=0))
            final_act[-1].append(np.concatenate([act_per_batch[i][layer_id][3] for i in range(10)], axis=0))


        for layer_id in [1, num_layers//2, num_layers]:
            svd_30 = svdvals(torch.tensor(final_act[layer_id-1][0]))
            svd_31 = svdvals(torch.tensor(final_act[layer_id-1][1]))
            svd_32 = svdvals(torch.tensor(final_act[layer_id-1][2]))
            svd_33 = svdvals(torch.tensor(final_act[layer_id-1][3]))
            
            plt.errorbar([i for i in range(svd_30.shape[-1])], svd_30.mean(axis = 0), yerr=svd_30.std(axis = 0) , label=f'KAN Linear 1 for KAN1 for Layer {layer_id}', marker='o', capsize=1, ms = 1)
            plt.errorbar([i for i in range(svd_31.shape[-1])], svd_31.mean(axis = 0), yerr=svd_31.std(axis = 0) , label=f'KAN Linear 2 for KAN1 for Layer {layer_id}', marker='o', capsize=1, ms = 1)
            plt.errorbar([i for i in range(svd_32.shape[-1])], svd_32.mean(axis = 0), yerr=svd_32.std(axis = 0) , label=f'KAN Linear 1 for KAN2 for Layer {layer_id}', marker='o', capsize=1, ms = 1)
            plt.errorbar([i for i in range(svd_33.shape[-1])], svd_33.mean(axis = 0), yerr=svd_33.std(axis = 0) , label=f'KAN Linear 2 for KAN2 for Layer {layer_id}', marker='o', capsize=1, ms = 1)
            plt.title(f'Init:{init},{self.model_configs} model at Epoch {epoch}')
            plt.xlabel("i'th singular value")
            plt.ylabel('Mean SV over input batch')
            plt.yscale('log')
            plt.legend()
            plt.savefig(f'svd_plots/Epoch {epoch}, SV for Layer {layer_id}, Init: {init},{self.model_configs}.jpg', dpi = 200, bbox_inches = 'tight')
            plt.close()

    
    def fit(self, train_dl, test_dl, init = 'default'):
        if self.checkpoint_epoch:
            PATH = f"../saved_models/{self.checkpoint_epoch}, {self.model_configs}.pt"
            checkpoint = torch.load(PATH, map_location = self.device, weights_only=True)
            state_dict = checkpoint['model_state_dict']
            # If the model was saved with DataParallel, keys will have `module.` prefix
            if torch.cuda.device_count() <= 1:
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                # Strip the `module.` prefix from the keys
                    name = k.replace('module.', '') 
                    new_state_dict[name] = v
            #self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.load_state_dict(new_state_dict)
            else:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                #self.model = nn.DataParallel(self.model)
                #self.model = self.model.to(self.device)

            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            run_epoch = checkpoint['epoch']
            run_loss = checkpoint['loss']
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs - self.checkpoint_epoch , eta_min=self.min_lr)
            
        for epoch in range(self.checkpoint_epoch + 1, self.epochs + 1):
            if epoch > 10:
                break
            if epoch == 1 or epoch == 5 or epoch == 10:
                PATH = f"../saved_models/{epoch}, {self.model_configs}.pt"
                prev_PATH = f"../saved_models/{epoch-10}, {self.model_configs}.pt"
                torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': self.epoch_loss,
                }, PATH)
                if os.path.exists(prev_PATH):
                    os.remove(prev_PATH)

                self.compute_svd(epoch, test_dl, init = init)
                    
            for batch in train_dl:
                self._train_one_step(batch)
            wandb.log({
                'epoch': epoch, 
                # 'lr': self.scheduler.get_last_lr(),
                'lr':self.optimizer.param_groups[0]["lr"]
                }, step=self.num_steps
            )
            
            if self.is_scheduler != 'none':
                self.scheduler.step()

            
            num_imgs = 0.
            self.epoch_loss, self.epoch_corr, self.epoch_acc = 0., 0., 0.
            for batch in test_dl:
                self._test_one_step(batch)
                num_imgs += batch[0].size(0)
            self.epoch_loss /= num_imgs
            self.epoch_acc = self.epoch_corr / num_imgs
            wandb.log({
                'val_loss': self.epoch_loss,
                'val_acc': self.epoch_acc
                }, step=self.num_steps
            )

            if epoch % 10 == 0:
                PATH = f"../saved_models/{epoch}, {self.model_configs}.pt"
                prev_PATH = f"../saved_models/{epoch-10}, {self.model_configs}.pt"
                torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': self.epoch_loss,
                }, PATH)
                if os.path.exists(prev_PATH):
                    os.remove(prev_PATH)
                
