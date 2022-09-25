import json
import math
import os
import sys
from datetime import datetime
import argparse
from xmlrpc.client import boolean
import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing
import torch.multiprocessing as mp
# from absl import flags
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from sklearn.metrics import average_precision_score
from config import get_default_config
from models.siamese_baseline import SiameseBaselineModelv1,SiameseLocalandMotionModelBIG,SiameseNewStage1
from utils import TqdmToLogger, get_logger,AverageMeter,accuracy,ProgressMeter,load_new_model_from_checkpoint
from datasets import CityFlowNLDataset
from datasets import CityFlowNLInferenceDataset
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import time
import torch.nn.functional as F
from transformers import BertTokenizer,RobertaTokenizer, RobertaModel
from collections import OrderedDict
import pdb


torch.multiprocessing.set_sharing_strategy('file_system')




class WarmUpLR(_LRScheduler):
    def __init__(self, lr_scheduler, warmup_steps, eta_min=1e-7):
        self.lr_scheduler = lr_scheduler
        self.warmup_steps = warmup_steps
        self.eta_min = eta_min
        super().__init__(lr_scheduler.optimizer, lr_scheduler.last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [self.eta_min + (base_lr - self.eta_min) * (self.last_epoch / self.warmup_steps)
                    for base_lr in self.base_lrs]
        return self.lr_scheduler.get_lr()

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        if epoch < self.warmup_steps:
            super().step(epoch)
        else:
            self.last_epoch = epoch
            self.lr_scheduler.step(epoch - self.warmup_steps)

    def state_dict(self):
        state_dict = {key: value for key, value in self.__dict__.items() if key not in ('optimizer', 'lr_scheduler')}
        state_dict['lr_scheduler'] = self.lr_scheduler.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        lr_scheduler = state_dict.pop('lr_scheduler')
        self.__dict__.update(state_dict)
        self.lr_scheduler.load_state_dict(lr_scheduler)

best_sim_loss = 2.5729e-01
def evaluate(model,valloader,epoch,cfg,index=0):
    global best_sim_loss
    # print("Test::::")
    model.eval()
    sim_loss = 0.0

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    # ap_lang = AverageMeter('AP Lang', ':6.2f')
    ap_vis = AverageMeter('AP Sim', ':6.2f')
    progress = ProgressMeter(
        len(valloader),
        [batch_time, data_time, losses, ap_vis],
        prefix="Test Epoch: [{}]".format(epoch))
    end = time.time()
    
    with torch.no_grad():
        for batch_idx,batch in enumerate(valloader):
            image,text,id_car, target_ind, ind = batch
            tokens = tokenizer.batch_encode_plus(text, padding='longest', return_tensors='pt')
            data_time.update(time.time() - end)
            
            pairs,logit_scale,cls_logits = model(tokens['input_ids'].cuda(),tokens['attention_mask'].cuda(),image.cuda())
            
            logit_scale = logit_scale.mean().exp()
            loss =0 

            # for visual_embeds,lang_embeds in pairs:
            visual_embeds,lang_embeds = pairs[index]
            sim_i_2_t = torch.matmul(torch.mul(logit_scale, visual_embeds), torch.t(lang_embeds))
            sim_t_2_i = sim_i_2_t.t()
             
            batch_sim = torch.zeros(sim_i_2_t.shape)
            batch_size = len(ind)
            # pdb.set_trace()
            for t_i in range(batch_size):
                for t_j in range(batch_size):
                    if(ind[t_j] in target_ind[t_i]):
                        batch_sim[t_i][t_j] = 1

            
            loss_t_2_i = nn.BCEWithLogitsLoss()(sim_t_2_i, batch_sim.cuda())
            loss_i_2_t = nn.BCEWithLogitsLoss()(sim_i_2_t, batch_sim.T.cuda())
            loss += (2*loss_t_2_i+loss_i_2_t)/3

            sim_loss += loss
            
            ap_vis_t = average_precision_score(batch_sim, F.sigmoid(sim_t_2_i).detach().cpu().numpy())
            
            # pdb.set_trace()
            # acc1, acc5 = accuracy(sim_t_2_i, torch.arange(image.size(0)).cuda(), topk=(1, 5))
            losses.update(loss.item(), image.size(0))
            # ap_lang.update(ap_lang_t, image.size(0))
            ap_vis.update(ap_vis_t, image.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            progress.display(batch_idx)

    sim_loss = sim_loss / len(valloader)
    if sim_loss < best_sim_loss:
        best_sim_loss = sim_loss
        print("New best!:", best_sim_loss)
        checkpoint_file = args.name+"/checkpoint_best_eval_loss_{}.pth".format(best_sim_loss)
        torch.save(
            {"epoch": epoch, 
             "state_dict": model.state_dict(),
             "optimizer": optimizer.state_dict()}, checkpoint_file)


parser = argparse.ArgumentParser(description='AICT5 Training')
parser.add_argument('--load_existing', default=False, type=boolean)
parser.add_argument('--config', default="configs/baseline.yaml", type=str,
                    help='config_file')
parser.add_argument('--name', default="baseline", type=str,
                    help='experiments')
args = parser.parse_args()

cfg = get_default_config()
cfg.merge_from_file(args.config)


print("Cuda: ",torch.cuda.is_available())

transform_train = torchvision.transforms.Compose([
    # torchvision.transforms.RandomResizedCrop(cfg.DATA.SIZE, scale=(0.8, 1.)),
    torchvision.transforms.Resize((cfg.DATA.SIZE,cfg.DATA.SIZE)),
    torchvision.transforms.RandomApply([torchvision.transforms.RandomRotation(10)],p=0.5),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
transform_test = torchvision.transforms.Compose([
    torchvision.transforms.Resize((cfg.DATA.SIZE,cfg.DATA.SIZE)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

use_cuda = True
train_data=CityFlowNLDataset(cfg.DATA, json_path = cfg.DATA.TRAIN_JSON_PATH, transform=transform_test)
trainloader = DataLoader(dataset=train_data, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=cfg.TRAIN.NUM_WORKERS)
val_data=CityFlowNLDataset(cfg.DATA,json_path = cfg.DATA.EVAL_JSON_PATH, transform=transform_test,Random = False)
valloader = DataLoader(dataset=val_data, batch_size=cfg.TRAIN.BATCH_SIZE*2, shuffle=False, num_workers=cfg.TRAIN.NUM_WORKERS)
os.makedirs(args.name,exist_ok = True)

if cfg.MODEL.NAME == "base":
    model = SiameseBaselineModelv1(cfg.MODEL)
elif cfg.MODEL.NAME == "dual-stream":
    model = SiameseLocalandMotionModelBIG(cfg.MODEL)
elif cfg.MODEL.NAME == "new":
    model = SiameseNewStage1(cfg.MODEL)

else:
    assert cfg.MODEL.NAME in ["base","dual-stream","new"] , "unsupported model"
    
    
#***************PLEASE CHANGE THE LOAD WHEN LOADING A MODEL TRAINED BY MEEEE!!!!*****************

# model = load_new_model_from_checkpoint(model, cfg.MODEL.CHECKPOINT, cfg.MODEL.NUM_CLASS, cfg.MODEL.EMBED_DIM)
# else:
checkpoint = torch.load(cfg.MODEL.CHECKPOINT)
new_state_dict = OrderedDict()
for k, v in checkpoint['state_dict'].items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)
    
if use_cuda:
    model.cuda()
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    # cudnn.benchmark = True


optimizer = torch.optim.AdamW(model.parameters(), lr = cfg.TRAIN.LR.BASE_LR, weight_decay=1e-4)
optimizer.load_state_dict(checkpoint['optimizer'])

for param_group in optimizer.param_groups:
    print(param_group['lr'])
    break
# step_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=len(trainloader)*cfg.TRAIN.ONE_EPOCH_REPEAT*cfg.TRAIN.LR.DELAY , gamma=0.1)
# scheduler = WarmUpLR(lr_scheduler = step_scheduler , warmup_steps=int(1.*cfg.TRAIN.LR.WARMUP_EPOCH*len(trainloader)))

# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4,6,8,10,12,14,16,20,22,24,26,28,30,32,34,36,38], gamma=0.8)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[i for i in range(2,80,2)], gamma=0.8)


if cfg.MODEL.BERT_TYPE == "BERT":
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
elif cfg.MODEL.BERT_TYPE == "ROBERTA":
    tokenizer = RobertaTokenizer.from_pretrained(cfg.MODEL.BERT_NAME)


model.train()
global_step = 0

for epoch in range(cfg.TRAIN.EPOCH):
    evaluate(model,valloader,epoch,cfg,0)
    model.train()
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    # ap_lang = AverageMeter('AP Lang', ':6.2f')
    # ap_vis = AverageMeter('AP Vis', ':6.2f')
    ap_sim = AverageMeter('AP Sim', ':6.2f')
    progress = ProgressMeter(
        len(trainloader),
        [batch_time, data_time, losses, ap_sim],
        prefix="Epoch: [{}]".format(epoch))
    end = time.time()
    
    for batch_idx,batch in enumerate(trainloader):
        image, text, id_car, target_ind, ind = batch
        tokens = tokenizer.batch_encode_plus(text, padding='longest',return_tensors='pt')
        data_time.update(time.time() - end)
        global_step+=1
        optimizer.zero_grad()
        
        pairs,logit_scale,cls_logits = model(tokens['input_ids'].cuda(),tokens['attention_mask'].cuda(),image.cuda())
        
        logit_scale = logit_scale.mean().exp()
        loss = 0 

        batch_size = len(ind)
        # pdb.set_trace()
        batch_sim = torch.zeros((batch_size, batch_size))
        for t_i in range(batch_size):
            for t_j in range(batch_size):
                if(ind[t_j] in target_ind[t_i]):
                    batch_sim[t_i][t_j] = 1

        visual_embeds,lang_embeds = pairs[0]
        sim_i_2_t = torch.matmul(torch.mul(logit_scale, visual_embeds), torch.t(lang_embeds))
        sim_t_2_i = sim_i_2_t.t()
        loss_t_2_i = nn.BCEWithLogitsLoss()(sim_t_2_i, batch_sim.cuda())
        loss_i_2_t = nn.BCEWithLogitsLoss()(sim_i_2_t, batch_sim.T.cuda())
        loss += (2*loss_t_2_i+loss_i_2_t)/3

        for cls_logit in cls_logits:
            #using bcewithlogitsloss (Sigmoid + BCE loss) instead of (Softmax + ) CrossEntropy for multiclass classification
            # print("*****Logits shape: ",cls_logit.shape)
            # print("Target shape: ",id_car.shape)
            loss+= (nn.BCEWithLogitsLoss()(cls_logit, id_car.cuda())/len(cls_logits))

        # ap_vis_t = average_precision_score(id_car, F.sigmoid(cls_logits[0]).detach().cpu().numpy())
        # ap_lang_t = average_precision_score(id_car, F.sigmoid(cls_logits[1]).detach().cpu().numpy())

        ap_sim_t = average_precision_score(batch_sim, F.sigmoid(sim_t_2_i).detach().cpu().numpy())


        losses.update(loss.item(), image.size(0))
        # ap_lang.update(ap_lang_t, image.size(0))
        # ap_vis.update(ap_vis_t, image.size(0))
        ap_sim.update(ap_sim_t, image.size(0))

        loss.backward()
        optimizer.step()
        
        scheduler.step()
        batch_time.update(time.time() - end)
        end = time.time()

        # if batch_idx % cfg.TRAIN.PRINT_FREQ == 0:
        progress.display(batch_idx)

    if epoch%8==1:
        checkpoint_file = args.name+"/checkpoint_%d.pth"%epoch
        torch.save(
            {"epoch": epoch, "global_step": global_step,
             "state_dict": model.state_dict(),
             "optimizer": optimizer.state_dict()}, checkpoint_file)
    # if ap_lang.avg > best_top1:
    #     best_top1 = ap_lang.avg
    #     checkpoint_file = args.name+"/checkpoint_best.pth"
    #     torch.save(
    #         {"epoch": epoch, "global_step": global_step,
    #          "state_dict": model.state_dict(),
    #          "optimizer": optimizer.state_dict()}, checkpoint_file)
