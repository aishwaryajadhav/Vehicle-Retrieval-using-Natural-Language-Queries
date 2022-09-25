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

from models.TripletLoss import TripletLoss
from config import get_default_config
from models.siamese_baseline import SiameseBaselineModelv1,SiameseLocalandMotionModelBIG,SiameseNewStage1,SiameseNewStage2, TripletNewStage2, TripletNewStage1
from utils import TqdmToLogger, get_logger,AverageMeter,accuracy,ProgressMeter,load_new_model_from_checkpoint_stage2, load_new_model_from_checkpoint_ts1
from datasets import CityFlowNLDataset
from datasets import CityFlowNLDataset_Stage2, CityFlowNLDataset_Stage2_TripletLoss, CityFlowNLDataset_Stage1_TripletLoss
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import time
import torch.nn.functional as F
from transformers import BertTokenizer,RobertaTokenizer, RobertaModel
from collections import OrderedDict
import pdb

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')



best_loss = float('inf')
def evaluate(model,valloader,epoch,cfg,loss_fn,index=0):
    global best_loss
    print("Test::::")
    model.eval()
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    # top1_acc = AverageMeter('Acc@1', ':6.2f')
    # top5_acc = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(valloader),
        [batch_time, data_time, losses],
        prefix="Test Epoch: [{}]".format(epoch))
    end = time.time()
    with torch.no_grad():
        
        tot_los = 0
        for batch_idx,batch in enumerate(valloader):
            text, pos, neg = batch
            
            tokens = tokenizer.batch_encode_plus(text, padding='longest', return_tensors='pt')
            # data_time.update(time.time() - end)
            pairs,logit_scale = model(tokens['input_ids'].cuda(),tokens['attention_mask'].cuda(),pos.cuda(),neg.cuda())
            
            logit_scale = logit_scale.mean().exp()
            
            # for visual_embeds,lang_embeds in pairs:
            visual_embeds_pos,visual_embeds_neg,lang_embeds = pairs[index]
            
            loss = loss_fn(lang_embeds.cuda(), visual_embeds_pos.cuda(), visual_embeds_neg.cuda(), average = False)
            
            tot_los += loss.item()
     
        
        tot_los = tot_los / len(valloader.dataset)
        losses.update(tot_los, 1)
        batch_time.update(time.time() - end)
        progress.display(batch_idx)


    if tot_los < best_loss:
        best_loss = tot_los
        checkpoint_file = args.name+"/checkpoint_best_eval_loss_{}.pth".format(best_loss)
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
train_data=CityFlowNLDataset_Stage1_TripletLoss(cfg.DATA, json_path = cfg.DATA.TRAIN_JSON_PATH, transform=transform_train)
trainloader = DataLoader(dataset=train_data, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=cfg.TRAIN.NUM_WORKERS)

val_data=CityFlowNLDataset_Stage1_TripletLoss(cfg.DATA,json_path = cfg.DATA.EVAL_JSON_PATH, transform=transform_test,Random = False)
valloader = DataLoader(dataset=val_data, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False, num_workers=cfg.TRAIN.NUM_WORKERS)

os.makedirs(args.name,exist_ok = True)

if cfg.MODEL.NAME == "base":
    model = SiameseBaselineModelv1(cfg.MODEL)
elif cfg.MODEL.NAME == "dual-stream":
    model = SiameseLocalandMotionModelBIG(cfg.MODEL)
elif cfg.MODEL.NAME == "new":
    model = SiameseNewStage1(cfg.MODEL)
    
elif cfg.MODEL.NAME == "triplet_stage1":
    model = TripletNewStage1(cfg.MODEL)
elif cfg.MODEL.NAME == "triplet_stage2":
    model = TripletNewStage2(cfg.MODEL)
else:
    assert cfg.MODEL.NAME in ["base","dual-stream","new","new_stage2","triplet_stage1","triplet_stage2"] , "unsupported model"


if args.load_existing:
    if(cfg.MODEL.NAME == "new"):
        model = load_new_model_from_checkpoint(model, cfg.MODEL.CHECKPOINT, cfg.MODEL.NUM_CLASS, cfg.MODEL.EMBED_DIM)
    
    elif(cfg.MODEL.NAME == "triplet_stage1"): 
        load_new_model_from_checkpoint_ts1(model, cfg.MODEL.CHECKPOINT)
        
    elif(cfg.MODEL.NAME == "new_stage2" or cfg.MODEL.NAME == "triplet_stage2"):
        model = load_new_model_from_checkpoint_stage2(model, cfg.MODEL.CHECKPOINT, efficient_net = True)
        
    else:
        checkpoint = torch.load(cfg.EVAL.RESTORE_FROM)
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

scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4,6,8,10,12,14,16,20,22,24,26,28,30,32,34,36,38], gamma=0.08)

loss_fn = TripletLoss(margin = 2.0)

if cfg.MODEL.BERT_TYPE == "BERT":
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
elif cfg.MODEL.BERT_TYPE == "ROBERTA":
    tokenizer = RobertaTokenizer.from_pretrained(cfg.MODEL.BERT_NAME)


model.train()
global_step = 0

for epoch in range(cfg.TRAIN.EPOCH):
    evaluate(model,valloader,epoch,cfg,loss_fn,0)
    model.train()
    batch_time = AverageMeter('Time', ':6.3f')
    # data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    # top1_acc = AverageMeter('Acc@1', ':6.2f')
    # top5_acc = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        cfg.TRAIN.EPOCH,
        [batch_time, losses],
        prefix="Epoch: [{}]".format(epoch))
    

    tot_los = 0
    end = time.time()

    for batch_idx,batch in tqdm(enumerate(trainloader)):
        try:
            text, pos, neg = batch
        
            tokens = tokenizer.batch_encode_plus(text, padding='longest',return_tensors='pt')
            # data_time.update(time.time() - end)
            global_step+=1
            optimizer.zero_grad()
            
            pairs,logit_scale = model(tokens['input_ids'].cuda(),tokens['attention_mask'].cuda(),pos.cuda(),neg.cuda())
            
            logit_scale = logit_scale.mean().exp()
            
            visual_embeds_pos,visual_embeds_neg,lang_embeds = pairs[0]
            
            loss = loss_fn(lang_embeds.cuda(), visual_embeds_pos.cuda(), visual_embeds_neg.cuda())

            tot_los += (loss.item() * len(text))

            loss.backward()
            optimizer.step()        
            scheduler.step()

          
        except Exception as e:
            print("Some exception caught ", e)

   
    tot_los = tot_los / len(trainloader.dataset)
    losses.update(tot_los, 1)
    batch_time.update(time.time() - end)
    progress.display(epoch)
        
    if(epoch%3 == 0):
        checkpoint_file = args.name+"/checkpoint_%d.pth"%epoch
        torch.save(
            {"epoch": epoch, "global_step": global_step,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict()}, checkpoint_file)
    
    # if top5_acc.avg > best_top5:
    #     best_top5 = top5_acc.avg
    #     checkpoint_file = args.name+"/checkpoint_best.pth"
    #     torch.save(
    #         {"epoch": epoch, "global_step": global_step,
    #          "state_dict": model.state_dict(),
    #          "optimizer": optimizer.state_dict()}, checkpoint_file)
