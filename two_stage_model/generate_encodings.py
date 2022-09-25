from EncodingGenerator import EncodingGenerator
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
# from absl import flags
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from config import get_default_config
from models.siamese_baseline import SiameseNewStage1, SiameseNewStage2, TripletNewStage2, TripletNewStage1
from utils import load_new_model_from_checkpoint,load_new_model_from_checkpoint_stage2
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

parser = argparse.ArgumentParser(description='2-stage model encoding generation')
parser.add_argument('--config', default="configs/baseline.yaml", type=str,
                    help='config_file')
parser.add_argument('--save_dir', type=str,
                    help='Encoding save dir')
parser.add_argument('--dataset', default = 'train', type=str,
                    help='Evaluate of which json')
args = parser.parse_args()

os.makedirs(args.save_dir,exist_ok = True)


cfg = get_default_config()
cfg.merge_from_file(args.config)

use_cuda = True

if(args.dataset == 'train'):
    egen = EncodingGenerator(cfg, cfg.DATA.TRAIN_JSON_PATH)
else:
    egen = EncodingGenerator(cfg, cfg.DATA.EVAL_JSON_PATH)

#Generate stage 1 encodings first
print('Starting stage 1 processing')
#load stage 1 model
model = TripletNewStage1(cfg.STAGE1MODEL)
# model = load_new_model_from_checkpoint(model, cfg.STAGE1MODEL.CHECKPOINT, cfg.STAGE1MODEL.NUM_CLASS, cfg.STAGE1MODEL.EMBED_DIM)
checkpoint = torch.load(cfg.STAGE1MODEL.CHECKPOINT)
new_state_dict = OrderedDict()
for k, v in checkpoint['state_dict'].items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)

if use_cuda:
    model.cuda()
    torch.backends.cudnn.benchmark = True

egen.generate_text_encoding(model, egen.query_subjects, egen.subject_ids, os.path.join(args.save_dir, args.dataset+'_stage1_text_subject_encodings.pkl'))

egen.generate_image_encodings(model, egen.raw_frames, egen.frame_ids, cfg.DATA.CITYFLOW_PATH, os.path.join(args.save_dir, args.dataset+'_stage1_crops_car_encodings.pkl'), boxes = egen.boxes)

# delete stage 1 model
del model

print('Starting stage 2 processing')
#load stage 2 model
model = TripletNewStage2(cfg.STAGE2MODEL)
checkpoint = torch.load(cfg.STAGE2MODEL.CHECKPOINT)
new_state_dict = OrderedDict()
for k, v in checkpoint['state_dict'].items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)

if use_cuda:
    model.cuda()
    torch.backends.cudnn.benchmark = True

egen.generate_text_encoding(model, egen.masked_queries, egen.mq_ids, os.path.join(args.save_dir, args.dataset+'_stage2_text_masked_query_encodings.pkl'))

egen.generate_image_encodings(model, egen.motion_images, egen.track_ids, cfg.DATA.MOTION_PATH, os.path.join(args.save_dir, args.dataset+'_stage2_motion_encodings.pkl'))

# delete stage 2 model
del model
