import os
import pickle
import statistics
import argparse
import json 
import numpy as np
from torch import nn
import torch

parser = argparse.ArgumentParser(description='2-stage model encoding generation')
parser.add_argument('--encoding_dir', default="configs/baseline.yaml", type=str)
parser.add_argument('--dataset', default = 'train', type=str,help='Evaluate of which json')
parser.add_argument('--tracks', type=str,help='Train/val tracks json')
# parser.add_argument('--save_dir', type=str)

args = parser.parse_args()
threshold = 0.6


with open(os.path.join(args.encoding_dir, args.dataset+"_stage1_text_subject_encodings.pkl"), 'rb') as f:
    subj_embed = pickle.load(f)

with open(os.path.join(args.encoding_dir, args.dataset+"_stage1_crops_car_encodings.pkl"), 'rb') as f:
    crop_embed = pickle.load(f)

with open(args.tracks) as f:
    tracks = json.load(f)

# os.makedirs(args.save_dir,exist_ok = True)

#Stage 1

crop_ids = list(crop_embed.keys())
ds_s2 = {}
mrr_sum = 0.0

for qid, query in subj_embed.items():
    qscores = []
    for cid in crop_ids:
        cars = crop_embed[cid]
        score = nn.CosineSimilarity(dim = 0)(torch.tensor(query),torch.tensor(cars))
        qscores.append(score)
      
    valid_tracks = np.array(qscores) > threshold
    ds_s2[qid] = np.array(crop_ids)[valid_tracks]

    #MRR calc
    # target_rank = ret_tracks.index(qid) + 1
    # mrr_sum += (1/target_rank)        

# with open(os.path.join(args.save_dir, 'stage_1_retrievals.pkl'),'wb') as f:
#     pickle.dump(ds_s2, f)


def getTopKRecall(retreived, actual):
    i = 0
    actual = set(actual)
    retr = set(retreived)

    retr_len = len(actual.intersection(retr))

    return retr_len / len(actual)


#Stage 2

with open(os.path.join(args.encoding_dir, args.dataset+"_stage2_text_masked_query_encodings.pkl"), 'rb') as f:
    qm_embed = pickle.load(f)

with open(os.path.join(args.encoding_dir, args.dataset+"_stage2_motion_encodings.pkl"), 'rb') as f:
    motion_embed = pickle.load(f)

# for k in ds_s2.keys():
#     ds_s2[k] = tracks[k]['targets']

top5_count =0 
top10_count = 0
top15_count = 0
mmr_sum = 0.0
for qid, query in qm_embed.items():
    valid_tracks = []
    for vt_id in ds_s2[qid]:
        motions = motion_embed[vt_id]
        score = nn.CosineSimilarity(dim = 0)(torch.tensor(query),torch.tensor(motions))
        valid_tracks.append(score)

    rank = np.argsort(valid_tracks)
    ranked_tracks = np.array(ds_s2[qid])[rank]


    try:
        mmr_sum += 1/(list(ranked_tracks).index(qid) + 1)
    except ValueError as ve:
        mmr_sum += 0.0

    
    top5_count += getTopKRecall(ranked_tracks[:5],[qid])
    
    top10_count += getTopKRecall(ranked_tracks[:10],[qid])
    
    # top15_count += getTopKRecall(ranked_tracks[:15],[qid])

mmr = mmr_sum / len(qm_embed.keys())
print(mmr)

print("Top 5 Recall: ", top5_count/len(qm_embed.items()))
print("Top 10 Recall: ", top10_count/len(qm_embed.items()))
# print("Top 15 Recall: ", top15_count/len(subj_embed.keys()))



    