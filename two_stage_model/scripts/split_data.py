import os
import os.path as osp
import json
import sys
import random
random.seed(888)

with open(sys.argv[1]) as f:
    tracks_train = json.load(f)

# for k,v in tracks_train.items():
#     p = "/".join(v["frames"][0].split("/")[:-1])
#     p = "../../../data/11775_Project/"+p[1:]
#     if(os.path.isdir(p)):
#         # print(p)
#         keys.append(k)
  
keys = list(tracks_train.keys())
print(len(keys))

random.shuffle(keys)

train_data = dict()
val_data = dict()
for key in keys[:100]:
	val_data[key] = tracks_train[key]
for key in keys[100:]:
	train_data[key] = tracks_train[key]

print("Train data len: ",len(train_data))
print("Val data len: ",len(val_data))

data_folder = "/".join(sys.argv[1].split('/')[:-1])

with open(osp.join(data_folder, "train.json"), "w") as f:
    json.dump(train_data, f,indent=4)
with open(osp.join(data_folder, "val.json"), "w") as f:
    json.dump(val_data, f,indent=4)
