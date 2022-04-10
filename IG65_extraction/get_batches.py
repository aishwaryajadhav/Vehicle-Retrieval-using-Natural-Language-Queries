import json

path = "/home/ubuntu/efs/data/aicity/train_tracks.json"

file = open(path)
tracks = json.load(file)
file.close()

track_names = list(tracks.keys())

num_tracks = len(track_names)

batch_size = 300

output_path = "/home/ubuntu/efs/data/aicity/batch_tracks/{0}.txt"

for i in range(0, num_tracks, batch_size):
    track_list = track_names[i:i+batch_size]
    file = open(output_path.format(i), "w")
    file.write("\n".join(track_list))
    file.close()
