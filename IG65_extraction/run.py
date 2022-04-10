import os
import glob


batch_path = "/home/ubuntu/efs/data/aicity/batch_tracks/*.txt"

files = glob.glob(batch_path)


run_command = "python image_to_vid.py /home/ubuntu/efs/data/aicity/train_tracks.json {0}"

for file in files:
    os.system(run_command.format(file))

