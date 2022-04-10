import json
import sys
import cv2
import numpy as np
import tqdm

image_base_path = "/home/ubuntu/efs/data/aicity"
video_out_base = "/home/ubuntu/efs/data/aicity/validation_tracks/{0}.mp4"

def image_to_video(image_paths, video_name):
    img=[]
    for i in range(len(image_paths)):
        img.append(cv2.imread(image_base_path+image_paths[i][1:]))

    height,width,layers=img[1].shape

    video_path = video_out_base.format(video_name)
    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
    video=cv2.VideoWriter(video_path,fourcc,1,(width,height))

    for j in range(len(img)):
        video.write(img[j])

    cv2.destroyAllWindows()
    video.release()


tracks_path = sys.argv[1]
names = sys.argv[2]

file = open(tracks_path)
tracks = json.load(file)
file.close()


track_names = []
file = open(names)
names = file.read()
file.close()
track_names = names.split("\n")

num_tracks = len(track_names)

for i in tqdm.tqdm(range(num_tracks)):
    frames = tracks[track_names[i]]["frames"]
    
    image_to_video(frames, track_names[i])
