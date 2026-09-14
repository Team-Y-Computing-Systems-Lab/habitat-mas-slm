import gzip 
import json 

# path = "/home/rlwagun/files/simulation-test/habitat-lab/data/datasets/hssd/0/hssd_height_per.json.gz"
path = "/home/rlwagun/files/simulation-test/habitat-lab/data/datasets/rearrange_pick/replica_cad/v0/rearrange_pick_replica_cad_v0/train/train_counter_L_analysis_5000_500.json.gz"



with gzip.open(path, 'rt') as f: 
    data = json.load(f) 

print(data.keys())
print("episodes", len(data["episodes"]))

episode = data["episodes"][0]

for key, value in episode.items():
    print(f"\n ---- {key} ---- ")
    print(f"value: {value}" )