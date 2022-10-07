import os
import glob
import numpy as np

import shutil

from tqdm import tqdm


def run(folder_name):    
    file_names = glob.glob(os.path.join(folder_name, '*.npy'))

    n_episode = len(file_names)
    dataset = []
    rewards = []
    print("find %d npy files!" % len(file_names))
    for file_name in file_names:
        data = np.load(file_name, allow_pickle=True).item()
        dataset.append(data)
        rewards.append(data["reward"].sum())

    rewards = np.array(rewards, dtype=np.float32)
    flags = rewards>np.median(rewards)
    eq_flags = rewards==np.median(rewards)
    for i in tqdm(range(len(eq_flags))):
        if (flags.sum() < len(flags)/2) and eq_flags[i]:
            flags[i] = True    

    folder_name_output_optimal = "{}/optimal".format(folder_name)
    os.makedirs(folder_name_output_optimal, exist_ok=True)
    folder_name_output_suboptimal = "{}/suboptimal".format(folder_name)
    os.makedirs(folder_name_output_suboptimal, exist_ok=True)

    for i in tqdm(range(n_episode)):
        basename = os.path.basename(file_names[i])
        if flags[i]:
            shutil.move(file_names[i], '{}/{}'.format(folder_name_output_optimal, basename))
        else:
            shutil.move(file_names[i], '{}/{}'.format(folder_name_output_suboptimal, basename))

def main():
    run("./demonstrations")

if __name__=="__main__":
    main()