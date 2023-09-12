import os
import pickle
import matplotlib.pyplot as plt

def sdf2udf():

    base_dir = "/home/xiaoyan/4DRep_DeepSDF2D/data/simple_data/raw/partial_observation_0714/Full_sdf"

    write_dir = "/home/xiaoyan/4DRep_DeepSDF2D/data/simple_data/raw/partial_observation_0714/Full_udf"

    fname_list = os.listdir(base_dir)
    for fname in fname_list:
        with open(os.path.join(base_dir, fname), 'rb') as f:
            pck = pickle.load(f)

        # import ipdb; ipdb.set_trace()
        pck['sdf'] = abs(pck['sdf'])
        with open(os.path.join(write_dir, fname), 'wb') as f:
            pickle.dump(pck, f)

def visualize():
    vis_dir = base_dir = "/home/xiaoyan/4DRep_DeepSDF2D/data/simple_data/raw/base/sdf"
    save_dir = "/home/xiaoyan/4DRep_DeepSDF2D/data/simple_data/raw/base/vis"

    fname_list = os.listdir(vis_dir)
    for fname in fname_list:
        with open(os.path.join(vis_dir, fname), 'rb') as f:
            pck = pickle.load(f)
        # import ipdb; ipdb.set_trace()
        img = plt.scatter(pck['points'][:,0], pck['points'][:,1], c=pck['sdf'])
        plt.colorbar(img)
        plt.savefig(os.path.join(save_dir, fname.split('.')[0]+".png"))
        plt.clf()
        



if __name__ == '__main__':
    # sdf2udf()
    visualize()