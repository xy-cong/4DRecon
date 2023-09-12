import os
import matplotlib.pyplot as plt
import pickle
from PIL import Image
import numpy as np
from skimage import io


if __name__ == '__main__':
    # exp_name_list = ['full', 'x3', 'x5', 'full_partial']
    # profix = ['_sdf_base', '_sdf_base_topo', '_udf_abs', '_udf_no_abs']
    # base_dir = "/home/xiaoyan/4DRep_DeepSDF2D/work_dir"
    # save_dir = "/home/xiaoyan/4DRep_DeepSDF2D/visualization/vis_imgs"
    # for exp_name in exp_name_list:
        
    #     exp_img = []
    #     for pro in profix:
    #         if pro == '_sdf_base_topo':
    #             mid_part = 'seq_hand_arap_topo'
    #             prefix = 'hand_' + exp_name + '_sdf_base_topo'
    #             prefix_2 = 'hand_' + exp_name + '_sdf_base'
    #         else:
    #             mid_part = 'seq_hand_arap'
    #             prefix = 'hand_' + exp_name + pro
    #             prefix_2 = 'hand_' + exp_name + pro

    #         exp_results_dir = os.path.join(base_dir, prefix, mid_part, prefix_2, 'results', 'train', 'interp_sdf', 'vis_imgs')
    #         imgs_col = []
    #         for idx in ['3999_all.png', '6999_all.png', '9999_all.png']:
    #             if idx == '9999_all.png' and exp_name == 'x3' and pro == '_sdf_base_topo':
    #                 idx = '11999_all.png'
    #             img_path = os.path.join(exp_results_dir, idx)
    #             imgs_col.append(np.array(Image.open(img_path).convert('RGB')))
    #         imgs_col = np.concatenate(imgs_col, axis=0)
    #         exp_img.append(imgs_col)
    #     exp_img = np.concatenate(exp_img, axis=1)
    #     io.imsave(os.path.join(save_dir, 'hand_'+exp_name+'.png'), exp_img)
    
    img_1 = np.array(Image.open("/home/xiaoyan/4DRep_DeepSDF2D/work_dir/hand_full_partial_sdf_base/seq_hand_arap/hand_full_partial_sdf_base/results/train/interp_sdf/vis_imgs/3999_all.png").convert("RGB"))
    img_2 = np.array(Image.open("/home/xiaoyan/4DRep_DeepSDF2D/work_dir/hand_full_partial_sdf_base/seq_hand_arap/hand_full_partial_sdf_base/results/train/interp_sdf/vis_imgs/6999_all.png").convert("RGB"))
    img_3 = np.array(Image.open("/home/xiaoyan/4DRep_DeepSDF2D/work_dir/hand_full_partial_sdf_base/seq_hand_arap/hand_full_partial_sdf_base/results/train/interp_sdf/vis_imgs/9999_all.png").convert("RGB"))

    img = np.concatenate((img_1, img_2, img_3), axis=1)

    io.imsave("/home/xiaoyan/4DRep_DeepSDF2D/visualization/vis_imgs/partial.png", img)






