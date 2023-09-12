#!/usr/bin/env python
# coding=utf-8
import os
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt


def vis_2d_mesh(ax, vertices, faces, size, color='k', linewidth=2):
    """
    vertices: (N, 2) [-1, 1]
    faces: start from 0
    size: (w, h)
    """
    w, h = size
    assert(np.min(faces) == 0)

    # drawing mesh with triangle only                                                                                                                                                                                                              
    vertices = ((vertices[:, :2] + 1) * 0.5 * np.array([w, h]))
    for face in faces:
        i, j, k = face
        ax.plot((vertices[i, 0], vertices[j, 0]), (vertices[i, 1], vertices[j, 1]), linewidth=linewidth, color=color)
        ax.plot((vertices[j, 0], vertices[k, 0]), (vertices[j, 1], vertices[k, 1]), linewidth=linewidth, color=color)
        ax.plot((vertices[k, 0], vertices[i, 0]), (vertices[k, 1], vertices[i, 1]), linewidth=linewidth, color=color)
    return ax  


def vis_test_recon(dump_dir, fname_list):
    for fname in fname_list:
        pkl = pickle.load(open(f"{dump_dir}/{fname}", "rb"))

        sdf_pred = pkl['sdf_pred']
        contours_pred = pkl['contours_pred']
        point_samples = pkl['point_samples']
        sdf_samples = pkl['sdf_samples']
        vertices = pkl['vertices']
        faces = pkl['faces']

        # vis_image = get_image_2d_mesh(sdf_pred, vertices, faces)
        fig, (ax1, ax2) = plt.subplots(1, 2)
       
        rand_idx = np.random.permutation(point_samples.shape[0])[:2000]
        ax1.scatter(point_samples[rand_idx, 0], point_samples[rand_idx, 1], c=(sdf_samples[rand_idx]>0).astype(np.float32), s=2)
        ax1.set_ylim(-1, 1)
        ax1.axis('square')
        ax1.invert_yaxis()

        ax2_im = ax2.imshow(sdf_pred, vmin=-0.1, vmax=0.1, cmap='seismic')
        # fig.colorbar(ax2_im)
        for contour in contours_pred:
            ax2.plot(contour[:, 1], contour[:, 0], linewidth=2, color='k')
        vis_2d_mesh(ax2, vertices, faces, size=sdf_pred.shape, color='g', linewidth=1)
     
        plt.show()
        plt.close()


def vis_recon_all(dump_dir, fname_list):
    fig, ax = plt.subplots(1, 1)
    for fname in fname_list:
        pkl = pickle.load(open(f"{dump_dir}/{fname}", "rb"))

        sdf_pred = pkl['sdf_pred']
        contours_pred = pkl['contours_pred']
        point_samples = pkl['point_samples']
        sdf_samples = pkl['sdf_samples']
        vertices = pkl['vertices']
        faces = pkl['faces']

        ax_im = ax.imshow(sdf_pred, vmin=-0.1, vmax=0.1, cmap='seismic')
        for contour in contours_pred:
            ax.plot(contour[:, 1], contour[:, 0], linewidth=2, color='k')
     
    ax.imshow(sdf_pred, vmin=-0.1, vmax=0.1, cmap='seismic')
    plt.show()
    plt.close()



# def vis_interp(dump_dir, vis_img_dir, fname_list):
#     fig, (ax1, ax2) = plt.subplots(1, 2)
#     for fname in fname_list:
#         # print(fname)
#         pkl = pickle.load(open(f"{dump_dir}/{fname}", "rb"))

#         sdf_pred = pkl['sdf_pred']
#         contours_pred = pkl['contours_pred']

#         ax2_im = ax2.imshow(sdf_pred, vmin=-0.1, vmax=0.1, cmap='seismic')
#         # fig.colorbar(ax2_im)
#         for contour in contours_pred:
#             ax2.plot(contour[:, 1], contour[:, 0], linewidth=2, color='k')
     
#     # plt.show()
#     plt.savefig(os.path.join(vis_img_dir, dump_dir.split('/')[-1]+'_all.png'))
#     plt.close()

#     fig, axs = plt.subplots(4, 8)
#     for i, fname in enumerate(fname_list):
#         ax = axs[i//8][i%8]
#         # print(fname)
#         pkl = pickle.load(open(f"{dump_dir}/{fname}", "rb"))

#         sdf_pred = pkl['sdf_pred']
#         contours_pred = pkl['contours_pred']

#         ax.imshow(sdf_pred, vmin=-0.1, vmax=0.1, cmap='seismic')
#         for contour in contours_pred:
#             ax.plot(contour[:, 1], contour[:, 0], linewidth=2, color='k')
#     # plt.show()
#     plt.savefig(os.path.join(vis_img_dir, dump_dir.split('/')[-1]+'_interps.png'))
#     plt.close()

def vis_interp(dump_dir, vis_img_dir, fname_list):
    for fname in fname_list[:-1]:
        
        pkl = pickle.load(open(f"{dump_dir}/{fname}", "rb"))

        contours_pred = pkl['contours_pred']

        # sdf_pred = pkl['sdf_pred']
        # contours_pred = pkl['contours_pred']

        # ax2_im = plt.imshow(sdf_pred, vmin=-0.1, vmax=0.1, cmap='seismic')
        for contour in contours_pred:
            plt.plot(contour[:, 1], contour[:, 0], linewidth=2, color='k')
            
     
    # plt.show()
    plt.savefig(os.path.join(vis_img_dir, dump_dir.split('/')[-1]+'_all.png'))
    plt.close()

    fig, axs = plt.subplots(4, 8)
    # fig, axs = plt.subplots(1, 2)
    for i, fname in enumerate(fname_list):
        ax = axs[i//8][i%8]
        # ax = axs[i%2]
        # print(fname)
        pkl = pickle.load(open(f"{dump_dir}/{fname}", "rb"))

        sdf_pred = pkl['sdf_pred']
        contours_pred = pkl['contours_pred']
        
        ax_ = ax.imshow(sdf_pred, vmin=-0.1, vmax=0.1, cmap='seismic')
        fig.colorbar(ax_)
        for contour in contours_pred:
            ax.plot(contour[:, 1], contour[:, 0], linewidth=2, color='k')
    # plt.show()
    plt.savefig(os.path.join(vis_img_dir, dump_dir.split('/')[-1]+'_interps.png'))
    plt.close()

def mkdir_ifnotexists(directory):
    if not os.path.exists(directory): 
        os.mkdir(directory)
    
if __name__ == '__main__':
#     # hand_x3
    # dump_dir = '../work_dir/Human2D/seq_hand_x_0_0_0/arap/hand_x3/sdf_seq_hand_x3_ARAP_SE4k_SLSLine_w1e-3/results/train/interp_sdf/3999/'
    exp_name = '0722_full_200000_surface_pts_8_hidden_layers_deepsdf_decoder'
    base_dir = os.path.join('/home/xiaoyan/4DRep_DeepSDF2D/work_dir', exp_name, 'run/results/train/interp_sdf/')
    interp_epoch = os.listdir(base_dir)
    vis_img_dir = os.path.join(base_dir, 'vis_imgs')
    mkdir_ifnotexists(vis_img_dir)
    for epoch in interp_epoch:
        dump_dir = os.path.join(base_dir, epoch)
        # print(dump_dir)
        # dump_dir = os.path.join(base_dir, f'18999')
        fname_list = os.listdir(dump_dir)
        fname_list.sort()
        vis_interp(dump_dir, vis_img_dir, fname_list)
    print(len(interp_epoch))

# from IPython import embed; embed()


# import numpy as np
# from PIL import Image
# from skimage import io
# import os
# base_dir = '/home/xiaoyan/4DRep_DeepSDF2D/work_dir/templete_static_siren_decoder_0721/run/results/train/interp_sdf/vis_imgs'
# imgs_list = []
# for idx in range(4, 45, 5):
#     img_path = os.path.join(base_dir, f"{idx:02d}99_all.png")

#     img = np.array(Image.open(img_path).convert("RGB"))
#     imgs_list.append(img)

# # img_path = os.path.join(base_dir, f"18999_all.png")

# # img = np.array(Image.open(img_path).convert("RGB"))
# # imgs_list.append(img)

# img_1 = np.concatenate([img for img in imgs_list[:3]], axis=1)
# img_2 = np.concatenate([img for img in imgs_list[3:6]], axis=1)
# img_3 = np.concatenate([img for img in imgs_list[6:9]], axis=1)
# # img_4 = np.concatenate([img for img in imgs_list[15:20]], axis=1)

# img = np.concatenate((img_1, img_2, img_3), axis=0)
# io.imsave(os.path.join(base_dir, 'vis_all.png'), img)