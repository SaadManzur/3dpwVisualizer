import os
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from utils.pose import draw_skeleton

INDICES = [0, 2, 5, 8, 1, 4, 7, 12, 16, 18, 20, 17, 19, 21]

class TDPWDataset(object):

    def __init__(self, spath, ipath):
        
        self._spath = spath
        self._ipath = ipath

    def load_sequence(self, spath, seq):

        seq_file = os.path.join(spath, f"{seq}.pkl")

        data = np.load(seq_file, encoding='latin1', allow_pickle=True)

        intrinsic = data['cam_intrinsics']
        f = np.array([intrinsic[0, 0], intrinsic[1, 1]]).reshape((2, 1))
        c = np.array(intrinsic[:2, 2]).reshape((2, 1))

        jnts_2d = []

        subjects = data['jointPositions']

        for i_sub in range(len(subjects)):

            subject = subjects[i_sub]

            jnts_2d.append([])

            for i_frame in range(subject.shape[0]):

                cam_pose = data['cam_poses'][i_frame, :, :]
                R = cam_pose[:3, :3]
                T = cam_pose[:3, 3].reshape((3, -1))

                pos3d_world = subject[i_frame].reshape((24, 3))[INDICES, :]
                pos3d_world_h = np.vstack((pos3d_world.T, np.ones((1, 14))))
                pos3d_cam = np.matmul(cam_pose, pos3d_world_h).T[:, :3]
                proj_2d = np.divide(pos3d_cam[:, :2], pos3d_cam[:, 2:])
                pixel_2d = f[:, 0] * proj_2d + c[:, 0]

                jnts_2d[i_sub].append(pixel_2d)

            jnts_2d[i_sub] = np.array(jnts_2d[i_sub])
        
        assert len(jnts_2d) == len(subjects)

        return jnts_2d, c*2

    def visualize(self, joints, ipath, seq, width=1920, height=1080):

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)

        plt.ion()
        plt.show()

        for i in range(joints[0].shape[0]):

            img_path = os.path.join(os.path.join(ipath, seq), f"image_{i:05d}.jpg")

            ax.imshow(plt.imread(img_path))

            for j in range(len(joints)):

                draw_skeleton(joints[j][i, :, :], ax)

            ax.set_xlim((0, width))
            ax.set_ylim((height, 0))

            plt.draw()
            plt.pause(0.0001)
            ax.clear()

    def view(self, seq):

        joints, wh = self.load_sequence(self._spath, seq)

        self.visualize(joints, self._ipath, seq, wh[0, 0], wh[1, 0])