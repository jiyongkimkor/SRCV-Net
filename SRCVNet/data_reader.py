import cv2
import random
import numpy as np
import scipy.signal as sig


kx = np.array([[-1, 0, 1]])
ky = np.array([[-1], [0], [1]])


def read_left(filename):
    rgb = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    rdx, rdy = sig.convolve2d(rgb[:, :, 2], kx, 'same'), sig.convolve2d(rgb[:, :, 2], ky, 'same')
    gdx, gdy = sig.convolve2d(rgb[:, :, 1], kx, 'same'), sig.convolve2d(rgb[:, :, 1], ky, 'same')
    bdx, bdy = sig.convolve2d(rgb[:, :, 0], kx, 'same'), sig.convolve2d(rgb[:, :, 0], ky, 'same')
    dx = cv2.merge([bdx, gdx, rdx])
    dy = cv2.merge([bdy, gdy, rdy])
    rgb = rgb.astype('float32') / 127.5 - 1.0
    dx = dx.astype('float32') / 127.5
    dy = dy.astype('float32') / 127.5
    return rgb, dx, dy


def read_right(filename):
    image = cv2.imread(filename, cv2.IMREAD_UNCHANGED).astype('float32') / 127.5 - 1.0
    return image


def read_disp(filename):
    disp = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    disp_8x = cv2.resize(disp, (128, 128)) / 8.0
    disp_4x = cv2.resize(disp, (256, 256)) / 4.0
    disp = np.expand_dims(disp, -1)
    disp_8x = np.expand_dims(disp_8x, -1)
    disp_4x = np.expand_dims(disp_4x, -1)
    return disp_8x, disp_4x, disp


def read_batch(left_paths, right_paths, disp_paths):
    lefts, dxs, dys, rights, d8s, d4s, ds = [], [], [], [], [], [], []
    for left_path, right_path, disp_path in zip(left_paths, right_paths, disp_paths):
        left, dx, dy = read_left(left_path)
        right = read_right(right_path)
        d8, d4, d = read_disp(disp_path)
        lefts.append(left)
        dxs.append(dx)
        dys.append(dy)
        rights.append(right)
        d8s.append(d8)
        d4s.append(d4)
        ds.append(d)
    return np.array(lefts), np.array(rights), np.array(dxs), np.array(dys),\
            np.array(d8s), np.array(d4s), np.array(ds)


def load_batch(all_left_paths, all_right_paths, all_disp_paths, batch_size=2, reshuffle=False):
    assert len(all_left_paths) == len(all_disp_paths)
    assert len(all_right_paths) == len(all_disp_paths)

    i = 0
    while True:
        lefts, rights, dxs, dys, d8s, d4s, ds = read_batch(
            left_paths=all_left_paths[i * batch_size:(i + 1) * batch_size],
            right_paths=all_right_paths[i * batch_size:(i + 1) * batch_size],
            disp_paths=all_disp_paths[i * batch_size:(i + 1) * batch_size])
        yield [lefts, rights, dxs, dys], [d8s, d4s, ds]
        
        i = (i + 1) % (len(all_left_paths) // batch_size)
        if reshuffle:
            if i == 0:
                paths = list(zip(all_left_paths, all_right_paths, all_disp_paths))
                random.shuffle(paths)
                all_left_paths, all_right_paths, all_disp_paths = zip(*paths)


# +
import tensorflow as tf

class BatchLoader(tf.keras.utils.Sequence):
    def __init__(self, all_left_paths, all_right_paths, all_disp_paths, batch_size=2, reshuffle=False):
        assert len(all_left_paths) == len(all_disp_paths)
        assert len(all_right_paths) == len(all_disp_paths)
        
        
        self.all_left_paths = all_left_paths
        self.all_right_paths = all_right_paths
        self.all_disp_paths = all_disp_paths
        self.batch_size = batch_size
        self.reshuffle = reshuffle
        self.total_batches = len(all_left_paths) // batch_size

    def __len__(self):
        return self.total_batches

    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = (index + 1) * self.batch_size
        lefts, rights, dxs, dys, d8s, d4s, ds = read_batch(
            left_paths=self.all_left_paths[start_idx:end_idx],
            right_paths=self.all_right_paths[start_idx:end_idx],
            disp_paths=self.all_disp_paths[start_idx:end_idx]
        )
        return [lefts, rights, dxs, dys], [d8s, d4s, ds]

        if self.reshuffle and index == self.total_batches - 1:
            paths = list(zip(self.all_left_paths, self.all_right_paths, self.all_disp_paths))
            random.shuffle(paths)
            self.all_left_paths, self.all_right_paths, self.all_disp_paths = zip(*paths)
