import numpy as np
import torch

def whole2parts(motion, mode='t2m', window_size=None):
    if type(motion) == np.ndarray:
        aug_data = torch.from_numpy(motion).float()
    else:
        aug_data = motion
    if mode == 't2m':
        joints_num = 22
        s = 0
        e = 4
        root_data = aug_data[:, s:e]
        s = e
        e = e + (joints_num - 1) * 3
        ric_data = aug_data[:, s:e]
        s = e
        e = e + (joints_num - 1) * 6
        rot_data = aug_data[:, s:e]
        s = e
        e = e + joints_num * 3
        local_vel = aug_data[:, s:e]
        s = e
        e = e + 4
        feet = aug_data[:, s:e]
        R_L_idx = torch.Tensor([2, 5, 8, 11]).to(torch.int64)
        L_L_idx = torch.Tensor([1, 4, 7, 10]).to(torch.int64)
        B_idx = torch.Tensor([3, 6, 9, 12, 15]).to(torch.int64)
        R_A_idx = torch.Tensor([9, 14, 17, 19, 21]).to(torch.int64)
        L_A_idx = torch.Tensor([9, 13, 16, 18, 20]).to(torch.int64)
        nframes = root_data.shape[0]
        if window_size is not None:
            assert nframes == window_size
        ric_data = ric_data.reshape(nframes, -1, 3)
        rot_data = rot_data.reshape(nframes, -1, 6)
        local_vel = local_vel.reshape(nframes, -1, 3)
        root_data = torch.cat([root_data, local_vel[:, 0, :]], dim=1)
        R_L = torch.cat([ric_data[:, R_L_idx - 1, :], rot_data[:, R_L_idx - 1, :], local_vel[:, R_L_idx, :]], dim=2)
        L_L = torch.cat([ric_data[:, L_L_idx - 1, :], rot_data[:, L_L_idx - 1, :], local_vel[:, L_L_idx, :]], dim=2)
        B = torch.cat([ric_data[:, B_idx - 1, :], rot_data[:, B_idx - 1, :], local_vel[:, B_idx, :]], dim=2)
        R_A = torch.cat([ric_data[:, R_A_idx - 1, :], rot_data[:, R_A_idx - 1, :], local_vel[:, R_A_idx, :]], dim=2)
        L_A = torch.cat([ric_data[:, L_A_idx - 1, :], rot_data[:, L_A_idx - 1, :], local_vel[:, L_A_idx, :]], dim=2)
        Root = root_data
        R_Leg = torch.cat([R_L.reshape(nframes, -1), feet[:, 2:]], dim=1)
        L_Leg = torch.cat([L_L.reshape(nframes, -1), feet[:, :2]], dim=1)
        Backbone = B.reshape(nframes, -1)
        R_Arm = R_A.reshape(nframes, -1)
        L_Arm = L_A.reshape(nframes, -1)
    elif mode == 'kit':
        joints_num = 21
        s = 0
        e = 4
        root_data = aug_data[:, s:e]
        s = e
        e = e + (joints_num - 1) * 3
        ric_data = aug_data[:, s:e]
        s = e
        e = e + (joints_num - 1) * 6
        rot_data = aug_data[:, s:e]
        s = e
        e = e + joints_num * 3
        local_vel = aug_data[:, s:e]
        s = e
        e = e + 4
        feet = aug_data[:, s:e]
        R_L_idx = torch.Tensor([11, 12, 13, 14, 15]).to(torch.int64)
        L_L_idx = torch.Tensor([16, 17, 18, 19, 20]).to(torch.int64)
        B_idx = torch.Tensor([1, 2, 3, 4]).to(torch.int64)
        R_A_idx = torch.Tensor([3, 5, 6, 7]).to(torch.int64)
        L_A_idx = torch.Tensor([3, 8, 9, 10]).to(torch.int64)
        nframes = root_data.shape[0]
        if window_size is not None:
            assert nframes == window_size
        ric_data = ric_data.reshape(nframes, -1, 3)
        rot_data = rot_data.reshape(nframes, -1, 6)
        local_vel = local_vel.reshape(nframes, -1, 3)
        root_data = torch.cat([root_data, local_vel[:, 0, :]], dim=1)
        R_L = torch.cat([ric_data[:, R_L_idx - 1, :], rot_data[:, R_L_idx - 1, :], local_vel[:, R_L_idx, :]], dim=2)
        L_L = torch.cat([ric_data[:, L_L_idx - 1, :], rot_data[:, L_L_idx - 1, :], local_vel[:, L_L_idx, :]], dim=2)
        B = torch.cat([ric_data[:, B_idx - 1, :], rot_data[:, B_idx - 1, :], local_vel[:, B_idx, :]], dim=2)
        R_A = torch.cat([ric_data[:, R_A_idx - 1, :], rot_data[:, R_A_idx - 1, :], local_vel[:, R_A_idx, :]], dim=2)
        L_A = torch.cat([ric_data[:, L_A_idx - 1, :], rot_data[:, L_A_idx - 1, :], local_vel[:, L_A_idx, :]], dim=2)
        Root = root_data
        R_Leg = torch.cat([R_L.reshape(nframes, -1), feet[:, 2:]], dim=1)
        L_Leg = torch.cat([L_L.reshape(nframes, -1), feet[:, :2]], dim=1)
        Backbone = B.reshape(nframes, -1)
        R_Arm = R_A.reshape(nframes, -1)
        L_Arm = L_A.reshape(nframes, -1)
    else:
        raise Exception()
    return [Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm]
