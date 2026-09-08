import numpy as np
import numpy.core.umath_tests as ut
from visualization.Quaternions import Quaternions

class Animation:

    def __init__(self, rotations, positions, orients, offsets, parents, names, frametime):
        self.rotations = rotations
        self.positions = positions
        self.orients = orients
        self.offsets = offsets
        self.parents = parents
        self.names = names
        self.frametime = frametime

    def __len__(self):
        return len(self.rotations)

    @property
    def shape(self):
        return (self.rotations.shape[0], self.rotations.shape[1])

    def copy(self):
        return Animation(self.rotations.copy(), self.positions.copy(), self.orients.copy(), self.offsets.copy(), self.parents.copy(), self.names, self.frametime)

def transforms_local(anim):
    transforms = anim.rotations.transforms()
    transforms = np.concatenate([transforms, np.zeros(transforms.shape[:2] + (3, 1))], axis=-1)
    transforms = np.concatenate([transforms, np.zeros(transforms.shape[:2] + (1, 4))], axis=-2)
    transforms[:, :, 0:3, 3] = anim.positions
    transforms[:, :, 3:4, 3] = 1.0
    return transforms

def transforms_multiply(t0s, t1s):
    return ut.matrix_multiply(t0s, t1s)

def transforms_blank(anim):
    ts = np.zeros(anim.shape + (4, 4))
    ts[:, :, 0, 0] = 1.0
    ts[:, :, 1, 1] = 1.0
    ts[:, :, 2, 2] = 1.0
    ts[:, :, 3, 3] = 1.0
    return ts

def transforms_global(anim):
    locals = transforms_local(anim)
    globals = transforms_blank(anim)
    globals[:, 0] = locals[:, 0]
    for i in range(1, anim.shape[1]):
        globals[:, i] = transforms_multiply(globals[:, anim.parents[i]], locals[:, i])
    return globals

def positions_global(anim):
    positions = transforms_global(anim)[:, :, :, 3]
    return positions[:, :, :3] / positions[:, :, 3, np.newaxis]
