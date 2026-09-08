import numpy as np
from visualization import Animation
from visualization import AnimationStructure
from visualization.Quaternions import Quaternions

class BasicInverseKinematics:

    def __init__(self, animation, positions, iterations=1, silent=True):
        self.animation = animation
        self.positions = positions
        self.iterations = iterations
        self.silent = silent

    def __call__(self):
        children = AnimationStructure.children_list(self.animation.parents)
        for i in range(self.iterations):
            for j in AnimationStructure.joints(self.animation.parents):
                c = np.array(children[j])
                if len(c) == 0:
                    continue
                anim_transforms = Animation.transforms_global(self.animation)
                anim_positions = anim_transforms[:, :, :3, 3]
                anim_rotations = Quaternions.from_transforms(anim_transforms)
                jdirs = anim_positions[:, c] - anim_positions[:, np.newaxis, j]
                ddirs = self.positions[:, c] - anim_positions[:, np.newaxis, j]
                jsums = np.sqrt(np.sum(jdirs ** 2.0, axis=-1)) + 1e-10
                dsums = np.sqrt(np.sum(ddirs ** 2.0, axis=-1)) + 1e-10
                jdirs = jdirs / jsums[:, :, np.newaxis]
                ddirs = ddirs / dsums[:, :, np.newaxis]
                angles = np.arccos(np.sum(jdirs * ddirs, axis=2).clip(-1, 1))
                axises = np.cross(jdirs, ddirs)
                axises = -anim_rotations[:, j, np.newaxis] * axises
                rotations = Quaternions.from_angle_axis(angles, axises)
                if rotations.shape[1] == 1:
                    averages = rotations[:, 0]
                else:
                    averages = Quaternions.exp(rotations.log().mean(axis=-2))
                self.animation.rotations[:, j] = self.animation.rotations[:, j] * averages
            if not self.silent:
                anim_positions = Animation.positions_global(self.animation)
                error = np.mean(np.sum((anim_positions - self.positions) ** 2.0, axis=-1) ** 0.5)
                print('[BasicInverseKinematics] Iteration %i Error: %f' % (i + 1, error))
        return self.animation
