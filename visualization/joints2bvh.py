from pathlib import Path
import visualization.Animation as Animation
from visualization.InverseKinematics import BasicInverseKinematics
from visualization.Quaternions import Quaternions
import visualization.BVH_mod as BVH
from visualization.remove_fs import remove_fs

class Joint2BVHConvertor:

    def __init__(self):
        self.template = BVH.load(str(Path(__file__).parent / 'data/template.bvh'), need_quater=True)
        self.re_order = [0, 1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12, 15, 13, 16, 18, 20, 14, 17, 19, 21]
        self.re_order_inv = [0, 1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12, 14, 18, 13, 15, 19, 16, 20, 17, 21]

    def convert(self, positions, filename, iterations=10, foot_ik=True):
        positions = positions[:, self.re_order]
        new_anim = self.template.copy()
        new_anim.rotations = Quaternions.id(positions.shape[:-1])
        new_anim.positions = new_anim.positions[0:1].repeat(positions.shape[0], axis=-0)
        new_anim.positions[:, 0] = positions[:, 0]
        if foot_ik:
            positions = remove_fs(positions, None, fid_l=(3, 4), fid_r=(7, 8), interp_length=5, force_on_floor=True)
        ik_solver = BasicInverseKinematics(new_anim, positions, iterations=iterations, silent=True)
        new_anim = ik_solver()
        glb = Animation.positions_global(new_anim)[:, self.re_order_inv]
        if filename is not None:
            BVH.save(filename, new_anim, names=new_anim.names, frametime=1 / 20, order='zyx', quater=True)
        return (new_anim, glb)
