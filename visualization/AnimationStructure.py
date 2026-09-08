import numpy as np

def joints(parents):
    return np.arange(len(parents), dtype=int)

def children_list(parents):

    def joint_children(i):
        return [j for (j, p) in enumerate(parents) if p == i]
    return list(map(lambda j: np.array(joint_children(j)), joints(parents)))
