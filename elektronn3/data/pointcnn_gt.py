
# Copyright (c) 2018 Philipp J. Schubert
# All rights reserved

# download/import all necessary work packages
import numpy as np
from knossos_utils.skeleton_utils import load_skeleton
from sklearn.neighbors import KDTree
from syconn.proc.graphs import bfs_smoothing
from syconn.reps.super_segmentation import SuperSegmentationObject
import re
import os
from syconn.proc.graphs import mesh2batch_gt, mesh2batch_gt_boxes
from syconn.mp.shared_mem import start_multiprocess_imap
from sklearn.model_selection import train_test_split
import time
from syconn.reps.super_segmentation_object import SuperSegmentationObject
start_time = time.time()

# create function that converts information in string type to the information in integer type
def str2intconverter(comment, gt_type):
    if gt_type == "axgt":
        if comment == "gt_axon":
            return 1
        elif comment == "gt_dendrite":
            return 0
        elif comment == "gt_soma":
            return 2
        else:
            return -1
    elif gt_type == "spgt":
        if "head" in comment:
            return 1
        elif "neck" in comment:
            return 0
        elif "shaft" in comment:
            return 2
        elif "other" in comment:
            return 3
        else:
            return -1
    else: raise ValueError("Given groundtruth type is not valid.")


def get_vertex_labels(kzip_path, gt_type="spgt", n_voting=40):
    """

    Parameters
    ----------
    kzip_path : str
    gt_type :  str

    Returns
    -------
    vertices : np.array
    vertex_labels : np.array
    """
    #assert gt_type in ["axgt", "spgt"], "Currently only spine and axon GT is supported"  #n_labels = 3 if gt_type == "axgt" else 4
    sso_id = int(re.findall("/(\d+).", kzip_path)[0])
    ssv = SuperSegmentationObject(sso_id, version=gt_type)
    indices, vertices, normals = ssv.mesh

    # # Load mesh
    vertices = vertices.reshape((-1, 3))

    # load skeleton
    skel = load_skeleton(kzip_path)["skeleton"]
    skel_nodes = list(skel.getNodes())

    node_coords = np.array([n.getCoordinate() * sso.scaling for n in skel_nodes])
    node_labels = np.array([str2intconverter(n.getComment(), gt_type) for n in skel_nodes], dtype=np.int)
    node_coords = node_coords[(node_labels != -1)]
    node_labels = node_labels[(node_labels != -1)]

    # create KD tree from skeleton node coordinates
    tree = KDTree(node_coords)
    # transfer labels from skeleton to mesh
    dist, ind = tree.query(vertices, k=1)
    vertex_labels = node_labels[ind]

    vertex_labels = bfs_smoothing(vertices, vertex_labels, n_voting=n_voting)
    dest_folder = os.path.expanduser("~") + "/spine_gt_pointcloud/"
    if not os.path.isdir(dest_folder):
        os.makedirs(dest_folder)
    np.save('{}/sso_{}_verts.k.zip'.format(dest_folder, sso_id), vertices)
    np.save('{}/sso_{}_vertlabels.k.zip'.format(dest_folder, sso_id), vertex_labels)
    return vertices, vertex_labels


def point_GT_generation(vertices_paths, vertex_labels_paths, dest_dir=None): #PATHS
    """
    Generates a .npy GT file from all kzip paths.
    Parameters
    ----------
    Returns
    -------
    """
    if dest_dir is None:
        dest_dir = os.path.expanduser("~") + "/spine_gt_pointcloud/"
    params = [(vp, vlp) for vp, vlp in zip(vertices_paths, vertex_labels_paths)]
    res = start_multiprocess_imap(gt_generation_helper, params, nb_cpus=5,
                                  debug=True)

    # Create Dataset splits for training and validation
    all_raw_data = []
    all_label_data = []
    for i in range(len(vertices_paths)):
        all_raw_data.append(res[i][0])
        all_label_data.append(res[i][1])
    all_raw_data = np.concatenate(all_raw_data)
    all_label_data = np.concatenate(all_label_data)
    print("Shuffling batches.")
    np.random.seed(0)
    ixs = np.arange(len(all_raw_data))
    np.random.shuffle(ixs)
    all_raw_data = all_raw_data[ixs]
    all_label_data = all_label_data[ixs]
    raw_train, raw_valid, label_train, label_valid = \
        train_test_split(all_raw_data, all_label_data, train_size=0.85, shuffle=False)
    print("Writing npy files.")
    np.save('{}/raw_train.npy'.format(dest_dir), raw_train)
    np.save('{}/label_train.npy'.format(dest_dir), label_train)
    np.save('{}/raw_valid.npy'.format(dest_dir), raw_valid)
    np.save('{}/label_valid.npy'.format(dest_dir), label_valid)


def gt_generation_helper(args):
    vertices_path, vertex_labels_path = args
    vertices = np.load(vertices_path)
    vertex_labels = np.load(vertex_labels_path)
    sso_id = int(re.findall("_(\d+)_", vertices_path)[0])
    ssv = SuperSegmentationObject(sso_id)
    sample_locs = np.concatenate(ssv.sample_locations())
    batch, batch_label = mesh2batch_gt_boxes(vertices, vertex_labels, sample_locs)
    return batch, batch_label


if __name__ == "__main__":
    label_file_folder = "/wholebrain/u/shum/spine_gt_pointcloud/gt_phil/"
    file_names_vertices = ["/sso_4741011_verts.k.zip.npy",
                           "/sso_23044610_verts.k.zip.npy",
                           "/sso_18279774_verts.k.zip.npy",
                           "/sso_26331138_verts.k.zip.npy",
                           "/sso_27965455_verts.k.zip.npy"]
    file_names_vertlabels = ["/sso_4741011_vertlabels.k.zip.npy",
                             "/sso_23044610_vertlabels.k.zip.npy",
                             "/sso_18279774_vertlabels.k.zip.npy",
                             "/sso_26331138_vertlabels.k.zip.npy",
                             "/sso_27965455_vertlabels.k.zip.npy"]
    file_paths_vertices = [label_file_folder + "/" + fname for fname in file_names_vertices]
    file_paths_vertlabels = [label_file_folder + "/" + fname for fname in file_names_vertlabels]
    point_GT_generation(file_paths_vertices, file_paths_vertlabels)


    # file_names = ["/23044610.037.k.zip", "/4741011.074.k.zip",
    #               "/18279774.089.k.zip", "/26331138.046.k.zip",
    #               "/27965455.039.k.zip"]
    # file_paths = [label_file_folder + "/" + fname for fname in file_names][::-1]
    #get_vertex_labels(file_paths)

    print("---Runtime is %s seconds ---" % (time.time() - start_time))