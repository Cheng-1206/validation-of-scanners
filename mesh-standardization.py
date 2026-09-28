# environment--batch process
import numpy as np
import trimesh
import pymeshlab
import os

from sklearn.decomposition import PCA

def move_vertices_to_center(mesh):
    mesh2 = mesh.copy()
    vertices = np.array(mesh2.vertices)
    _min = np.min(vertices, axis=0)
    _max = np.max(vertices, axis=0)
    center = (_min + _max) / 2
    mesh2.vertices -= center
    return mesh2

def clean_mesh(meshset):
    meshset.apply_filter("meshing_remove_duplicate_faces")
    meshset.apply_filter("meshing_remove_duplicate_vertices")
    meshset.apply_filter("meshing_remove_unreferenced_vertices")
    meshset.apply_filter("meshing_remove_folded_faces")
    meshset.apply_filter("meshing_remove_null_faces")

    meshset.apply_filter("meshing_repair_non_manifold_edges", method=0)
    # ms.apply_filter("meshing_repair_non_manifold_vertices")

def close_holes(meshset):
    meshset.apply_filter("meshing_repair_non_manifold_edges", method=0)
    #     clean_mesh(ms)
    meshset.apply_filter("meshing_close_holes", maxholesize=10000, selfintersection=False)
    clean_mesh(meshset)

def rotate_center(mesh):
    mesh2 = mesh.copy()

    # for MV iphone scan process start
    angle2 = np.pi / 2
    direction2 = [1, 0, 0]
    center2 = [0, 0, 0]
    rot_matrix2 = trimesh.transformations.rotation_matrix(angle2, direction2, center2)

    mesh2.apply_transform(rot_matrix2)
    # for some iphone scan results end


    vertices = np.array(mesh2.vertices)
    _min = np.min(vertices, axis=0)
    _max = np.max(vertices, axis=0)
    model_height = _max-_min
    print(model_height)
    center = (_min + _max) / 2
    mesh2.vertices -= center

    return mesh2

def get_trunk_level_circumferences(tri_mesh, offset_low=None, offset_high=None, lc_size=64, margin=20):
    bounding_box_value = tri_mesh.bounding_box.bounds
    lowest_val = bounding_box_value[0][1]
    highest_val = bounding_box_value[1][1]
    if offset_low is None:
        offset_low = lowest_val + margin
    if offset_high is None:
        offset_high = highest_val - margin
    space = np.linspace(offset_low, offset_high, lc_size)
    v_dir = [0, 1, 0]

    res = []
    for s in space:
        new_origin = [0, s, 0]
        section = tri_mesh.section(plane_origin=new_origin, plane_normal=v_dir)
        #         slice_2D, _ = section.to_planar()

        if section is None:
            raise Exception(f"Section is None at offset {s}")

        if not section.is_closed:
            print(f"At offset {s}, has {len(section.entities)} parts")
            section.fill_gaps(distance=1e-5)
        res.append(float(section.length))
    return res


def run_center(filepath, targetpath):
    file = filepath
    target_file = targetpath
    tri_mesh = trimesh.load_mesh(file)
    tri_mesh = rotate_center(tri_mesh)
    tri_mesh = move_vertices_to_center(tri_mesh)
    tri_mesh.export(target_file)


def run_lc(targetpath, offset_low, offset_high):
    target_file = targetpath

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(target_file)
    close_holes(ms)

    mesh = ms.current_mesh()
    vertices = mesh.vertex_matrix()
    faces = mesh.face_matrix()
    tri_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    res = get_trunk_level_circumferences(tri_mesh, offset_low, offset_high)
    return res



def align_body_vertically(mesh): #mine PCA

    # Center the mesh
    vertices = mesh.vertices.copy()

    # Center vertices
    centroid = vertices.mean(axis=0)
    centered = vertices - centroid

    pca = PCA(n_components=3)
    pca.fit(centered)

    # PCA components (rows are principal directions)
    components = pca.components_  # shape (3,3)
    variances = pca.explained_variance_

    target_axes = np.array([
        [0, 0, 1],  # Z
        [1, 0, 0],  # X
        [0, 1, 0],  # Y
    ])

    # Build rotation matrix
    R = target_axes.T @ components

    # Ensure right-handed coordinate system
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1

    T = np.eye(4)
    T[:3, :3] = R

    mesh.apply_translation(-centroid)
    mesh.apply_transform(T)

    return mesh



def trim_turntable(mesh): #mine PCA
    mesh2 = mesh.copy()

    vertices2 = np.array(mesh2.vertices)
    _min2 = np.min(vertices2, axis=0)
    _max2 = np.max(vertices2, axis=0)

    z_threshold2 = (_max2[2] - _min2[2])/10 + _min2[2]
    xy_threshold2 = _max2[0]/5
    face_centers2 = mesh2.vertices[mesh2.faces].mean(axis=1)
    faces_to_keep_mask2_0 = face_centers2[:, 2] <= z_threshold2
    faces_to_keep_mask2_1 = face_centers2[:, 0] <= xy_threshold2
    faces_to_keep_mask2_2 = face_centers2[:, 0] >= -xy_threshold2
    faces_to_keep_mask2_3 = face_centers2[:, 1] <= xy_threshold2
    faces_to_keep_mask2_4 = face_centers2[:, 1] >= -xy_threshold2
    faces_to_keep_mask2 = faces_to_keep_mask2_0 & faces_to_keep_mask2_1 & faces_to_keep_mask2_2 & faces_to_keep_mask2_3 & faces_to_keep_mask2_4
    mesh2.update_faces(faces_to_keep_mask2)
    mesh2.remove_unreferenced_vertices()

    vertices_sample = np.array(mesh2.vertices)

    vertices = mesh.vertices
    vertices = np.array(vertices)
    _min = np.min(vertices, axis=0)
    _max = np.max(vertices, axis=0)

    if vertices_sample.size == 0:
        z_threshold = (_max[2] - _min[2]) / 30 + _min[2] # a bit too small
    else:
        _min_sample = np.min(vertices_sample, axis=0)
        _max_sample = np.max(vertices_sample, axis=0)

        tolerance = (_max[2] - _max_sample[2]) / 50  # 1/50 of pure height
        z_threshold = _max_sample[2] + tolerance

    face_centers = mesh.vertices[mesh.faces].mean(axis=1)
    faces_to_remove_mask = face_centers[:, 2] <= z_threshold
    mesh.update_faces(~faces_to_remove_mask)
    mesh.remove_unreferenced_vertices()

    return mesh


def trim_turntable_append(mesh):  # mine PCA

    vertices = mesh.vertices
    vertices = np.array(vertices)
    _min = np.min(vertices, axis=0)
    _max = np.max(vertices, axis=0)

    z_threshold = (_max[2] - _min[2])/10 + _min[2]
    xy_threshold = _max[0]*0.75
    face_centers = mesh.vertices[mesh.faces].mean(axis=1)
    faces_to_remove_mask_0 = face_centers[:, 2] <= z_threshold
    faces_to_remove_mask_1 = face_centers[:, 0] >= xy_threshold
    faces_to_remove_mask_2 = face_centers[:, 0] <= -xy_threshold
    faces_to_remove_mask_3 = face_centers[:, 1] >= xy_threshold
    faces_to_remove_mask_4 = face_centers[:, 1] <= -xy_threshold
    faces_to_remove_mask = faces_to_remove_mask_0 & (faces_to_remove_mask_1 | faces_to_remove_mask_2 | faces_to_remove_mask_3 | faces_to_remove_mask_4)
    mesh.update_faces(~faces_to_remove_mask)
    mesh.remove_unreferenced_vertices()

    return mesh


def measure2(f3, meshori): #mine PCA
    # get width of handle part of mesh
    vertices2 = np.array(meshori.vertices)
    _min2 = np.min(vertices2, axis=0)
    _max2 = np.max(vertices2, axis=0)

    zu_threshold2 = (_max2[2] - _min2[2])*2/3 + _min2[2]
    zl_threshold2 = (_max2[2] - _min2[2]) / 3 + _min2[2]
    face_centers2 = meshori.vertices[meshori.faces].mean(axis=1)
    faces_to_remove_mask2_1 = face_centers2[:, 2] >= zu_threshold2
    faces_to_remove_mask2_2 = face_centers2[:, 2] <= zl_threshold2
    faces_to_remove_mask2 = faces_to_remove_mask2_1 | faces_to_remove_mask2_2
    meshori.update_faces(~faces_to_remove_mask2)
    meshori.remove_unreferenced_vertices()

    vertices_mesh_middle = np.array(meshori.vertices)
    _min_mesh_middle = np.min(vertices_mesh_middle, axis=0)
    _max_mesh_middle = np.max(vertices_mesh_middle, axis=0)
    width_meshori = abs(_max_mesh_middle[0] - _min_mesh_middle[0])

    # get width of handle part of f3
    vertices1 = np.array(f3.vertices)
    _min1 = np.min(vertices1, axis=0)
    _max1 = np.max(vertices1, axis=0)
    width_f3 = abs(_max1[0] - _min1[0])

    scale = width_f3 / width_meshori

    return scale

def get_scale(f3path, meshoripath):
    f3 = trimesh.load_mesh(f3path)
    meshori = trimesh.load_mesh(meshoripath)
    meshori = align_body_vertically(meshori)  # x hands; y thickness; z height
    meshori = trim_turntable(meshori)
    meshori = trim_turntable_append(meshori)
    meshori = align_body_vertically(meshori)
    scale = measure2(f3, meshori)

    return scale



if __name__== '__main__':

    dir_f3_path = "./pair_data/pair_fit3d"
    dir_meshori_path = "./pair_data/pair_polycam"

    dir_mesh_path = "./pair_data/pair_fit3d_final"
    target_path = "./pair_data/pair_polycam_final"

    files = [file for file in os.listdir(dir_f3_path)]

    for file in files:
        f3path = os.path.join(dir_f3_path, file)
        meshoripath = os.path.join(dir_meshori_path, file.split(".")[0]+"_mesh.obj")

        meshpath = os.path.join(dir_mesh_path, file.split(".")[0]+"_mesh.obj")
        targetpath = os.path.join(target_path, file.split(".")[0]+"_mesh.obj")

        scale = get_scale(f3path, meshoripath)
        print(meshpath)
        print(scale)
        trimmed_mesh = trimesh.load_mesh(meshpath)
        trimmed_mesh.apply_scale(scale)
        trimmed_mesh.export(targetpath)

