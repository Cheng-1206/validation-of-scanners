# environment--batch process
import numpy as np
import trimesh
import pymeshlab
import os
import xlwt

# from sklearn.decomposition import PCA

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


def get_msrpoints(f3_path, mesh_path): #mine PCA


    f3 = trimesh.load_mesh(f3_path)
    mesh = trimesh.load_mesh(mesh_path)

    vertices_f3 = np.array(f3.vertices)
    _min_f3 = np.min(vertices_f3, axis=0)
    _amax_f3 = np.argmax(vertices_f3, axis=0)
    _max_f3 = np.max(vertices_f3, axis=0)
    _amax_f3 = np.argmax(vertices_f3, axis=0)

    vertices_mesh = np.array(mesh.vertices)
    _min_mesh = np.min(vertices_mesh, axis=0)
    _max_mesh = np.max(vertices_mesh, axis=0)

    z_handle = vertices_f3[_amax_f3][0][2]
    max_z = np.minimum(_max_f3[2], _max_mesh[2])  # estimated head top
    min_z = np.minimum(_min_f3[2], _min_mesh[2])  # estimated feet bottom


    ########  get calf point  #############
    f3_4calf = f3.copy()

    z_threshold_4calf_upper = min_z + (max_z - min_z) / 3  # /4 for some special
    z_threshold_4calf_lower = min_z + (max_z - min_z) / 15
    face_centers_4calf = f3_4calf.vertices[f3_4calf.faces].mean(axis=1)
    faces_to_remove_mask_upper = face_centers_4calf[:, 2] >= z_threshold_4calf_upper
    faces_to_remove_mask_lower = face_centers_4calf[:, 2] <= z_threshold_4calf_lower
    faces_to_remove_mask_4calf = (faces_to_remove_mask_upper | faces_to_remove_mask_lower)
    f3_4calf.update_faces(~faces_to_remove_mask_4calf)
    f3_4calf.remove_unreferenced_vertices()

    vertices_4calf = np.array(f3_4calf.vertices)
    _amin_4calf = np.argmin(vertices_4calf, axis=0)
    z_calf = vertices_4calf[_amin_4calf][1][2]

    test_max_calf = np.max(vertices_4calf, axis=0)
    test_min_calf = np.min(vertices_4calf, axis=0)
    print("top calf:", test_max_calf[2])
    print("bottom calf:", test_min_calf[2])
    # f3_4calf.export("E:/polyPro/pair_data/test")


    ########  get belly point  #############
    f3_4torso = f3.copy()

    z_threshold_4torso_upper = max_z - (max_z - min_z) / 3 # * 0.4 for special
    z_threshold_4torso_lower = max_z - (max_z - min_z) * 0.55
    face_centers_4torso = f3_4torso.vertices[f3_4torso.faces].mean(axis=1)
    faces_to_remove_mask_upper = face_centers_4torso[:, 2] >= z_threshold_4torso_upper
    faces_to_remove_mask_lower = face_centers_4torso[:, 2] <= z_threshold_4torso_lower
    faces_to_remove_mask_right = face_centers_4torso[:, 0] >= _max_f3[0] / 2
    faces_to_remove_mask_left = face_centers_4torso[:, 0] <= _min_f3[0] / 2
    faces_to_remove_maskf3_4torso = (faces_to_remove_mask_upper | faces_to_remove_mask_lower | faces_to_remove_mask_right | faces_to_remove_mask_left)
    f3_4torso.update_faces(~faces_to_remove_maskf3_4torso)
    f3_4torso.remove_unreferenced_vertices()

    vertices_4torso = np.array(f3_4torso.vertices)
    _amax_4torso = np.argmax(vertices_4torso, axis=0)
    z_belly = vertices_4torso[_amax_4torso][1][2]

    ########  get hip point  #############
    _min_f3_4torso = np.min(vertices_4torso, axis=0)
    _max_f3_4torso = np.max(vertices_4torso, axis=0)
    face_centers_4lowertorso = f3_4torso.vertices[f3_4torso.faces].mean(axis=1)
    faces_to_remove_maskf3_4lowertorso = (face_centers_4lowertorso[:, 2] >= (_max_f3_4torso[2] - (_max_f3_4torso[2] - _min_f3_4torso[2])/3))
    f3_4torso.update_faces(~faces_to_remove_maskf3_4lowertorso)
    f3_4torso.remove_unreferenced_vertices()

    vertices_4lowertorso = np.array(f3_4torso.vertices)
    _amin_4lowertorso = np.argmin(vertices_4lowertorso, axis=0)
    z_hip = vertices_4lowertorso[_amin_4lowertorso][1][2]

    markers = np.zeros(4)
    markers[0] = z_calf  # calf point
    markers[1] = z_hip  # hip point
    markers[2] = z_belly  # belly point
    markers[3] = z_handle + 70  # wrist point

    return markers


def get_mainmsrmnts(obj_path, markers):
    # 1st part
    obj_file = obj_path

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(obj_file)
    close_holes(ms)

    obj = ms.current_mesh()
    vertices = obj.vertex_matrix()
    faces = obj.face_matrix()
    tri_mesh_obj = trimesh.Trimesh(vertices=vertices, faces=faces)


    # 2nd part
    all_circumferences = np.array([])
    v_dir = [0, 0, 1]

    for s in [0, 1, 2, 3]: # upper 3 markers
        new_origin = [0, 0, markers[s]]
        section = tri_mesh_obj.section(plane_origin=new_origin, plane_normal=v_dir)
        #         slice_2D, _ = section.to_planar()

        if section is None:
            raise Exception(f"Section is None at offset {markers[s]}")

        if not section.is_closed:
            print(f"At offset {markers[s]}, has {len(section.entities)} parts")
            section.fill_gaps(distance=1e-5)
        # res.append(float(section.length))


        if section is not None:
            # Convert to 2D to get easy planar lengths
            section_2d, transform = section.to_planar()
            # get all closed edges
            polygons = section_2d.polygons_full

            if len(polygons) == 0:
                return None

            # calculate centroid
            centroids = np.array([poly.centroid for poly in polygons])

            x_coords = np.array([pt.x for pt in centroids])
            sorted_indices = np.argsort(x_coords)
            sorted_polygons = [polygons[i] for i in sorted_indices]

            # calculate circumference
            circumferences = [poly.length for poly in sorted_polygons]

            # largest_length = 0
            # # Iterate through entities and get their length
            # for i, entity in enumerate(section_2d.entities):
            #     # Length of specific entity
            #     length = entity.length(section_2d.vertices)
            #     print(f"Entity {i}: {length}")
            #     largest_length = max(largest_length, length)
            all_circumferences = np.concatenate((all_circumferences, np.array(circumferences)), axis=0)

    return all_circumferences



if __name__== '__main__':

    dir_f3_path = "./pair_data/pair_fit3d_final"
    dir_mesh_path = "./pair_data/pair_polycam_final_registered"
    target_excel_path = "./pair_data/measurements.xls"

    wb = xlwt.Workbook()
    sheet1 = wb.add_sheet('Sheet1')
    row = 1

    files = [file for file in os.listdir(dir_f3_path)]

    for file in files:
        f3path = os.path.join(dir_f3_path, file)
        meshpath = os.path.join(dir_mesh_path, file.split(".")[0]+"_mesh.obj")
        sheet1.write(row, 0, file.split(".")[0])  # first column write file name
        print(file.split(".")[0])

        msrpoints = get_msrpoints(f3path, meshpath)
        print("f3 processing")
        f3_main_msrmts = get_mainmsrmnts(f3path, msrpoints)  # get_mainmsrmnts_sw gets shoulder width instead of shoulder circumference
        print("mesh processing")
        mesh_main_msrmts = get_mainmsrmnts(meshpath, msrpoints)  # get_mainmsrmnts_sw gets shoulder width instead of shoulder circumference
        msr_f3_mesh = np.concatenate((f3_main_msrmts, mesh_main_msrmts), axis=0)
        for i in range(4):  # size of msrpoints
            sheet1.write(row, i + 1, msrpoints[i])
        if msr_f3_mesh.size < 50:
            for i in range(msr_f3_mesh.size):
                sheet1.write(row, i + 5, msr_f3_mesh[i])
        else:
            print(f"Scan {file} has big errors, please check later")

        row = row + 1

    wb.save(target_excel_path)

