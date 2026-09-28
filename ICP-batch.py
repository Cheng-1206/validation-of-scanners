import open3d as o3d
import numpy as np
import copy
import os




def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])

def find_best_scale(source, target, threshold, trans_init):
    best_scale = 0.95
    smallest_rmse = 100

    for scale_value in np.arange(0.95, 1.10, 0.01):
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        source_temp.scale(scale_value, center=source_temp.get_center())
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source_temp, target_temp, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
        evaluation = o3d.pipelines.registration.evaluate_registration(
            source_temp, target_temp, threshold, reg_p2p.transformation)
        # print("scale: ", scale_value)
        # print("Evaluation: ", evaluation)
        if evaluation.inlier_rmse < smallest_rmse:
            best_scale = scale_value
            smallest_rmse = evaluation.inlier_rmse

    return best_scale

if __name__== '__main__':

    dir_f3_path = "./pair_data/pair_fit3d_final"
    dir_mesh_path = "./pair_data/pair_polycam_final"
    target_path = "./pair_data/pair_polycam_final_registered"

    demo_icp_pcds = o3d.data.DemoICPPointClouds()
    threshold = 500000
    # for raw pc files
    trans_init = np.asarray([[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]])

    files = [file for file in os.listdir(dir_f3_path)]

    for file in files:
        f3path = os.path.join(dir_f3_path, file)
        meshpath = os.path.join(dir_mesh_path, file.split(".")[0]+"_mesh.obj")
        targetpath = os.path.join(target_path, file.split(".")[0]+"_mesh.obj")


        target_mesh = o3d.io.read_triangle_mesh(f3path)  # blue fit3d mesh "E:/polyPro/pair_data/pair_f3_final/102-1-1.obj"
        target = o3d.geometry.PointCloud()
        target.points = target_mesh.vertices
        target = target.farthest_point_down_sample(10000)

        source_mesh = o3d.io.read_triangle_mesh(meshpath)  # yellow polycam mesh "E:/polyPro/pair_data/pair_mesh_ffinal/121-2-1_mesh.obj"
        source = o3d.geometry.PointCloud()
        source.points = source_mesh.vertices
        print(meshpath)
        best_scale = find_best_scale(source, target, threshold, trans_init)
        print("Best scale:", best_scale)
        source.scale(best_scale, center=source.get_center())  # 1080 for pc to fit3d
        # source.translate((0, 800, 0))  # (0,880,0) for raw pc
        source = source.farthest_point_down_sample(10000)

        reg_p2p = o3d.pipelines.registration.registration_icp(
            source, target, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())

        evaluation = o3d.pipelines.registration.evaluate_registration(
            source, target, threshold, reg_p2p.transformation)
        print("Evaluation: ", evaluation)

        scaled_mesh = source_mesh.scale(best_scale, center=source.get_center())
        registered_mesh = scaled_mesh.transform(reg_p2p.transformation)
        o3d.io.write_triangle_mesh(targetpath, registered_mesh)
        # draw_registration_result(source, target, reg_p2p.transformation)












