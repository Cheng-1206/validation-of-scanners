import open3d as o3d
import numpy as np
import copy
import matplotlib.pyplot as plt
import copy

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


demo_icp_pcds = o3d.data.DemoICPPointClouds()
source_mesh = o3d.io.read_triangle_mesh("./pair_data/pair_polycam_final/1-1-1_polycam.obj")  # yellow
source_mesh.compute_vertex_normals()
source = o3d.geometry.PointCloud()
source.points = source_mesh.vertices
# source.scale(1100, center=source.get_center()) # 1080 for pc to fit3d
source.translate((0, 800, 0))  # (0,880,0) for raw pc
# source = source.farthest_point_down_sample(10000)

target_mesh = o3d.io.read_triangle_mesh("./pair_data/pair_fit3d_final/1-1-1.obj")
target_mesh.compute_vertex_normals()
target = o3d.geometry.PointCloud()
target.points = target_mesh.vertices
# target = target.farthest_point_down_sample(10000)

threshold = 500000
trans_init = np.asarray([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])

reg_p2p = o3d.pipelines.registration.registration_icp(
    source, target, threshold, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPoint())

evaluation = o3d.pipelines.registration.evaluate_registration(
    source, target, threshold, reg_p2p.transformation)
print("Evaluation: ", evaluation)

source_t = copy.deepcopy(source)
source_t.transform(reg_p2p.transformation)
source_mesh_t = copy.deepcopy(source_mesh)
source_mesh_t.translate((0, 800, 0))
source_mesh_t.transform(reg_p2p.transformation)


# =============================
# Compute distance (ref -> target)
# =============================
mesh_for_dist = o3d.geometry.PointCloud()
mesh_for_dist.points = source_mesh_t.vertices
mesh_for_dist_target = o3d.geometry.PointCloud()
mesh_for_dist_target.points = target_mesh.vertices
distances = np.asarray(mesh_for_dist.compute_point_cloud_distance(target))

print("Mean distance:", np.mean(distances))
print("Max distance:", np.max(distances))

# =============================
# Normalize distances for color
# =============================
d_min = np.min(distances)
d_max = np.percentile(distances, 99)  # avoid outliers 95

norm_dist = (distances - d_min) / (d_max - d_min)
norm_dist = np.clip(norm_dist, 0, 1)

# =============================
# Map to colormap
# =============================
colors = plt.get_cmap("jet")(norm_dist)[:, :3]

source_t.colors = o3d.utility.Vector3dVector(colors)

# =============================
# Visualization
# =============================
source_mesh_t.vertex_colors = o3d.utility.Vector3dVector(colors)
# source_mesh_t.paint_uniform_color([0, 0, 1]) # test with full blue color
o3d.visualization.draw_geometries([source_mesh_t])
