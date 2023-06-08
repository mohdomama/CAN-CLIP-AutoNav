import colorsys
import open3d as o3d
import numpy as np


def visualize_pcd(pcd):
    assert pcd.shape[1] == 3, 'Invalid point cloud shape!'

    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
    o3d.visualization.draw_geometries([pcd_o3d])


def create_o3d_pcd(pcd, colors = None):
    assert pcd.shape[1] == 3, 'Invalid point cloud shape!'
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
    if colors is not None:
        colors = o3d.utility.Vector3dVector(colors/255.0)
        pcd_o3d.colors = colors

    return pcd_o3d

def visualize_multiple_pcd(pcd_list, color_list = None, bg = (0,0,0)):

    pcd_o3d_list = []
    num_pcd = len(pcd_list)
    for i, pcd in enumerate(pcd_list):
        assert pcd.shape[1] == 3, 'Invalid point cloud shape!'

        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
        if color_list is None:
            (r, g, b) = colorsys.hsv_to_rgb((1/(num_pcd)) * i, 1, 1)
            pcd_o3d.paint_uniform_color([r, g, b])
        else:
            if color_list[i] is None:
                (r, g, b) = colorsys.hsv_to_rgb((1/(num_pcd)) * i, 1, 1)
                pcd_o3d.paint_uniform_color([r, g, b])
            else: 
                color = o3d.utility.Vector3dVector(color_list[i]/255.0)
                pcd_o3d.colors = color
        pcd_o3d_list.append(pcd_o3d)

    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # for geometry in pcd_o3d_list:
    #     vis.add_geometry(geometry)
    # opt = vis.get_render_option()
    # opt.background_color = np.asarray([0, 0, 1])
    # vis.run()
    # vis.destroy_window()

    o3d.visualization.draw_geometries(pcd_o3d_list)

def pick_points(pcd, color=None):
    assert pcd.shape[1] == 3, 'Invalid point cloud shape!'
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
    if color is not None:
        color = o3d.utility.Vector3dVector(color/255.0)
        pcd_o3d.colors = color

    print("")
    print(
        "1) Pick points using [shift + left click]"
    )
    print("   Press [shift + right click] to undo point picking")
    print("2) After picking points, press 'Q' to close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd_o3d)
    vis.run()  # user picks points
    vis.destroy_window()
    print("")
    return vis.get_picked_points()

