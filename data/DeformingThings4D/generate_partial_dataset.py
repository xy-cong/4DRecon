import bpy
import bmesh
import os
import numpy as np
import mathutils
import cv2
import sys
import time
from mathutils import Matrix, Vector, Quaternion, Euler
from mathutils.bvhtree import BVHTree
import glob
import open3d as o3d
from plyfile import PlyData, PlyElement



def save_to_ply(pts, normal, faces, filename):
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("element vertex {}\n".format(len(pts)))
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property float nx\n")
        f.write("property float ny\n")
        f.write("property float nz\n")
        f.write("element face {}\n".format(len(faces)))
        f.write("property list uchar int vertex_index\n")
        f.write("end_header\n")
        for p, n in zip(pts, normal):
            f.write("{} {} {} {} {} {}\n".format(p[0], p[1], p[2], n[0], n[1], n[2]))
        for face in faces:
            f.write("{} ".format(len(face)) + " ".join(map(str, face)) + "\n")

def save_to_obj(pts, normal, faces, filename):
    with open(filename, 'w') as f:
        for p in pts:
            f.write("v {} {} {}\n".format(p[0], p[1], p[2]))
        for n in normal:
            f.write("vn {} {} {}\n".format(n[0], n[1], n[2]))
        for face in faces:
            # OBJ索引从1开始，所以需要为每个索引加1
            f.write("f {} {} {}\n".format(face[0]+1, face[1]+1, face[2]+1))


D=bpy.data
C=bpy.context
pi = 3.14

def opencv_to_blender(T):
    """T: ndarray 4x4
       usecase: cam.matrix_world =  world_to_blender( np.array(cam.matrix_world))
    """
    origin = np.array(((1, 0, 0, 0),
        (0, -1, 0, 0),
        (0, 0, -1, 0),
        (0, 0,  0, 1)))
    return np.matmul(T,origin) #T * origin

def blender_to_opencv(T):
    transform = np.array(((1, 0, 0, 0),
              (0, -1, 0, 0),
              (0, 0, -1, 0),
              (0, 0, 0, 1)))
    return np.matmul(T,transform)#T * transform


def set_camera( bpy_cam,  angle=pi / 3, W=600, H=500):
    """TODO: replace with setting by intrinsics """
    bpy_cam.angle = angle
    bpy_scene = bpy.context.scene
    bpy_scene.render.resolution_x = W
    bpy_scene.render.resolution_y = H

def look_at(obj_camera, point):
    # import ipdb; ipdb.set_trace()
    loc_camera = obj_camera.matrix_world.to_translation()
    direction = point - loc_camera
    # point the cameras '-Z' and use its 'Y' as up
    rot_quat = direction.to_track_quat('-Z', 'Y')
    # assume we're using euler rotation
    obj_camera.rotation_euler = rot_quat.to_euler()

def get_calibration_matrix_K_from_blender(camd):
    '''
    refer to: https://blender.stackexchange.com/questions/15102/what-is-blenders-camera-projection-matrix-model
    the code from the above link is wrong, it cause a slight error for fy in 'HORIZONTAL' mode or fx in "VERTICAL" mode.
    We did change to fix this.
    '''
    f_in_mm = camd.lens
    scene = bpy.context.scene
    resolution_x_in_px = scene.render.resolution_x
    resolution_y_in_px = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100
    sensor_width_in_mm = camd.sensor_width
    sensor_height_in_mm = camd.sensor_height
    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
    if (camd.sensor_fit == 'VERTICAL'):
        s_v = resolution_y_in_px * scale / sensor_height_in_mm
        s_u = s_v/pixel_aspect_ratio
    else: # 'HORIZONTAL' and 'AUTO'
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
        s_u = resolution_x_in_px * scale / sensor_width_in_mm
        s_v = s_u/pixel_aspect_ratio
    alpha_u = f_in_mm * s_u
    alpha_v = f_in_mm * s_v
    u_0 = resolution_x_in_px*scale / 2
    v_0 = resolution_y_in_px*scale / 2
    skew = 0 # only use rectangular pixels
    K = Matrix(
        ((alpha_u, skew,    u_0),
        (    0  ,  alpha_v, v_0),
        (    0  ,    0,      1 )))
    return K


def anime_read( filename):
    """
    filename: path of .anime file
    return:
        nf: number of frames in the animation
        nv: number of vertices in the mesh (mesh topology fixed through frames)
        nt: number of triangle face in the mesh
        vert_data: vertice data of the 1st frame (3D positions in x-y-z-order)
        face_data: riangle face data of the 1st frame
        offset_data: 3D offset data from the 2nd to the last frame
    """
    f = open(filename, 'rb')
    nf = np.fromfile(f, dtype=np.int32, count=1)[0]
    nv = np.fromfile(f, dtype=np.int32, count=1)[0]
    nt = np.fromfile(f, dtype=np.int32, count=1)[0]
    vert_data = np.fromfile(f, dtype=np.float32, count=nv * 3)
    face_data = np.fromfile(f, dtype=np.int32, count=nt * 3)
    offset_data = np.fromfile(f, dtype=np.float32, count=-1)
    '''check data consistency'''
    if len(offset_data) != (nf - 1) * nv * 3:
        raise ("data inconsistent error!", filename)
    vert_data = vert_data.reshape((-1, 3))
    face_data = face_data.reshape((-1, 3))
    offset_data = offset_data.reshape((nf - 1, nv, 3))
    return nf, nv, nt, vert_data, face_data, offset_data

class AnimeRenderer:

    def __init__(self, anime_file, dum_path):
        #####################################################################
        self.nf, _, _, self.vert_data, self.face_data, offset_data = \
            anime_read(anime_file)
        self.offset_data = np.concatenate ( [ np.zeros( (1, offset_data.shape[1], offset_data.shape[2]) ), offset_data], axis=0)
        '''make object mesh'''

        # import ipdb; ipdb.set_trace()
        vertices = self.vert_data.tolist()
        edges = []
        faces = self.face_data.tolist()
        mesh_data = bpy.data.meshes.new('mesh_data')
        mesh_data.from_pydata(vertices, edges, faces)
        mesh_data.update()
        the_mesh = bpy.data.objects.new('the_mesh', mesh_data)
        the_mesh.data.vertex_colors.new() # init color
        bpy.context.collection.objects.link(the_mesh)
        #####################################################################
        self.the_mesh = the_mesh
        #####################################################################
        self.dum_path = dum_path
        self.frame_base_dir = os.path.join( self.dum_path, "frame")
        if not os.path.exists(self.frame_base_dir):
            os.makedirs(self.frame_base_dir)

        #####################################################################
        self.original_ply = []
        self.original_pts = []
        self.original_normals = []
        self.Anime_to_Original_Ply()
        self.dum_path = dum_path
        #####################################################################
        self.global_ply = []
        
        self.global_ply_path = os.path.join( self.dum_path, "global_ply")
        self.global_obj_path = os.path.join( self.dum_path, " ")
        if not os.path.exists(self.global_ply_path):
            os.makedirs(self.global_ply_path)
        if not os.path.exists(self.global_obj_path):
            os.makedirs(self.global_obj_path)

        self.Original_Ply_to_Global_Norm_Ply()
        #####################################################################
        self.point_clouds = []
        self.normal_clouds = []
        # import ipdb; ipdb.set_trace()
        for idx in range(self.nf):
            print("loading: ", idx)
            self.point_clouds.append(self.global_ply[idx][:,:3].tolist())
            self.normal_clouds.append(self.global_ply[idx][:,3:].tolist())

    def Anime_to_Original_Ply(self):
        #####################################################################
        for frame in range(self.nf):
            points = self.vert_data + self.offset_data[frame]
            # 创建 TriangleMesh 对象
            mesh = o3d.geometry.TriangleMesh()
            # 设置顶点和面
            mesh.vertices = o3d.utility.Vector3dVector(points)
            mesh.triangles = o3d.utility.Vector3iVector(self.face_data)
            # 计算顶点法线
            
            mesh.compute_vertex_normals()
            points_normals = np.asarray(mesh.vertex_normals)
            pts_normals = np.concatenate([points, points_normals], axis=1)
            self.original_ply.append(pts_normals)
            self.original_pts.append(points)
            self.original_normals.append(points_normals)

    def Original_Ply_to_Global_Norm_Ply(self):
        # # way 1
        # global_pts = np.array(self.original_pts).reshape(-1, 3)
        # global_pts -= np.mean(global_pts, axis=0, keepdims=True)
        # global_pts_max = np.amax(global_pts)
        # global_pts_min = np.amin(global_pts)
        # global_pts = (global_pts - global_pts_min) / (global_pts_max - global_pts_min)
        # global_pts -= 0.5
        # global_pts *= 2.
        # global_pts = global_pts.reshape(self.nf, -1, 3)
        # global_pts_normals = np.concatenate([global_pts, np.array(self.original_normals)], axis=2)
        # self.global_ply = global_pts_normals.tolist()

        # # way 2
        global_pts = np.array(self.original_pts).reshape(-1, 3)
        global_vmin = global_pts.min(axis=0)
        global_vmax = global_pts.max(axis=0)
        global_center = (global_vmin + global_vmax) / 2
        global_v_scale = 2 / np.sqrt(np.sum((global_vmax - global_vmin) ** 2)) * 0.95
        for idx in range(len(self.original_pts)):
            # import ipdb; ipdb.set_trace()
            pts = np.array(self.original_pts[idx])
            pts = (pts - global_center[None, :]) * global_v_scale
            normals = np.array(self.original_normals[idx])
            self.global_ply.append(np.concatenate([pts, normals], axis=1))
            if f"{idx:04d}.obj" in os.listdir(self.global_obj_path) and f"{idx:04d}.ply" in os.listdir(self.global_ply_path):
                print(f"{idx:04d}.obj / .ply already exists")
                continue
            save_to_obj(pts, normals, self.face_data, os.path.join(self.global_obj_path, f"{idx:04d}.obj"))
            save_to_ply(pts, normals, self.face_data, os.path.join(self.global_ply_path, f"{idx:04d}.ply"))
            


    def vis_frame(self, fid):
        '''update geometry to a frame (for debug)'''
        src_offset = self.offset_data[fid]
        bm = bmesh.new()
        bm.from_mesh(self.the_mesh.data)
        bm.verts.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        for i in range(len(bm.verts)):
            bm.verts[i].co = Vector(self.vert_data[i] + src_offset[i])
        bm.to_mesh(self.the_mesh.data)
        bm.free()

    def mash_render(self, H, W, location, look_at_pt, idx):
        """simply setup the camera"""
        bpy_camera = D.objects['Camera']
        bpy_camera.location, look_at_point = Vector(location), Vector(look_at_pt) # need to compute this for optimal view point
        set_camera(bpy_camera.data, angle=pi/3, W=W, H=H)
        bpy.context.view_layer.update() #update camera params
        look_at(bpy_camera, look_at_point)
        bpy.context.view_layer.update() #update camera params

        num_frame = self.offset_data.shape[0]
        camera = D.objects["Camera"]
        
        frame_idx_dir = os.path.join( self.frame_base_dir, f"{idx:04d}")
        if not os.path.exists(frame_idx_dir):
            os.makedirs(frame_idx_dir)
        depth_dir = os.path.join( frame_idx_dir, "depth")
        if not os.path.exists(depth_dir):
            os.makedirs(depth_dir)
        partial_objs_dir = os.path.join( frame_idx_dir, "partial_objs")
        if not os.path.exists(partial_objs_dir):
            os.makedirs(partial_objs_dir)

        #####################################################################
        '''prepare rays, (put this inside the for loop if the camera also moves)'''
        K = get_calibration_matrix_K_from_blender(camera.data)
        print(K)
        
        fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
        width, height = C.scene.render.resolution_x, C.scene.render.resolution_y
        cam_blender = np.array(camera.matrix_world)
        print (camera.matrix_world, camera.location)
        cam_opencv = blender_to_opencv(cam_blender)
        u, v = np.meshgrid(range(width), range(height))
        u = u.reshape(-1)
        v = v.reshape(-1)
        pix_position = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], -1)
        cam_rotation = cam_opencv[:3, :3]
        pix_position = np.matmul(cam_rotation, pix_position.transpose()).transpose()
        ray_direction = pix_position / np.linalg.norm(pix_position, axis=1, keepdims=True)
        ray_origin = cam_opencv[:3, 3:].transpose()

        ####################################################################
        '''visulize ray geometry(for debug)'''
        vis_ray = False
        if vis_ray:
            ray_end = ray_origin + ray_direction
            ray_vert = np.concatenate([ray_origin, ray_end], axis=0)
            ray_edge = [(0, r_end) for r_end in range(1, len(ray_end) + 1)]
            ray_mesh_data = bpy.data.meshes.new("the_raw")
            ray_mesh_data.from_pydata(ray_vert, ray_edge, [])
            ray_mesh_data.update()
            the_ray = bpy.data.objects.new('the_ray', ray_mesh_data)
            # the_mesh.data.vertex_colors.new()  # init color
            bpy.context.collection.objects.link(the_ray)
            
            # # 在lookatpoint创建一个新的球体
            # bpy.ops.mesh.primitive_uv_sphere_add(location=look_at_point)

            # # 设置球体的大小
            # bpy.context.object.scale = (0.1, 0.1, 0.1)

            bpy.ops.wm.save_as_mainfile(filepath="debug/vis.blend")
            exit()


        ####################################################################
        """dump intrinsics & extrinsics"""
        intrin_path = os.path.join(frame_idx_dir, "cam_intr.txt")
        extrin_path = os.path.join(frame_idx_dir, "cam_extr.txt")
        np.savetxt ( intrin_path, np.array(K))
        np.savetxt (extrin_path, cam_opencv)


        #####################################################################
        # time_spent = 0
        # ray_cnt = 0
        for src_frame_id in range(num_frame):
            print(src_frame_id)

            #####################################################################
            '''update geometry'''
            bm = bmesh.new()
            bm.from_mesh(self.the_mesh.data)
            bm.verts.ensure_lookup_table()
            bm.faces.ensure_lookup_table()
            for i in range(len(bm.verts)):
                bm.verts[i].co = Vector(self.point_clouds[src_frame_id][i])
            bm.to_mesh(self.the_mesh.data)
            self.the_mesh.data.update()


            #####################################################################
            """explicitly cast rays to get point cloud and scene flow"""
            # TODO: speedup the code
            # Currently, the code is a bit slower than directly rendering by composition layer of pass_z and pass_uv, (see: https://github.com/lvzhaoyang/RefRESH/tree/master/blender)
            # but since ray_cast return the faceID, this code is more flexible to use, e.g. generating model2frame dense correspondences)
            raycast_mesh = self.the_mesh
            ray_begin_local = raycast_mesh.matrix_world.inverted() @ Vector(ray_origin[0])
            depsgraph=bpy.context.evaluated_depsgraph_get()
            bvhtree = BVHTree.FromObject(raycast_mesh, depsgraph)
            pcl = np.zeros_like(ray_direction)
            subset_faces = []
            for i in range(ray_direction.shape[0]):
                # start = time.time()
                # hit, position, norm, faceID = raycast_mesh.ray_cast(ray_begin_local, Vector(ray_direction[i]), distance=60)
                position, norm, faceID, _ =bvhtree.ray_cast(ray_begin_local, Vector(ray_direction[i]), 50)
                # end = time.time()
                if position: # hit a triangle
                    # import ipdb; ipdb.set_trace()
                    pcl[i]= Matrix(cam_opencv).inverted() @ raycast_mesh.matrix_world @ position
                    face = bm.faces[faceID]
                    vert_index = [ v.index for v in face.verts]
                    subset_faces.append(vert_index)
                    vert_vector = [ v.co for v in face.verts ]

            # 提取子集中的顶点
            # import ipdb; ipdb.set_trace()
            subset_vertex_indices = np.unique(subset_faces)
            subset_vertices = np.asarray(self.point_clouds[src_frame_id])[subset_vertex_indices]
            subset_normals = np.asarray(self.normal_clouds[src_frame_id])[subset_vertex_indices]
            # # 为子集中的顶点重新编制索引
            index_map = {v: i for i, v in enumerate(subset_vertex_indices)}
            subset_faces = np.vectorize(index_map.get)(subset_faces)
            # 然后，我们处理面数据
            
            # import ipdb; ipdb.set_trace()
            save_to_obj(subset_vertices, subset_normals, subset_faces, os.path.join(partial_objs_dir, f"{src_frame_id:04d}.obj"))
            
            # faces = subset_faces.tolist()
            # for i in range(len(faces)):
            #     faces[i] = (faces[i],)
            # vertices_normals = [tuple(pt) + tuple(normal) for pt, normal in zip(subset_vertices, subset_normals)]
            # vertex_normal_element = PlyElement.describe(np.array(vertices_normals, dtype=[
            #     ('x', 'f4'),
            #     ('y', 'f4'),
            #     ('z', 'f4'),
            #     ('nx', 'f4'),
            #     ('ny', 'f4'),
            #     ('nz', 'f4')
            # ]), 'vertex')
            # ply_face = np.array(faces, dtype=[('vertex_indices', 'i4', (3,))])
            # face_element = PlyElement.describe(ply_face, 'face')

            # ply_data = PlyData([vertex_normal_element, face_element], text=True)
            # filename = f'{src_frame_id:04d}.ply'
            # ply_data.write(os.path.join(partial_plys_dir, filename))

            bm.free()

            #####################################################################
            """dump images"""
            depth = pcl[:,2].reshape((height, width))
            depth = (depth*1000).astype(np.uint16) #  resolution 1mm
            depth_path = os.path.join( depth_dir, "%04d"%src_frame_id + ".png")
            cv2.imwrite(depth_path , depth)



def circle_points(radius, center, num_points, plane='xz'):
    """
    Generate points on a circle.
    """
    angles = np.linspace(-np.pi / 2, 3*np.pi/2, num_points)
    if plane == 'xz':
        points = np.vstack([radius * np.cos(angles) + center[0],
                            np.full(num_points, center[1]),
                            radius * np.sin(angles) + center[2]]).T
    elif plane == 'xy':
        points = np.vstack([radius * np.cos(angles) + center[0],
                            radius * np.sin(angles) + center[1],
                            np.full(num_points, center[2])]).T
    elif plane == 'yz':
        points = np.vstack([np.full(num_points, center[0]),
                            radius * np.cos(angles) + center[1],
                            radius * np.sin(angles) + center[2]]).T
    return points

def sphere_points(radius, start, end, num_points):
    """
    Generate points on a sphere.
    """
    # Convert start and end to spherical coordinates
    start_spherical = cartesian_to_spherical(start)
    end_spherical = cartesian_to_spherical(end)

    # Interpolate in spherical coordinates
    theta = np.linspace(start_spherical[1], end_spherical[1], num_points)
    phi = np.linspace(start_spherical[2], end_spherical[2], num_points)

    # Convert back to cartesian coordinates
    points = np.zeros((num_points, 3))
    points[:, 0] = radius * np.sin(theta) * np.cos(phi)
    points[:, 1] = radius * np.sin(theta) * np.sin(phi)
    points[:, 2] = radius * np.cos(theta)

    return points

def cartesian_to_spherical(cartesian):
    """
    Convert cartesian coordinates to spherical coordinates.
    """
    x, y, z = cartesian
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    return r, theta, phi

def generate_trajectory():
    """
    Generate the entire trajectory.
    """
    num_points_xy = 120  
    num_points_sphere = 60
    trajectory = []

    # Circle on the plane z = 0
    trajectory.append(circle_points(2*np.sqrt(2), (0, 0, 0), num_points_xy, 'xy'))

    trajectory.append(sphere_points(2*np.sqrt(2), (0, -2*np.sqrt(2), 0), (0, -2, 2), num_points_sphere))

    # Circle on the plane z = 2
    trajectory.append(circle_points(2, (0, 0, 2), num_points_xy, 'xy'))

    trajectory.append(sphere_points(2*np.sqrt(2), (0, -2, 2), (0, -2*np.sqrt(2), 0), num_points_sphere))

    trajectory.append(sphere_points(2*np.sqrt(2), (0, -2*np.sqrt(2), 0), (0, -2, -2), num_points_sphere))

    # Circle on the plane z = -2
    trajectory.append(circle_points(2, (0, 0, -2), num_points_xy, 'xy'))

    # Move along the sphere to (2*sqrt(2), 0, 0)
    trajectory.append(sphere_points(2*np.sqrt(2), (0, -2, -2), (0, -2*np.sqrt(2), 0), num_points_sphere))





    # # Circle on the plane z = -2
    # trajectory.append(circle_points(2, (0, 0, -2), num_points_xy, 'xy'))

    # # Move along the sphere to (2*sqrt(2), 0, 0)
    # trajectory.append(sphere_points(2*np.sqrt(2), (0, -2, -2), (0, -2*np.sqrt(2), 0), num_points_sphere))

    # # Circle on the plane z = 0
    # trajectory.append(circle_points(2*np.sqrt(2), (0, 0, 0), num_points_xy, 'xy'))

    # # Move along the sphere to (2, 0, 2)
    # trajectory.append(sphere_points(2*np.sqrt(2), (0, -2*np.sqrt(2), 0), (0, -2, 2), num_points_sphere))

    # # Circle on the plane z = 2
    # trajectory.append(circle_points(2, (0, 0, 2), num_points_xy, 'xy'))


    # Concatenate all parts of the trajectory
    trajectory = np.concatenate(trajectory)

    return trajectory

def plot_trajectory(trajectory):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    center = [0,0,0]
    radius = 1
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))

    ax.plot_surface(x, y, z, color='b')

    # plt.show()
    plt.savefig("trajectory.png")

if __name__ == '__main__':


    argv = sys.argv
    argv = argv[argv.index("--") + 1:]  # get all args after "--"
    print(argv)
    anime_file = argv[0]
    dump_path = argv[1]

    location_trajectory = generate_trajectory()
    plot_trajectory(location_trajectory)

    #####################################################################
    """delete the default cube (which held the material)"""
    bpy.ops.object.select_all(action='DESELECT')
    bpy.data.objects['Cube'].select_set(state=True)
    bpy.ops.object.delete(use_global=False)
    renderer = AnimeRenderer(anime_file, dump_path)
    for idx in range(location_trajectory.shape[0]):
        #####################################################################

        # blender 的 xyz坐标关系
        # 右手系
        # x 向右为正
        # y 向前为正（指向屏幕里）
        # z 向上为正
        # bpy_camera.location, look_at_point = Vector ((2,-2,2)), Vector((0,0,1)) # need to compute this for optimal view point
        H = 500
        W = 600
        location = tuple(location_trajectory[idx])
        look_at_pt = (0,0,0) # camera的朝向，即camera永远会朝向look_at_pt
        renderer.mash_render(H, W, location, look_at_pt, idx)


