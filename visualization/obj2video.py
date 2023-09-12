import bpy
import os
from mathutils import Matrix, Vector, Quaternion, Euler
import math
# 定义 .obj 文件的目录
base_dir = "/home/xiaoyan/3D/4DRecon/4DRep_DeepSDF3D/work_dir/0811_partial_SDF_arap_128/run/results/train/interp_sdf"
obj_dir_epoch = '5999'
obj_dir = os.path.join(base_dir, obj_dir_epoch)
pi = 3.14

def set_camera( bpy_cam,  angle= pi / 3, W=600, H=500):
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

# 清除所有现有的网格数据
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.object.select_by_type(type='MESH')
bpy.ops.object.delete()
# 设定摄像机的位置和角度
camera = bpy.data.objects["Camera"]

camera.location, look_at_point =Vector((2,-2,0)), Vector((0,0,0))
set_camera(camera.data, angle=pi/4)
bpy.context.view_layer.update() #update camera params
look_at(camera, look_at_point)
bpy.context.view_layer.update() #update camera params
# 旋转摄像机一个角度，例如绕 Y 轴旋转 45 度
camera.rotation_euler[1] += math.radians(-90)  # 这里是绕 Y 轴旋转，你可以根据需要修改
camera.rotation_euler[0] += math.radians(-30)  # 这里是绕 X 轴旋转，你可以根据需要修改
camera.rotation_euler[2] += math.radians(30)  # 这里是绕 X 轴旋转，你可以根据需要修改

# 导入 .obj 文件并设置每个文件的可见性
NUM_OBJ = 25

# 设置背景色
world = bpy.context.scene.world
if not world:
    # 如果场景中没有世界设置，创建一个
    world = bpy.data.worlds.new("World")
    bpy.context.scene.world = world

# 设置背景色为红色（你可以根据需要修改这些值）
world.use_nodes = True
bg_node = world.node_tree.nodes["Background"]
bg_node.inputs[0].default_value = (1, 1, 1, 1)  # RGBA for red


# 创建黄色材质
yellow_material = bpy.data.materials.new(name="YellowMaterial")
yellow_material.use_nodes = True
bsdf = yellow_material.node_tree.nodes["Principled BSDF"]
bsdf.inputs["Base Color"].default_value = (0, 0, 1, 1)  # RGBA for yellow


# # 创建头顶光源
# light_data = bpy.data.lights.new(name="TopLight", type='POINT')
# light_data.energy = 1000  # 设置光源强度，可以根据需要调整
# light_object = bpy.data.objects.new(name="TopLight_Object", object_data=light_data)
# bpy.context.collection.objects.link(light_object)  # 将光源添加到当前集合
# light_object.location = (10, -10, 10)  # 设置光源的位置，这里是场景的上方


for i in range(NUM_OBJ):
    obj_path = os.path.join(obj_dir, f"{i:02d}.obj")
    
    # 导入 .obj 文件
    bpy.ops.import_scene.obj(filepath=obj_path)
    
    # 获取导入的对象
    obj = bpy.context.selected_objects[0]

    # 为物体分配黄色材质
    if obj.data.materials:
        # 如果物体已有材质，替换为黄色材质
        obj.data.materials[0] = yellow_material
    else:
        # 如果物体没有材质，添加黄色材质
        obj.data.materials.append(yellow_material)
    # 设置对象在特定帧上的可见性，使其停留 5 帧
    start_frame = i * 24
    end_frame = start_frame + 24
    obj.hide_render = True
    obj.keyframe_insert(data_path="hide_render", frame=start_frame - 1)
    obj.hide_render = False
    obj.keyframe_insert(data_path="hide_render", frame=start_frame)
    obj.hide_render = True
    obj.keyframe_insert(data_path="hide_render", frame=end_frame)

# 设置渲染路径和参数
bpy.context.scene.render.filepath = os.path.join(base_dir, f"{obj_dir_epoch}.mp4")
bpy.context.scene.render.image_settings.file_format = 'FFMPEG'
bpy.context.scene.render.ffmpeg.format = 'MPEG4'
bpy.context.scene.render.ffmpeg.codec = 'H264'
bpy.context.scene.render.fps = 24
bpy.context.scene.render.fps_base = 1.0
bpy.context.scene.frame_start = 0
bpy.context.scene.frame_end = NUM_OBJ*24

# 开始渲染
bpy.ops.render.render(animation=True)
