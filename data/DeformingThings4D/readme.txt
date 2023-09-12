DeformingThings4D dataset
Current statistic: 1972 animations and 122365 frames


### Folder structure
|--humanoids (200 animations, 34228 frames)
    |--clarie_run  #a animation folder [objectID]_[ActionID])
        |--clarie_run.anime # animation file, storing per-frame shape and
        |--screenshots # screenshots of animation
        |--clarie_run.fbx # raw blender animation file, only available for humanoids
|--animals (1772 animations, 88137 frames)


### Animation file (.anime)
Binary ".anime" file format, storing an animation.
The first frame is the canonical frame for which we store its triangle mesh.
From the 2nd to the last frame, we store the 3D offsets of the mesh vertices.
#length         #type       #contents
|1              |int32      |nf: number of frames in the animation 
|1              |int32      |nv: number of vertices in the mesh (mesh topology fixed through frames)
|1              |int32      |nt: number of triangle face in the mesh
|nv*3           |float32    |vertice data of the 1st frame (3D positions in x-y-z-order)
|nt*3           |int32      |triangle face data of the 1st frame
|(nf-1)*nv*3    |float32    |3D offset data from the 2nd to the last frame

