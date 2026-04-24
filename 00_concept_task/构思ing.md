036_cabinet
074_displaystand
076_breadbasket
117whiteboard_eraser and 119 mini-chalkboard

把柜子放在正前方

cabinet = create_sapien_urdf_obj(
    scene=task,
    pose=sapien.Pose([0.0, 0.1, 0.741], [0.7071068, 0, 0, 0.7071068]),
    modelname="036_cabinet",
    modelid=46653,
    fix_root_link=True,
)

**[0.0, 0.1, 0.741]**很合适，0.741是桌面高度，可调整

camera_head 是头上的相机

head_camera 是正对着机器人拍照的固定相机