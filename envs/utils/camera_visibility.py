import numpy as np
import transforms3d as t3d


def _as_vec(vec, expected_dim: int, name: str):
    arr = np.array(vec, dtype=np.float64).reshape(-1)
    if arr.shape[0] != int(expected_dim):
        raise ValueError(f"{name} must have shape ({expected_dim},), got {arr.shape}")
    return arr


def get_camera_fov_xy(image_w, image_h, fovy_rad):
    width = int(image_w)
    height = int(image_h)
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid image size: ({image_w}, {image_h})")

    fovy = float(fovy_rad)
    if not np.isfinite(fovy) or fovy <= 0.0 or fovy >= np.pi:
        raise ValueError(f"invalid vertical fov: {fovy_rad}")

    aspect = float(width) / float(height)
    fovx = 2.0 * np.arctan(np.tan(fovy * 0.5) * aspect)
    return float(fovx), float(fovy)


def world_point_to_camera_local(world_point, camera_pose):
    """
    Transform a world-space point into the local frame of the camera.

    In this repository the camera local axes are:
      - +X: forward
      - +Y: left
      - +Z: up
    """
    point_world = _as_vec(world_point, 3, "world_point")
    cam_pos = _as_vec(camera_pose.p, 3, "camera_pose.p")
    cam_quat = _as_vec(camera_pose.q, 4, "camera_pose.q")
    cam_rot = t3d.quaternions.quat2mat(cam_quat)
    return cam_rot.T @ (point_world - cam_pos)


def is_world_point_in_camera_fov(
    world_point,
    camera_pose,
    image_w,
    image_h,
    fovy_rad,
    near_eps=1e-4,
    far=None,
    horizontal_margin_rad=0.0,
    vertical_margin_rad=0.0,
    ret_debug=False,
):
    """
    Check whether a world-space point lies inside a camera frustum.
    """
    cam_point = world_point_to_camera_local(world_point, camera_pose)
    cam_x, cam_y, cam_z = cam_point.tolist()

    fovx_rad, fovy_rad = get_camera_fov_xy(image_w=image_w, image_h=image_h, fovy_rad=fovy_rad)
    yaw_limit = max(0.0, 0.5 * fovx_rad - max(float(horizontal_margin_rad), 0.0))
    pitch_limit = max(0.0, 0.5 * float(fovy_rad) - max(float(vertical_margin_rad), 0.0))

    yaw_err_rad = float(np.arctan2(cam_y, cam_x))
    pitch_err_rad = float(np.arctan2(cam_z, cam_x))
    inside_depth = bool(cam_x > max(float(near_eps), 0.0))
    if far is not None:
        inside_depth = bool(inside_depth and cam_x < float(far))
    inside_horizontal = bool(abs(yaw_err_rad) <= yaw_limit + 1e-12)
    inside_vertical = bool(abs(pitch_err_rad) <= pitch_limit + 1e-12)
    inside = bool(inside_depth and inside_horizontal and inside_vertical)

    if not ret_debug:
        return inside

    debug = {
        "camera_point": cam_point,
        "yaw_err_rad": yaw_err_rad,
        "pitch_err_rad": pitch_err_rad,
        "yaw_limit_rad": yaw_limit,
        "pitch_limit_rad": pitch_limit,
        "inside_depth": inside_depth,
        "inside_horizontal": inside_horizontal,
        "inside_vertical": inside_vertical,
    }
    return inside, debug


def _get_object_center_world(obj):
    if hasattr(obj, "get_pose"):
        pose = obj.get_pose()
    elif hasattr(obj, "pose"):
        pose = obj.pose
    else:
        raise TypeError("object must provide get_pose() or pose")

    if not hasattr(pose, "p"):
        raise TypeError("object pose must provide .p")
    return _as_vec(pose.p, 3, "object_pose.p")


def is_object_in_camera_fov(
    obj,
    camera_pose,
    image_w,
    image_h,
    fovy_rad,
    mode="center",
    near_eps=1e-4,
    far=None,
    horizontal_margin_rad=0.0,
    vertical_margin_rad=0.0,
    ret_debug=False,
):
    """
    Check whether an object falls inside the camera frustum.

    The first version only uses the object center as the representative point.
    """
    mode = str(mode).lower()
    if mode != "center":
        raise NotImplementedError(f"unsupported visibility mode: {mode}")

    world_point = _get_object_center_world(obj)
    visible = is_world_point_in_camera_fov(
        world_point=world_point,
        camera_pose=camera_pose,
        image_w=image_w,
        image_h=image_h,
        fovy_rad=fovy_rad,
        near_eps=near_eps,
        far=far,
        horizontal_margin_rad=horizontal_margin_rad,
        vertical_margin_rad=vertical_margin_rad,
        ret_debug=ret_debug,
    )
    if not ret_debug:
        return visible

    inside, debug = visible
    debug["world_point"] = world_point
    debug["mode"] = mode
    return inside, debug
