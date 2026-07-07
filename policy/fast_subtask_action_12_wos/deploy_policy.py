try:
    from fast_subtask_action_6_wos.deploy_policy import (
        StarVLAFastClient,
        _parse_image_size,
        eval,
        reset_model,
    )
except ModuleNotFoundError:
    from policy.fast_subtask_action_6_wos.deploy_policy import (
        StarVLAFastClient,
        _parse_image_size,
        eval,
        reset_model,
    )


def get_model(usr_args):
    return StarVLAFastClient(
        host=usr_args.get("host", "127.0.0.1"),
        port=int(usr_args.get("port", 7901)),
        unnorm_key=usr_args.get("unnorm_key", None),
        image_size=_parse_image_size(usr_args.get("image_size", [224, 224])),
        action_steps=int(usr_args.get("action_steps", 16)),
        action_type=usr_args.get("action_type", "qpos"),
        history_frames=int(usr_args.get("history_frames", 12)),
        history_stride=usr_args.get("history_stride", None),
        request_log_path=usr_args.get("request_log_path", None),
        request_image_dir=usr_args.get("request_image_dir", None),
    )
