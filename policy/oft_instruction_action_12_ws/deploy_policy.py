try:
    from starvla_astribot.deploy_policy import eval, get_model, reset_model
except ModuleNotFoundError:
    from policy.starvla_astribot.deploy_policy import eval, get_model, reset_model

