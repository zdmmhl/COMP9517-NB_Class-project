import torch.optim as optim

from models.factory import classifier_parameters


def build_optimizer(model, model_name, lr, weight_decay, backbone_lr_multiplier):
    head_params = classifier_parameters(model, model_name)
    head_ids = {id(parameter) for parameter in head_params}
    backbone_params = [
        parameter for parameter in model.parameters() if id(parameter) not in head_ids
    ]
    if model_name.endswith("pretrained") and backbone_lr_multiplier < 1.0:
        parameter_groups = [
            {"params": backbone_params, "lr": lr * backbone_lr_multiplier},
            {"params": head_params, "lr": lr},
        ]
    else:
        parameter_groups = [{"params": model.parameters(), "lr": lr}]
    return optim.AdamW(parameter_groups, weight_decay=weight_decay)
