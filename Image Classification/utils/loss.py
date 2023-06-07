import torch.nn as nn

loss_dict = {
    'CrossEntropyLoss': nn.CrossEntropyLoss,
    'BCELoss': nn.BCELoss,
    'BCEWithLogitsLoss': nn.BCEWithLogitsLoss,
    'NLLLoss': nn.NLLLoss
}

def select_loss(loss_name):
    if loss_name in loss_dict:
        loss_function = loss_dict[loss_name]()
    else:
        raise ValueError("Invalid loss function name: {}".format(loss_name))

    return loss_function