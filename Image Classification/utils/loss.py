import torch.nn as nn

loss_dict = {
    'CrossEntropyLoss': nn.CrossEntropyLoss,
    'NLLLoss': nn.NLLLoss,
    'BCEWithLogitsLoss': nn.BCEWithLogitsLoss,
    'BCELoss': nn.BCELoss
}

def select_loss(loss_name):
    if loss_name in loss_dict:
        loss_function = loss_dict[loss_name]()
    else:
        raise ValueError("Invalid loss function name: {}".format(loss_name))

    return loss_function