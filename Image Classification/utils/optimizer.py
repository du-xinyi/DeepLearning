import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler


optimizer_dict = {
    'SGD': optim.SGD,
    'RMSprop': optim.RMSprop,
    'Adagrad': optim.Adagrad,
    'Adam': optim.Adam,
    'AdamW': optim.AdamW
}

def optimizer(optimizer_name, lr_name, step, config, model_parameters):
    if optimizer_name in optimizer_dict:
        optimizer_class = optimizer_dict[optimizer_name]

        optimizer = None
        scheduler = None

        # 固定值学习率
        if lr_name == 'Fixed':
            learning_rate = config.get('learning rate', {}).get('Fixed')
            optimizer = optimizer_class(model_parameters, lr=learning_rate)

        # 余弦退火学习率
        elif lr_name == 'Cosine':
            cosine_lr = config.get('learning rate', {}).get('Cosine')
            if cosine_lr is not None:
                initial_lr = cosine_lr.get('initial_lr')
                final_lr = cosine_lr.get('final_lr')
                optimizer = optimizer_class(model_parameters, lr=initial_lr)
                scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=step, eta_min=final_lr)

        else:
            raise ValueError("Invalid learning_rate name: {}".format(lr_name))

        return optimizer, scheduler
    else:
        raise ValueError("Invalid optimizer name: {}".format(optimizer_name))