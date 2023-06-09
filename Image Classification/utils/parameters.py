import os
import yaml
import csv


# 保存参数
def opt_yaml(opt, type, num_classes, steps, config, start_time, end_time, save_dir):
    # 将参数转换为字典
    opt_dict = vars(opt)

    selected_params = {}

    # 添加参数
    selected_params.update({'type': type,
                            'num_classes': num_classes,
                            'steps': steps,
                            'batch_size': opt_dict['batch_size'],
                            'optimizer': opt_dict['optimizer'],
                            'num_workers': opt_dict['num_workers'],
                            'model': opt_dict['model'],
                            'lr': {
                                opt_dict['lr']: config['learning rate'][opt_dict['lr']]
                            }})

    # 优化器选择SGDM额外写入动量参数
    if opt_dict['optimizer'] == 'SGDM':
        selected_params.update({'momentum': config['momentum']})

    selected_params.update({'device': opt_dict['device'],
                            'start_time': start_time,
                            'end_time': end_time})

    with open(os.path.join(save_dir, 'cfg.yaml'), 'w') as file:
        yaml.dump(selected_params, file, sort_keys=False)


# 保存数据
def model_parameters(Train_Loss, Train_Accuracy, Val_Loss, Val_Accuracy, save_dir):
    with open(os.path.join(save_dir, 'result.csv'), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['epoch', 'train loss', 'train accuracy', 'val loss', 'val accuracy'])

        for epoch, (train_loss, train_accuracy, val_loss, val_accuracy) in enumerate(zip(Train_Loss, Train_Accuracy, Val_Loss, Val_Accuracy)):
            writer.writerow([epoch, train_loss, train_accuracy, val_loss, val_accuracy])