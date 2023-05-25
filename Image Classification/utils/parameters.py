import os
import yaml
import csv


def opt_yaml(opt, type, classes, steps, config, start_time, end_time, save_dir):
    # 将参数转换为字典
    opt_dict = vars(opt)

    selected_params = {}

    # 添加参数
    selected_params.update({'type': type,
                            'classes': classes,
                            'steps': steps,
                            'batch_size': opt_dict['batch_size'],
                            'optimizer': opt_dict['optimizer'],
                            'num_workers': opt_dict['num_workers'],
                            'model': opt_dict['model'],
                            'lr': {
                                opt_dict['lr']: config['learning rate'][opt_dict['lr']]
                            },
                            'device': opt_dict['device'],
                            'start_time': start_time,
                            'end_time': end_time})

    with open(os.path.join(save_dir, 'cfg.yaml'), 'w') as file:
        yaml.dump(selected_params, file, sort_keys=False)


def model_parameters(Loss, Accuracy, save_dir):
    with open(os.path.join(save_dir, 'result.csv'), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['epoch', 'accuracy', 'loss'])

        for epoch, (loss, accuracy) in enumerate(zip(Loss, Accuracy)):
            writer.writerow([epoch, loss, accuracy]) # 写入结果