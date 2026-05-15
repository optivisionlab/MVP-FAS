import torch


def get_loss_fucntion(cfg, loss_name, device):
    if loss_name == 'CrossEntropy':
        if cfg.TRAIN.WEIGHTS is not None:
            train_loss = torch.nn.CrossEntropyLoss(torch.tensor(cfg.TRAIN.WEIGHTS)).cuda(device)
            val_loss = torch.nn.CrossEntropyLoss(torch.tensor(cfg.TRAIN.WEIGHTS)).cuda(device)
        else:
            train_loss = torch.nn.CrossEntropyLoss().cuda(device)
            val_loss = torch.nn.CrossEntropyLoss().cuda(device)

    else:
        raise Exception("Undefined loss function")
    return train_loss, val_loss
