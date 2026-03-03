import torch

from models.MVP_FAS import mspt
def get_network(cfg, args, net_name='MVP_FAS', device='cpu', backbone='ViT-B/16'):
    if net_name == 'MVP_FAS':
        net = mspt(cfg=cfg, args=args, device=device, backbone=backbone)
    # net = torch.nn.DataParallel(net).cuda()
    return net

def set_pretrained_setting(net, optimizer, weight_path):
    checkpoint_dict = torch.load(weight_path)
    checkpoint = checkpoint_dict['state_dict']
    optim_checkpoint = checkpoint_dict['optimizer']
    last_epoch = checkpoint_dict['epoch'] - 1
    pretrained_dict = {k: v for k, v in checkpoint.items() if k in net.module.state_dict().keys()}
    net.module.load_state_dict(pretrained_dict)
    optimizer.load_state_dict(optim_checkpoint)
    return net, optimizer, last_epoch


def load_checkpoint(net, weight_path):
    checkpoint_dict = torch.load(weight_path, weights_only=False)
    net.load_state_dict(checkpoint_dict['state_dict'])
    net.eval()
    return net