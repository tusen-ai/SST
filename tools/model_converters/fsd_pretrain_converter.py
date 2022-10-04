import torch
from collections import OrderedDict

config = 'pretrain_config_name'

ckpt = torch.load(f'./work_dirs/{config}/latest.pth')
model = ckpt['state_dict']
new_model = OrderedDict()
for name in model:
    new_model['segmentor.'+name] = model[name]

ckpt['state_dict'] = new_model
torch.save(ckpt, f'./work_dirs/{config}/segmentor_pretrain.pth')