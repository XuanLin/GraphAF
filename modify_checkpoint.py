import torch
import collections
import sys
import os
import copy

checkpoint_in_path = sys.argv[1]
checkpoint_out_path = sys.argv[2]

ckpt = torch.load(checkpoint_in_path)
print(ckpt.keys())

model_state = a['model_state_dict']
model_state_key = list(model_state.keys())
print(model_state_key)
;¬
model_state_key_rename = []

for name in model_state_key:
    if 'module' in name:
        name_split = name.split('.')
        assert name_split[1] == 'module'
        name_split.pop(1)
        name_split.insert(0, 'module')
        new_name = '.'.join(name_split)
        model_state_key_rename.append(new_name)
    else:
        name_split = name.split('.')
        name_split.insert(0, 'module')
        new_name = '.'.join(name_split)
        model_state_key_rename.append(new_name)        

print(model_state_key_rename)
model_state_rename = collections.OrderedDict()
assert len(model_state_key) == len(model_state_key_rename)

for i in range(len(model_state_key)):
    model_state_rename[model_state_key_rename[i]] = model_state[model_state_key[i]]

ckpt['model_state_dict'] = model_state_rename

torch.save({**ckpt}, checkpoint_out_path)   

