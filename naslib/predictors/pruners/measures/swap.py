import torch
import torch.nn as nn

from . import measure
    
def register_hook(module):

    def hook_in_forward(module, input, output):
        if isinstance(input, tuple) and len(input[0].size()) == 4:
            module.postacts.append(output.detach())

    module.postacts = []
    module.register_forward_hook(hook=hook_in_forward)

@measure('swap')
def swap(net, inputs, targets, split_data=1, loss_fn=None):

    device = inputs.device

    for n, m in net.named_modules():
        if isinstance(m, nn.ReLU):
            register_hook(m)

    net.to(device)
    net.train()
    net.zero_grad()
    N = inputs.shape[0]

    with torch.no_grad():

        postact_batch = []
        for sp in range(split_data):
            postact_data = []

            st = sp * N // split_data
            en = (sp + 1) * N // split_data

            outputs = net.forward(inputs[st:en])
            for n, m in net.named_modules():
                if isinstance(m, nn.ReLU):
                    postact_data += m.postacts

            postacts = torch.cat([a.view(inputs.size(0), -1) for a in postact_data], 1)
            postact_batch.append(postacts)

        postact_data_ = torch.stack(postact_batch, dim=0)
        postact_signs = torch.sign(postact_data_)
        postact_signs = postact_signs.view(postact_signs.shape[0]*postact_signs.shape[1], postact_signs.shape[2])
        postact_signs = postact_signs.T

        swap_metric = torch.unique(postact_signs, dim=0).size(0)
    
    return(swap_metric)
        
