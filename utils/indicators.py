import torch
from torch.nn.functional import normalize
from torch import autograd
from pdb import set_trace as bp


def l2_reg_ortho(mdl):
    l2_reg = None
    # for W in mdl.parameters():
    for n, m in mdl.named_modules():
        if not hasattr(m, "sparse_weight"): continue
        # if W.ndimension() < 2:
        if m.sparse_weight.ndimension() < 2:
            continue
        else:
            cols = m.sparse_weight[0].numel()
            rows = m.sparse_weight.shape[0]
            w1 = (m.sparse_weight).view(-1, cols)
            wt = torch.transpose(w1, 0, 1)
            m  = torch.matmul(wt, w1)
            ident = torch.eye(cols, cols)
            ident = ident.cuda()

            w_tmp = (m - ident)
            height = w_tmp.size(0)
            u = normalize(w_tmp.new_empty(height).normal_(0, 1), dim=0, eps=1e-12)
            v = normalize(torch.matmul(w_tmp.t(), u), dim=0, eps=1e-12)
            u = normalize(torch.matmul(w_tmp, v), dim=0, eps=1e-12)
            sigma = torch.dot(u, torch.matmul(w_tmp, v))

            if l2_reg is None:
                l2_reg = (sigma)**2
            else:
                l2_reg = l2_reg + (sigma)**2
    return l2_reg


def get_ntk(network, Xtrain, Ytrain, Xtest=None, Ytest=None, criterion=torch.nn.CrossEntropyLoss(), recalbn=0, train_mode=False, num_classes=10):
    device = torch.cuda.current_device()
    ntks = []
    if train_mode:
        network.train()
    else:
        network.eval()
    ######
    grads_x = [] # size: #training samples. grads of all W from each validation samples
    grads_y = [] # size: #training samples. grads of all W from each validation samples
    targets_x_onehot_mean = []; targets_y_onehot_mean = []
    inputs = Xtrain.cuda(device=device, non_blocking=True)
    targets = Ytrain.cuda(device=device, non_blocking=True)
    targets_onehot = torch.nn.functional.one_hot(targets, num_classes=num_classes).float()
    targets_onehot_mean = targets_onehot - targets_onehot.mean(0)
    targets_x_onehot_mean.append(targets_onehot_mean)
    network.zero_grad()
    logit = network(inputs)
    if isinstance(logit, tuple):
        logit = logit[1]  # 201 networks: return features and logits
    for _idx in range(len(inputs)):
        # logit[_idx:_idx+1].backward(torch.ones_like(logit[_idx:_idx+1]), retain_graph=True)
        # grad = []
        # for name, W in network.named_parameters():
        #     if 'weight' in name and W.grad is not None:
        #         grad.append(W.grad.view(-1))
        # grads_x.append(torch.cat(grad, -1))

        # TODO only grad on weights, not masks
        grads = autograd.grad(logit[_idx:_idx+1].sum(), [ module.weight for name, module in network.named_modules() if hasattr(module, "sparse_weight") ], create_graph=True)
        grads_x.append(grads)

        network.zero_grad()
        torch.cuda.empty_cache()
    targets_x_onehot_mean = torch.cat(targets_x_onehot_mean, 0)
    # NTK cond
    grads_x = torch.stack(grads_x, 0)
    ntk = torch.einsum('nc,mc->nm', [grads_x, grads_x])
    eigenvalues, _ = torch.symeig(ntk)  # ascending
    # cond_x = eigenvalues[-1] / eigenvalues[0]
    cond_x = eigenvalues[-1] / eigenvalues[len(eigenvalues)//2]
    # TODO only grad on weights, not masks
    autograd.backward(cond_x, [ module.mask for name, module in network.named_modules() if hasattr(module, "sparse_weight") ], create_graph=True)
    # Val / Test set
    if Xtest is not None:
        inputs = Xtest.cuda(device=device, non_blocking=True)
        targets = Ytest.cuda(device=device, non_blocking=True)
        targets_onehot = torch.nn.functional.one_hot(targets, num_classes=num_classes).float()
        targets_onehot_mean = targets_onehot - targets_onehot.mean(0)
        targets_y_onehot_mean.append(targets_onehot_mean)
        network.zero_grad()
        logit = network(inputs)
        if isinstance(logit, tuple):
            logit = logit[1]  # 201 networks: return features and logits
        for _idx in range(len(inputs)):
            # TODO use autograd?
            logit[_idx:_idx+1].backward(torch.ones_like(logit[_idx:_idx+1]), retain_graph=True)
            grad = []
            for name, W in network.named_parameters():
                if 'weight' in name and W.grad is not None:
                    grad.append(W.grad.view(-1))
            grads_y.append(torch.cat(grad, -1))
            network.zero_grad()
            torch.cuda.empty_cache()
        grads_y = torch.stack(grads_y, 0)
        targets_y_onehot_mean = torch.cat(targets_y_onehot_mean, 0)
        try:
            # lastcellgrads_y = torch.stack(lastcellgrads_y, 0)
            _ntk_yx = torch.einsum('nc,mc->nm', [grads_y, grads_x])
            PY = torch.einsum('jk,kl,lm->jm', _ntk_yx, torch.inverse(ntk), targets_x_onehot_mean)
            prediction_mse = ((PY - targets_y_onehot_mean)**2).sum(1).mean(0)
        except RuntimeError:
            # RuntimeError: inverse_gpu: U(1,1) is zero, singular U.
            # prediction_mses.append(((targets_y_onehot_mean)**2).sum(1).mean(0).item())
            prediction_mses = -1 # bad gradients
    ######
    if Xtest is None:
        return cond_x
    else:
        return cond_x, prediction_mse


def ntk_differentiable(network, Xtrain, train_mode=True, need_graph=True):
    # XXX return one NTK per mini-batch in loader
    device = torch.cuda.current_device()
    ntks = []
    if train_mode:
        network.train()
    else:
        network.eval()
    ######
    grads = []
    inputs = Xtrain.cuda(device=device, non_blocking=True)
    for _idx in range(len(inputs)):
        # TODO only grad on weights, not masks
        _gradients = autograd.grad(outputs=network(inputs[_idx:_idx+1]).sum(), inputs=[ module.mask for name, module in network.named_modules()
                                    if hasattr(module, "sparse_weight") ], retain_graph=need_graph, create_graph=need_graph)
        grad = [] # grad of all weights for this sample
        for _grad in _gradients:
            if need_graph:
                grad.append(_grad.view(-1))
            else:
                grad.append(_grad.view(-1).detach())
        grads.append(torch.cat(grad, -1)) # grad of all weights for this sample
        if not need_graph:
            network.zero_grad()
            torch.cuda.empty_cache()
    grads = torch.stack(grads, 0) # #samples * #params
    ntk = torch.einsum('nc,mc->nm', [grads, grads])
    return ntk
