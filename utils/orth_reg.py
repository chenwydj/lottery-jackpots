import torch
from torch.nn.functional import normalize
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
