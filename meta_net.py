import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
import torch.nn.init as init


def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


class MetaModule(nn.Module):
    def params(self):
        for name, param in self.named_params(self):
            yield param

    def named_leaves(self):
        return []

    def named_submodules(self):
        return []

    def named_params(self, curr_module=None, memo=None, prefix=''):
        if memo is None:
            memo = set()

        if hasattr(curr_module, 'named_leaves'):
            for name, p in curr_module.named_leaves():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p
        else:
            for name, p in curr_module._parameters.items():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p

        for mname, module in curr_module.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in self.named_params(module, memo, submodule_prefix):
                yield name, p

    def update_params(self, lr_inner, first_order=False, source_params=None, detach=False):
        if source_params is not None:
            for tgt, src in zip(self.named_params(self), source_params):
                name_t, param_t = tgt
                # name_s, param_s = src
                # grad = param_s.grad
                # name_s, param_s = src
                grad = src
                if first_order:
                    grad = to_var(grad.detach().data)
                tmp = param_t - lr_inner * grad
                self.set_param(self, name_t, tmp)
        else:

            for name, param in self.named_params(self):
                if not detach:
                    grad = param.grad
                    if first_order:
                        grad = to_var(grad.detach().data)
                    tmp = param - lr_inner * grad
                    self.set_param(self, name, tmp)
                else:
                    param = param.detach_()
                    self.set_param(self, name, param)

    def set_param(self, curr_mod, name, param):
        if '.' in name:
            n = name.split('.')
            module_name = n[0]
            rest = '.'.join(n[1:])
            for name, mod in curr_mod.named_children():
                if module_name == name:
                    self.set_param(mod, rest, param)
                    break
        else:
            setattr(curr_mod, name, param)

    def detach_params(self):
        for name, param in self.named_params(self):
            self.set_param(self, name, param.detach())

    def copy(self, other, same_var=False):
        for name, param in other.named_params():
            if not same_var:
                param = to_var(param.data.clone(), requires_grad=True)
            self.set_param(name, param)


class MetaLinear(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Linear(*args, **kwargs)

        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
        self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]


class wpi(MetaModule):
    def __init__(self, input, hidden1, output):
        # 64, 128, 10
        super(wpi, self).__init__()

        self.linear1 = MetaLinear(input, hidden1)
        self.tanh = nn.Tanh()
        self.linear2 = MetaLinear(hidden1, hidden1)
        self.linear_mean = MetaLinear(hidden1, output)
        self.linear_var = MetaLinear(hidden1, output)

        self.cls_emb = nn.Embedding(output, 10)

        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_normal_(self.linear1.weight)
        self.linear1.bias.data.zero_()
        torch.nn.init.xavier_normal_(self.linear2.weight)
        self.linear2.bias.data.zero_()
        torch.nn.init.xavier_normal_(self.linear_mean.weight)
        self.linear_mean.bias.data.zero_()

    def encode(self, x):
        h1 = self.tanh(self.linear1(x))
        h2 = self.tanh(self.linear2(h1))
        mean = self.linear_mean(h2)
        log_var = self.linear_var(h2)
        return mean, log_var

    def forward(self, feat, target, sample_num):
        target = self.cls_emb(target)

        x = torch.cat([feat, target], dim=-1)

        mean, log_var = self.encode(x)  # or 100
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std.unsqueeze(0).repeat(sample_num,1,1))

        return F.sigmoid(mean + std*eps)


class wpi_dec(MetaModule):
    def __init__(self, input, hidden1, output):
        # 64, 128, 10
        super(wpi_dec, self).__init__()

        self.linear1 = MetaLinear(input, hidden1)
        self.tanh = nn.Tanh()
        self.linear2 = MetaLinear(hidden1, hidden1)
        self.linear_mean = MetaLinear(hidden1, output)

        self.cls_emb = nn.Embedding(output, 10)  # or 100

        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_normal_(self.linear1.weight)
        self.linear1.bias.data.zero_()
        torch.nn.init.xavier_normal_(self.linear2.weight)
        self.linear2.bias.data.zero_()
        torch.nn.init.xavier_normal_(self.linear_mean.weight)
        self.linear_mean.bias.data.zero_()

    def encode(self, x):
        h1 = self.tanh(self.linear1(x))
        h2 = self.tanh(self.linear2(h1))
        mean = self.linear_mean(h2)
        return mean

    def forward(self, feat, target):
        target = self.cls_emb(target)

        x = torch.cat([feat, target], dim=-1)
        mean = self.encode(x) # [100, 10]
        return F.sigmoid(mean)








