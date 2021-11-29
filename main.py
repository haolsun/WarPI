from meta_net import wpi, wpi_dec
from resnet32 import ResNet32
from wideresnet import WideResNet
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from dataloader import CIFAR10, CIFAR100
import argparse
import os, warnings
import math
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset (cifar10 [default] or cifar100)')
parser.add_argument('--corruption_prob', type=float, default=0.4, help='label noise')
parser.add_argument('--corruption_type', '-ctype', type=str, default='flip', help='Type of corruption ("unif" or "flip" or "flip2").')
parser.add_argument('--num_meta', type=int, default=1000)
parser.add_argument('--epochs', default=120, type=int, help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--batch_size', default=100, type=int, help='mini-batch size (default: 100)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float, help='weight decay (default: 5e-0.4)')
parser.add_argument('--WReNset', default=False, type=bool, help='using WideResNet-28-10')
parser.add_argument('--no-augment', dest='augment', action='store_false', help='whether to use standard augmentation (default: True)')
parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--prefetch', type=int, default=4, help='Pre-fetching threads.')
parser.add_argument('--gpuid', type=str, default='0')
parser.add_argument('--sample_number', type=int, default=3)
parser.set_defaults(augment=True)

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid


def build_dataset(root, args):
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    if args.augment:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),  (4, 4, 4, 4), mode='reflect').squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    if args.dataset == 'cifar10':

        train_data_meta = CIFAR10(
                root='../../data', train=True, meta=True, num_meta=args.num_meta, corruption_prob=args.corruption_prob,
                corruption_type=args.corruption_type, transform=train_transform, download=True, strong_t=None)

        train_data = CIFAR10(
                root='../../data', train=True, meta=False, num_meta=args.num_meta, corruption_prob=args.corruption_prob,
                corruption_type=args.corruption_type, transform=train_transform, download=True, seed=args.seed)

        test_data = CIFAR10(root='../../data', train=False, transform=test_transform, download=True)


    elif args.dataset == 'cifar100':
        train_data_meta = CIFAR100(
            root='../../data', train=True, meta=True, num_meta=args.num_meta, corruption_prob=args.corruption_prob,
            corruption_type=args.corruption_type, transform=train_transform, download=True, strong_t=None)
        train_data = CIFAR100(
            root='../../data', train=True, meta=False, num_meta=args.num_meta, corruption_prob=args.corruption_prob,
            corruption_type=args.corruption_type, transform=train_transform, download=True, seed=args.seed)
        test_data = CIFAR100(root='../../data', train=False, transform=test_transform, download=True)

    train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size, shuffle=True,
            num_workers=args.prefetch, pin_memory=True)
    train_meta_loader = torch.utils.data.DataLoader(
            train_data_meta, batch_size=args.batch_size, shuffle=True,
            num_workers=args.prefetch, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                                  num_workers=args.prefetch, pin_memory=True)

    return train_loader, train_meta_loader, test_loader
def build_model(args):

    if args.WReNset:
        model = WideResNet(depth=28, num_classes = args.dataset == 'cifar10' and 10 or 100, widen_factor=10)
    else:
        model = ResNet32(args.dataset == 'cifar10' and 10 or 100)

    if torch.cuda.is_available():
        model.cuda()
        torch.backends.cudnn.benchmark = True

    return model
def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def loss_function(logits, onehot_label):
    log_prob = torch.nn.functional.log_softmax(logits, dim=1)
    loss = - torch.sum(log_prob * onehot_label) / logits.size(0)
    return loss



def train(train_loader, train_meta_loader, model, vnet, optimizer_model, optimizer_vnet, epoch):
    import tqdm
    print('Epoch: %d, lr: %.5f' % (epoch, optimizer_model.param_groups[0]['lr']))

    train_loss = 0
    meta_loss = 0
    acc_meta = 0.0
    acc_train = 0.0

    num = 0
    train_meta_loader_iter = iter(train_meta_loader)
    for batch_idx, (inputs, targets, path) in enumerate(tqdm.tqdm(train_loader, ncols=0)):
        num = batch_idx
        model.train()

        meta_model = build_model(args).cuda()
        meta_model.load_state_dict(model.state_dict())

        oringal_targets = targets.cuda()
        inputs, targets = inputs.cuda(), targets.cuda()
        targets_onehot = torch.nn.functional.one_hot(targets, num_classes).float().cuda()

        # ========================== step 1 ====================================
        outputs = meta_model(inputs)

        #  sample number
        v_lambda = vnet(outputs.detach(), targets, args.sample_number)
        l_f_meta = loss_function((v_lambda * outputs).view(-1, num_classes),
                                 targets_onehot.repeat(args.sample_number, 1)) # monte carlo estimation

        # updata copy_model`s params
        meta_model.zero_grad()
        grads = torch.autograd.grad(l_f_meta, (meta_model.params()), create_graph=True)
        meta_lr = optimizer_model.param_groups[0]['lr']
        meta_model.update_params(lr_inner=meta_lr, source_params=grads)
        del grads

        # ========================= step 2 =====================================
        try:
            inputs_val, targets_val, _ = next(train_meta_loader_iter)
        except StopIteration:
            train_meta_loader_iter = iter(train_meta_loader)
            inputs_val, targets_val, _ = next(train_meta_loader_iter)
        inputs_val, targets_val = inputs_val.cuda(), targets_val.cuda()  # [500,3,32,32], [500]

        y_g_hat = meta_model(inputs_val)
        prec_train = accuracy(y_g_hat.data, targets_val.data, topk=(1,))[0]
        acc_meta += prec_train

        l_g_meta = F.cross_entropy(y_g_hat, targets_val.long())

        # update vnet params
        optimizer_vnet.zero_grad()
        l_g_meta.backward()
        optimizer_vnet.step()

        # ========================= step 3 =====================================
        outputs = model(inputs)
        prec_train = accuracy(outputs.data, oringal_targets.data, topk=(1,))[0]
        acc_train += prec_train

        with torch.no_grad():
            w_new = vnet(outputs.detach(), targets, args.sample_number)

        # v_lambda = vnet(outputs.detach(), targets, args.sample_number)
        loss = loss_function((w_new * outputs).view(-1, num_classes),
                                 targets_onehot.repeat(args.sample_number, 1))
        # loss = loss_function(w_new*outputs, targets_onehot)


        # update model params
        optimizer_model.zero_grad()
        loss.backward()
        optimizer_model.step()

        train_loss += loss.item()
        meta_loss += l_g_meta.item()

    return train_loss/(num+1), meta_loss/(num+1), acc_train/(num+1), acc_meta/(num+1)


def test(model, test_loader):
    model.eval()
    correct = 0
    test_loss = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets, _) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            test_loss += F.cross_entropy(outputs, targets).item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    return accuracy


def adjust_learning_rate(optimizer, epochs, args):
    lr = args.lr * ((0.1 ** int(epochs >= 80)) * (0.1 ** int(epochs >= 100)))  # For WRN-28-10

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    torch.manual_seed(args.seed)

    root = './data_and_load'
    train_loader, train_meta_loader, test_loader = build_dataset(root, args)
    model = build_model(args).cuda()

    global  num_classes
    if args.dataset == 'cifar10':
        num_classes = 10
    else:
        num_classes = 100
    vnet = wpi(2*num_classes, 100, num_classes).cuda()

    optimizer_model = torch.optim.SGD(model.params(), lr=0.1, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer_vnet = torch.optim.Adam(vnet.params(), 3e-4)

    best_acc = 0.0
    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer_model, epoch, args)
        train_loss, meta_loss, acc_train, acc_meta = train(train_loader, train_meta_loader, model, vnet, optimizer_model, optimizer_vnet, epoch)
        test_acc = test(model=model, test_loader=test_loader)
        print("epoch:[%d/%d]\t train_loss:%.4f\t meta_loss:%.4f\t train_acc:%.4f\t meta_acc:%.4f\t test_acc:%.4f\t \n" % ((epoch + 1), args.epochs, train_loss, meta_loss, acc_train, acc_meta, test_acc))
        print( "epoch:[%d/%d]\t, train_loss:%.4f\t, meta_loss:%.4f\t, train_acc:%.4f\t, meta_acc:%.4f\t, test_acc:%.4f\t \n" % ( (epoch + 1), args.epochs, train_loss, meta_loss, acc_train, acc_meta, test_acc), file=mytxt)

        if test_acc >= best_acc:
            best_acc = test_acc
    print('best_acc: ', best_acc)


if __name__ == '__main__':
    txt_name = args.dataset + '_' + args.corruption_type + '_' + str(args.corruption_prob)
    print(txt_name)
    mytxt = open(txt_name + '.txt', mode='a', encoding='utf-8')
    main()
    mytxt.close()
