import os
import argparse
import torch
from torchvision import transforms
import torch.nn.functional as F
from dataset import Clothing1M
from resnet50 import resnet50, wpi, wpi_dec
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def get_args():
    parser = argparse.ArgumentParser(description='Clothing1M')
    parser.add_argument('--train_batch_size', type=int, default=32, help='input batch size for training (default: 256)')
    parser.add_argument('--meta_batch_size', type=int, default=32, help='input batch size for meta (default: 256)')
    parser.add_argument('--test_batch_size', type=int, default=32, help='input batch size for testing (default: 256)')

    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 120)')
    parser.add_argument('--gpu_id', type=int, default=0, help='index of gpu to use (default: 0)')

    parser.add_argument('--lr', type=float, default=0.01, help='init learning rate (default: 0.1)')
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
    parser.add_argument('--num_classes', type=int, default=14, help='random seed (default: 14)')
    parser.add_argument('--root', type=str, default='', help='random seed (default: 14)')


    args = parser.parse_args()
    return args

def creat_model():
    model = resnet50(pretrained=True)
    return model

def adjust_lr(optimizer, epochs):
    lr = args.lr * (0.1 * int(epochs >= 5))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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

def train(train_loader,train_meta_loader, model, wpi_, optimizer_model, optimizer_vnet, epoch):
    print('\nEpoch: %d' % epoch)

    train_loss = 0
    meta_loss = 0

    train_meta_loader_iter = iter(train_meta_loader)
    for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader)):
        model.train()
        meta_model = creat_model().cuda()
        meta_model.load_state_dict(model.state_dict())

        inputs, targets = inputs.to(device), targets.to(device)
        outputs, feat = meta_model(inputs)
        targets_onehot = torch.nn.functional.one_hot(targets, 14).float().cuda()

        v_lambda = wpi_(outputs.detach(), targets, args.sample_number)
        # print((v_lambda * outputs).view(-1, num_classes).size())
        l_f_meta = loss_function((v_lambda * outputs).view(-1, 14),
                                 targets_onehot.repeat(args.sample_number, 1))  # monte carlo estimation

        # updata copy_model`s params
        meta_model.zero_grad()
        grads = torch.autograd.grad(l_f_meta, (meta_model.params()), create_graph=True)
        meta_lr = args.lr * (0.1 ** int(epoch >= 5))  # For ResNet32
        meta_model.update_params(lr_inner=meta_lr, source_params=grads)
        del grads

        try:
            inputs_val, targets_val = next(train_meta_loader_iter)
        except StopIteration:
            train_meta_loader_iter = iter(train_meta_loader)
            inputs_val, targets_val = next(train_meta_loader_iter)
        inputs_val, targets_val = inputs_val.to(device), targets_val.to(device)  # [500,3,32,32], [500]

        y_g_hat = meta_model(inputs_val)
        l_g_meta = F.cross_entropy(y_g_hat, targets_val)

        prec_meta = accuracy(y_g_hat.data, targets_val.data, topk=(1,))[0]

        # update vnet params
        optimizer_vnet.zero_grad()
        l_g_meta.backward()
        optimizer_vnet.step()

        outputs = model(inputs)
        prec_train = accuracy(outputs.data, targets.data, topk=(1,))[0]

        with torch.no_grad():
            w_new = wpi_(outputs.detach(), targets, args.sample_number)

        loss = F.cross_entropy(w_new * outputs, targets)

        optimizer_model.zero_grad()
        loss.backward()
        optimizer_model.step()

        train_loss += loss.item()
        meta_loss += l_g_meta.item()

        if (batch_idx + 1) % 1000 == 0:
            print('Epoch: [%d/%d]\t'
                  'Iters: [%d/%d]\t'
                  'Loss: %.4f\t'
                  'MetaLoss:%.4f\t'
                  'Prec@1 %.2f\t'
                  'Prec_meta@1 %.2f' % (
                      (epoch + 1), args.epochs, batch_idx + 1, len(train_loader.dataset) / args.batch_size,
                      (train_loss / (batch_idx + 1)),
                      (meta_loss / (batch_idx + 1)), prec_train, prec_meta))
            print('Epoch: [%d/%d]\t'
                  'Iters: [%d/%d]\t'
                  'Loss: %.4f\t'
                  'MetaLoss:%.4f\t'
                  'Prec@1 %.2f\t'
                  'Prec_meta@1 %.2f' % (
                      (epoch + 1), args.epochs, batch_idx + 1, len(train_loader.dataset) / args.batch_size,
                      (train_loss / (batch_idx + 1)),
                      (meta_loss / (batch_idx + 1)), prec_train, prec_meta), file=mytxt)


def test(model, test_loader):
    model.eval()
    correct = 0
    test_loss = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            test_loss += F.cross_entropy(outputs, targets).item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print('Test set: Average loss_picture: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(test_loss, correct, len(test_loader.dataset),accuracy))
    print('Test set: Average loss_picture: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(test_loss, correct, len(test_loader.dataset),accuracy), file=mytxt)

    return accuracy

def main():
    global  device
    torch.manual_seed(args.seed)
    device = torch.device('cuda:' + str(args.gpu_id) if torch.cuda.is_available() else 'cpu')
    root = args.root
    kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}

    transform_train = transforms.Compose([transforms.Resize((256, 256)),
                                          transforms.RandomCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transform_test = transforms.Compose([transforms.Resize((224, 224)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    train_dataset = Clothing1M(root, mode='train', transform=transform_train)
    val_dataset = Clothing1M(root, mode='val', transform=transform_test)  # meta_data
    test_dataset = Clothing1M(root, mode='test', transform=transform_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.meta_batch_size, shuffle=False, **kwargs)  # meta_data
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    base_model = creat_model().cuda()
    # base_model = torch.nn.DataParallel(base_model.cuda(), device_ids=[0, 1])
    vnet = wpi(28, 100, 14).cuda()
    # base_model = torch.nn.DataParallel(base_model.cuda(), device_ids=[0, 1])

    optimizer_model = torch.optim.SGD(base_model.params(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    optimizer_vnet = torch.optim.Adam(vnet.params(), 3e-4)

    best_acc = 0.0
    for epoch in range(args.epochs):
        adjust_lr(optimizer_model, epoch)
        train(train_loader, val_loader, base_model, vnet, optimizer_model, optimizer_vnet, epoch)
        test_acc = test(model=base_model, test_loader=test_loader)
        if test_acc >= best_acc:
            best_acc = test_acc

    print('best accuracy:', best_acc)
    print('best accuracy:', best_acc, file=mytxt)



if __name__ == '__main__':
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
    mytxt = open('clothing1m_log/base_resnet50.txt', mode='a', encoding='utf-8')
    main()
    mytxt.close()











