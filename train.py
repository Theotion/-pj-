from config import *
from model import P_Net, R_Net, O_Net, LossFn
import random
import time
import torch
import torch.nn as nn
import torch.optim as opt
import torchvision
from torch.utils.data import DataLoader
import time
import argparse
import os
import os.path as osp


'''def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--class_data_txt_path',
                        default='F:/MTCNN/wider_face_split/wider_face_train_bbx_gt.txt',
                        type=str, help='the path of WILDER FACE .txt file')
    parser.add_argument('--class_data_dir', default='F:/MTCNN/WIDER_train/images',
                        type=str, help='the dir of WILDER FACE image file')
    parser.add_argument('--class_val_data_txt_path',
                        default='F:/MTCNN/wider_face_split/wider_face_val_bbx_gt.txt',
                        type=str, help='the path of WILDER FACE test.txt file')#用于测试集输出
    parser.add_argument('--class_val_data_dir', default='F:\MTCNN/WIDER_val\images',
                        type=str, help='the dir of WILDER FACE test image file')#用于测试集输出
    parser.add_argument('--class_data_augment', default=5,
                        type=int, help='the augment ratio for create pnet data set')
    parser.add_argument('--landmark_data_txt_path',
                        default='F:/MTCNN/train/trainImageList.txt',
                        type=str, help='the path of CelebA .txt file')
    parser.add_argument('--landmark_data_dir', default='F:/MTCNN/train', type=str,
                        help='the dir of CelebA image file')
    parser.add_argument('--p_net_data', type=str,
                        default='F:/MTCNN/Dataset/P_Net_dataset/P_Net_dataset.txt',
                        help='the path of .txt file including the training data path')
    parser.add_argument('--r_net_data', type=str,
                        default='F:/MTCNN/Dataset/R_Net_dataset/R_Net_dataset.txt',
                        help='the path of .txt file including the training data path')
    parser.add_argument('--o_net_data', type=str,
                        default='F:/MTCNN/Dataset/O_Net_dataset/O_Net_dataset.txt',
                        help='the path of .txt file including the training data path')
    parser.add_argument('--save_folder', type=str,
                        default='F:/MTCNN/MTCNN_Pytorch-master/weights',  # './MTCNN_weighs',
                        help='the folder of p/r/onet_para.pkl, p/r/onet.pkl saved')
    parser.add_argument('--train_net', type=str,
                        default='pnet', choices=['pnet', 'rnet', 'onet'],
                        help='choose net to train')
    parser.add_argument('--lr', type=float,
                        default=0.001,
                        help='initial learning rate')
    # parser.add_argument('--sub_epoch', type=int,
    #                     default=1000,
    #                     help='some batches make up a sub_epoch ')
    parser.add_argument('--batch_size', type=int,
                        default=32,
                        help='batch_size ')
    parser.add_argument('--num_workers', type=int,
                        default=4,
                        help='workers for loading the data')
    parser.add_argument('--half_lr_steps', type=int,
                        default=5000,
                        help='half the lr every half_lr_steps iter')
    # parser.add_argument('--lr_schedule', type=dict,
    #                     default={40000:},
    #                     help='half the lr every half_lr_steps iter')
    parser.add_argument('--save_steps', type=int,
                        default=100,
                        help='save para, model every save_steps iter')

    args = parser.parse_args()

    return args'''
def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--class_data_txt_path',
                        default='D:\计算机视觉导论\dataset\WIDER_Face\wider_face_split/wider_face_train_bbx_gt.txt',
                        type=str, help='the path of WILDER FACE .txt file')
    parser.add_argument('--class_data_dir', default='D:\计算机视觉导论\dataset\WIDER_Face\WIDER_train/images',
                        type=str, help='the dir of WILDER FACE image file')
    parser.add_argument('--class_val_data_txt_path',
                        default='D:\计算机视觉导论\dataset\WIDER_Face\wider_face_split/wider_face_val_bbx_gt.txt',
                        type=str, help='the path of WILDER FACE .txt file')
    parser.add_argument('--class_val_data_dir', default='D:\计算机视觉导论\dataset\WIDER_Face\WIDER_val\images',
                        type=str, help='the dir of WILDER FACE image file')
    parser.add_argument('--class_data_augment', default=5,
                        type=int, help='the augment ratio for create pnet data set')
    parser.add_argument('--landmark_data_txt_path',
                        default='D:\计算机视觉导论\dataset\CNN_FacePoint/train/trainImageList.txt',
                        type=str, help='the path of CelebA .txt file')
    parser.add_argument('--landmark_data_dir', default='D:\计算机视觉导论\dataset\CNN_FacePoint/train', type=str,
                        help='the dir of CelebA image file')
    parser.add_argument('--p_net_data', type=str,
                        default='D:\计算机视觉导论\dataset/P_Net_dataset/P_Net_dataset.txt',
                        help='the path of .txt file including the training data path')
    parser.add_argument('--r_net_data', type=str,
                        default='D:\计算机视觉导论\dataset/R_Net_dataset/R_Net_dataset.txt',
                        help='the path of .txt file including the training data path')
    parser.add_argument('--o_net_data', type=str,
                        default='D:\计算机视觉导论\dataset/O_Net_dataset/O_Net_dataset.txt',
                        help='the path of .txt file including the training data path')
    parser.add_argument('--save_folder', type=str,
                        default='./weights',  # './MTCNN_weighs',
                        help='the folder of p/r/onet_para.pkl, p/r/onet.pkl saved')
    parser.add_argument('--train_net', type=str,
                        default='pnet', choices=['pnet', 'rnet', 'onet'],
                        help='choose net to train')
    parser.add_argument('--lr', type=float,
                        default=0.001,
                        help='initial learning rate')
    # parser.add_argument('--sub_epoch', type=int,
    #                     default=1000,
    #                     help='some batches make up a sub_epoch ')
    parser.add_argument('--batch_size', type=int,
                        default=32,
                        help='batch_size ')
    parser.add_argument('--num_workers', type=int,
                        default=4,
                        help='workers for loading the data')
    parser.add_argument('--half_lr_steps', type=int,
                        default=5000,
                        help='half the lr every half_lr_steps iter')
    # parser.add_argument('--lr_schedule', type=dict,
    #                     default={40000:},
    #                     help='half the lr every half_lr_steps iter')
    parser.add_argument('--save_steps', type=int,
                        default=100,
                        help='save para, model every save_steps iter')

    args = parser.parse_args()

    return args

# optimizer =Opt.Adam(pnet.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, amsgrad=True)
def load_net(args, net_name):
    # pnet, rnet, onet = P_Net(), R_Net(), O_Net()
    net_list = {'pnet': P_Net(), 'rnet': R_Net(), 'onet': O_Net()}
    try:
        net = net_list[net_name].to(DEVICE)
        try:
            print('===> loading the saved net weights...')
            _ = osp.join(args.save_folder, net_name + '.pkl')
            print('===> check {} saved path({}):{}'.format(net_name, _, osp.exists(_)))
            net.load_state_dict(torch.load(_, map_location=DEVICE))
            return net  # , rnet, onet
        except Exception:
            print('*** fail to load the saved net weights!')
            return net
    except Exception:
        print('*** Net name wrong!')


def load_para(file_name):
    # para = None
    try:
        print('===> loading the saved parameters...')
        para = torch.load(osp.join(args.save_folder, file_name))
    except Exception:
        print('*** fail to load the saved parameters!')
        print('===> initailizing the parameters...')
        para = {
            'lr': args.lr,
            'iter': 0,
            'loss': [],
            'val_result': [],
            'optimizer_param': None
        }
        save_safely(para, dir_path=args.save_folder, file_name=file_name)
    return para


def get_dataset(args, net_name):
    from dataset import mtcnn_dataset
    net_list = {'pnet': args.p_net_data,
                'rnet': args.r_net_data,
                'onet': args.o_net_data}
    _ = None
    try:
        _ = net_list[net_name]
    except Exception:
        print('*** Net name wrong!')

    # print(osp.exists(_))
    print('===> loading the data_list, path is {}'.format(_))
    data_list = load_txt(_)
    data_dir = osp.split(_)[0]
    print('===> the data dir is {}'.format(data_dir))
    dataset = mtcnn_dataset(data_list=data_list, data_dir=data_dir)
    return DataLoader(dataset,
                      batch_size=args.batch_size,
                      shuffle=True,
                      num_workers=args.num_workers,
                      pin_memory=False)


def get_inplace_data_set(args, net_name):
    from dataset import InplaceDataset
    from create_dataset import create_pnet_data_txt_parser, landmark_dataset_txt_parser
    img_faces = create_pnet_data_txt_parser(args.class_data_txt_path, args.class_data_dir)
    img_face_landmark = landmark_dataset_txt_parser(args.landmark_data_txt_path, args.landmark_data_dir)
    if net_name == 'pnet':
        IDS = InplaceDataset(img_face_landmark, img_faces, cropsize=12)
    elif net_name == 'rnet':
        IDS = InplaceDataset(img_face_landmark, img_faces, cropsize=24, pnet=load_net(args, 'pnet').to('cpu'))
    elif net_name == 'onet':
        IDS = InplaceDataset(img_face_landmark, img_faces, cropsize=48,
                             pnet=load_net(args, 'pnet').to('cpu'), rnet=load_net(args, 'rnet').to('cpu'))
    return DataLoader(IDS,
                      batch_size=args.batch_size,
                      shuffle=True,
                      num_workers=args.num_workers,
                      pin_memory=False)


def get_inplace_data_set_val(args, net_name):
    from dataset import InplaceDataset
    from create_dataset import create_pnet_data_txt_parser, landmark_dataset_txt_parser
    img_faces = create_pnet_data_txt_parser(args.class_val_data_txt_path, args.class_val_data_dir)
    img_face_landmark = landmark_dataset_txt_parser(args.landmark_data_txt_path, args.landmark_data_dir)
    if net_name == 'pnet':
        IDS = InplaceDataset(img_face_landmark, img_faces, cropsize=12)
    elif net_name == 'rnet':
        IDS = InplaceDataset(img_face_landmark, img_faces, cropsize=24, pnet=load_net(args, 'pnet').to('cpu'))
    elif net_name == 'onet':
        IDS = InplaceDataset(img_face_landmark, img_faces, cropsize=48,
                             pnet=load_net(args, 'pnet').to('cpu'), rnet=load_net(args, 'rnet').to('cpu'))
    return DataLoader(IDS,
                      batch_size=args.batch_size,
                      shuffle=True,
                      num_workers=args.num_workers,
                      pin_memory=False)


def save_safely(file, dir_path, file_name):
    if not osp.exists(dir_path):
        os.mkdir(dir_path)
        print('*** dir not exist, created one')
    save_path = osp.join(dir_path, file_name)
    if osp.exists(save_path):
        temp_name = save_path + '.temp'
        torch.save(file, temp_name)
        os.remove(save_path)
        os.rename(temp_name, save_path)
        print('*** find the file conflict while saving, saved safely')
    else:
        torch.save(file, save_path)


def train_net(args, net_name='pnet', use_inplace_dataset=True, loss_config=[]):
    net = load_net(args, net_name)
    para = load_para(net_name + '_para.pkl')
    lr = para['lr']
    iter_count = para['iter']
    optimizer = opt.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, amsgrad=True)#这里也因学习率预热而改变
    loss = LossFn(cls_factor=loss_config[0], box_factor=loss_config[1], landmark_factor=loss_config[2])
    if para['optimizer_param'] is not None:
        optimizer.state_dict()['param_groups'][0].update(para['optimizer_param'])
        print('===> updated the param of optimizer.')
    t_500=0#用于声明初始化500次输出时间
    losslist=[]
    t_losslist = []
    while True:
        if use_inplace_dataset:
            data_set = get_inplace_data_set(args, net_name)
        else:
            data_set = get_dataset(args, net_name)
        t0 = time.perf_counter()
        for _, (img_tensor, label, offset, landmark_flag, landmark) in enumerate(data_set, iter_count):
            iter_count += 1
            # print('tp:{}'.format(tp))
            # update lr rate
            wrap = (img_tensor, label, offset, landmark)
            (img_tensor, label, offset, landmark) = [i.to(DEVICE) for i in wrap]
            det, box, ldmk = net(img_tensor)
            optimizer.zero_grad()
            # print('offset:', offset)
            all_loss = loss.total_loss(gt_label=label, pred_label=det, gt_offset=offset, pred_offset=box,
                                       landmark_flag=landmark_flag, pred_landmark=ldmk, gt_landmark=landmark)
            t1 = time.perf_counter()
            t_500+=t1-t0
            if iter_count%500==0:
                print('===> iter:{}\t| loss:{:.8f}\t| lr:{:.12f} | time:{:.8f}'
                      .format(iter_count, all_loss.item(), lr, t_500))  # 该两行用于每500次迭代输出
                losslist.append([iter_count, all_loss.item(), lr])
                t_500=0
            if net_name=='onet':
                if iter_count % 500 == 0:  # 从这里开始用于onet，用于输出test结果
                    data_set_val = get_inplace_data_set_val(args, net_name)
                    tt0 = t1
                    for _, (img_tensor, label, offset, landmark_flag, landmark) in enumerate(data_set_val, iter_count):
                        wrap = (img_tensor, label, offset, landmark)
                        (img_tensor, label, offset, landmark) = [i.to(DEVICE) for i in wrap]
                        det, box, ldmk = net(img_tensor)
                        all_loss_t = loss.total_loss(gt_label=label, pred_label=det, gt_offset=offset, pred_offset=box,
                                                   landmark_flag=landmark_flag, pred_landmark=ldmk,
                                                   gt_landmark=landmark)
                        tt1 = time.perf_counter()
                        print('===> valtra:{}\t| loss:{:.8f}\t| lr:{:.12f} | time:{:.8f}'
                              .format(iter_count, all_loss_t.item(), lr, tt1 - tt0))
                        t_losslist.append([iter_count, all_loss_t.item(), lr])
                        break  # onet应用结束
            if iter_count>=25000:
                print(losslist)
                if net_name =='onet':
                    print(t_losslist)
                break
            # print('===> iter:{}\t| loss:{:.8f}\t| lr:{:.12f} | time:{:.8f}'
            #       .format(iter_count, all_loss.item(), lr, t1 - t0))#该两行用于每次迭代输出
            # print(all_loss)
            if iter_count<=1000:#以下为学习率预热过程
                if iter_count==0:
                    lr = (iter_count + 1) / 1000 * args.lr
                else:
                    lr = iter_count / 1000 * args.lr
                para.update({'lr': lr})
                for param_groups in optimizer.param_groups:
                    param_groups['lr'] = lr
                if iter_count%100==0:
                    print('*** lr updated:{}'.format(lr))#学习率预热结束
            t0 = time.perf_counter()
            all_loss.backward()
            optimizer.step()
            if 0 == iter_count % args.save_steps:
                if (0 == iter_count % args.half_lr_steps) & (iter_count > 1000):#这里也因学习率预热而改动
                    lr /= 2#这个是lr阶段式减少
                    para.update({'lr': lr})
                    for param_groups in optimizer.param_groups:
                        param_groups['lr'] = lr
                    print('*** lr updated:{}'.format(lr))
                para.update({
                    'lr': lr,
                    'iter': iter_count,
                    'optimizer_param': optimizer.state_dict()['param_groups'][0]
                })
                save_safely(net.state_dict(), args.save_folder, net_name + '.pkl')
                save_safely(para, args.save_folder, net_name + '_para.pkl')
        if iter_count >= 25000:
            break


def load_txt(data_path):
    samples = []
    with open(data_path, 'r') as f:
        lines = list(map(lambda line: line.strip().split('\n'), f))
        # lines[[str],[str],[]...]
        lines = [i[0] for i in lines]
        # lines [str,str...]
        for line in lines:
            line = line.split()
            if DEBUG: print('line:', line)
            img_path = line[0]
            img_class = line[1]
            offset, landmark = [], []
            if img_class is 'p' or 'pf' or 'l':
                offset = [float(s) for s in line[2:6]]
            if img_class is 'l':
                landmark = [float(s) for s in line[6:]]
            sample = [img_path, img_class, offset, landmark]
            samples.append(sample)
    # print('samples:', samples)
    return samples


net_loss_config = {
    'pnet': [1.0, 0.5, 0.5],
    'rnet': [1.0, 0.5, 0.5],
    'onet': [1.0, 0.5, 1.0]
}

if __name__ == '__main__':
    args = config()
    # data_set = get_dataset(args, 'pnet')[:10]
    # print(data_set)
    # train_net(args, args.train_net, loss_config=net_loss_config[args.train_net])
    if args.train_net=='pnet':
        train_net(args, args.train_net, loss_config=net_loss_config[args.train_net])
        # train_net(args, 'rnet', loss_config=net_loss_config['rnet'])
        # train_net(args, 'onet', loss_config=net_loss_config['onet'])
    if args.train_net=='rnet':
        train_net(args, args.train_net, loss_config=net_loss_config[args.train_net])
        train_net(args, 'onet', loss_config=net_loss_config['onet'])
    if args.train_net=='onet':
        train_net(args, args.train_net, loss_config=net_loss_config[args.train_net])
