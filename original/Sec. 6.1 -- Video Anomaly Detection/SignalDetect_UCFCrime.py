import os
import sys
import time

import cv2
import numpy as np
from tqdm import tqdm  # progress bar
import numpy.random as rd

import torch
import torch.nn as nn

# from torch.nn.utils.rnn import pad_sequence

GPU_id = sys.argv[0][-4]
Data_dir = '/mnt/sdb1/weit/datasets/UCF_Crimes'
Mod_dir = 'VideoDetect_{}'.format(GPU_id)
Inp_dim = 400
Mid_dim = 192
Out_dim = 1  # 14-1
Mid_layers = 3


class RegRNN(nn.Module):
    def __init__(self, input_size, out_dim, mid_dim, mid_layers):
        super(RegRNN, self).__init__()

        # self.rnn = nn.LSTM(input_size, mid_dim, mid_layers, dropout=0.25)  # rnn
        self.rnn = nn.GRU(input_size, mid_dim, mid_layers, dropout=0.5)  # rnn
        self.reg_dropout = nn.Dropout(p=0.5)
        self.reg = nn.Sequential(
            nn.BatchNorm1d(mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, mid_dim),
            nn.BatchNorm1d(mid_dim),
            nn.ReLU(),
            self.reg_dropout,
            nn.Linear(mid_dim, out_dim),
            nn.Sigmoid(),
        )  # regression
        self.relu6 = nn.ReLU6()

    def forward(self, x):
        self.rnn.dropout = rd.uniform(0.375, 0.625)
        x = self.rnn(x)[0]

        self.reg_dropout.p = rd.uniform(0.375, 0.625)
        seq_len, batch_size, hid_dim = x.shape
        x = x.view(-1, hid_dim)
        x = self.reg(x)

        x = self.relu6((x - 0.10) * 7.5)
        x = x.view(seq_len, batch_size, -1)
        return x

    def not_relu6(self, x):
        x = self.rnn(x)[0]

        seq_len, batch_size, hid_dim = x.shape
        x = x.view(-1, hid_dim)
        x = self.reg(x)

        # x = self.relu6((x - 0.10) * 8)
        x = x.view(seq_len, batch_size, -1)
        return x

    def output_y_h(self, x, h):
        y, h = self.rnn(x, h)

        seq_len, batch_size, hid_dim = y.size()
        y = y.view(-1, hid_dim)
        y = self.reg(y)
        y = y.view(seq_len, batch_size, -1)
        return y, h


class DataUCFCrimes:
    def __init__(self, root_dir='./UCF_Crimes'):
        assert os.path.isdir(root_dir)

        self.root_dir = root_dir
        self.video_dirs = [
            'Normal_Videos_event',  # 1
            'Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary',
            'Explosion', 'Fighting',
            'RoadAccidents', 'Robbery', 'Shooting', 'Shoplifting',
            'Stealing', 'Vandalism',

            'Training_Normal_Videos_Anomaly',  # 14
            'Testing_Normal_Videos_Anomaly',  # 15
        ]

        self.mp4_dir_name = 'Videos'
        self.npy_dir_name = 'Videos_I3D_npy'

    def get_dir_walks(self, dir_name):
        assert dir_name in {self.mp4_dir_name, self.npy_dir_name, }
        dir_path = '{}/{}'.format(self.root_dir, dir_name)
        child_walks = os.walk(dir_path)
        # for path, dirs, files in child_walks:
        #     print(len(path), len(dirs), len(files))

        file_walks = self.video_dirs.copy()
        for path, dirs, files in child_walks:
            name = path.split('/')[-1]
            try:
                idx = file_walks.index(name)
                files.sort()
                file_walks[idx] = (path, dirs, files)
            except ValueError:
                pass

        # for i in video_walks:
        #     name = i[0].split('/')[-1]
        #     print(name)
        return file_walks  # [video_type, (path, dirs, files)]

    @staticmethod
    def mp42npy(path, max_len=2 ** 17):
        cap = cv2.VideoCapture(path)
        is_opened, img = cap.read()

        cap_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # cap_fps = int(cap.get(cv2.CAP_PROP_FPS))
        # cap_gap = int(1000 / cap_fps)
        # print(img.shape, cap_len, cap_fps)

        cap_len = max_len if cap_len > max_len else cap_len

        imgs = np.empty((cap_len, 224, 224, 3), dtype=np.uint8)
        for i in range(cap_len):
            img = cv2.resize(img, (224, 224))
            # print(imgs.shape, img.shape, is_greyscale)
            imgs[i] = img
            # cv2.imshow('', img)
            # cv2.waitKey(cap_gap // 2)
            img = cap.read()[1]

        # # imgs = list()
        # imgs = np.empty((cap_len, 224, 224, 3), dtype=np.uint8)
        # while is_opened:
        #     img = cv2.resize(img, (224, 224))
        #     # imgs.append(img)
        #     imgs
        #
        #     # cv2.imshow('', img)
        #     # cv2.waitKey(cap_gap // 2)
        #     is_opened, img = cap.read()
        #
        # imgs = np.stack(imgs)
        # print(imgs.shape)
        return imgs

    def print_information(self):
        informance_dict = {
            'root_dir': self.root_dir,
            'chile_dirs': self.video_dirs,
            'link of source code': "https://github.com/WaqasSultani/AnomalyDetectionCVPR2018",
            'link of dataset': 'https://visionlab.uncc.edu/download/category/60-data',
            'latest update time': '2019-12-24 by Github Yonv1943 Zen4 Jia1Hao2'
        }

        for k, v in informance_dict.items():
            print("{}:\t{}".format(k, v))
        print('\n')

    @staticmethod
    def play_video(path):
        # path = '/mnt/sdb1/weit/datasets/UCF_Crimes/Videos/Abuse/Abuse001_x264.mp4'  # repeat
        # path = '/mnt/sdb1/weit/datasets/UCF_Crimes/Videos/Fighting/Fighting032_x264.mp4'  # switch
        # path = '/mnt/sdb1/weit/datasets/UCF_Crimes/Videos/Stealing/Stealing068_x264.mp4'  #
        # path = '/mnt/sdb1/weit/datasets/UCF_Crimes/Videos/Burglary/Burglary062_x264.mp4'  # shape 224 30
        cap = cv2.VideoCapture(path)

        is_opened, img = cap.read()
        cap_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap_fps = int(cap.get(cv2.CAP_PROP_FPS))
        cap_gap = int(1000 / cap_fps)
        print(img.shape, cap_len, cap_fps)
        while is_opened:
            cv2.imshow('', img)
            cv2.waitKey(cap_gap // 2)
            is_opened, img = cap.read()
        # else:
        #     cv2.waitKey()


class BatchSeq:
    def __init__(self, device):
        self.device = device

        '''get_train_eval_walks'''
        rate_of_train_set = 0.8
        data = DataUCFCrimes(Data_dir)
        seq_walks = data.get_dir_walks(data.npy_dir_name)
        # for npy_dir, dirs, files in seq_walks:
        #     name = npy_dir.split('/')[-1]
        #     print(name, len(os.listdir(npy_dir)))

        _video_dirs = [
            'Normal_Videos_event',  # 0

            'Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary',
            'Explosion', 'Fighting',
            'RoadAccidents', 'Robbery', 'Shooting', 'Shoplifting',
            'Stealing', 'Vandalism',  # 13

            'Training_Normal_Videos_Anomaly',  # 14
            'Testing_Normal_Videos_Anomaly',  # 15
        ]

        # seq_walks = ['Normal_Videos_Anomaly', 'Normal_Videos_event', 'Abuse', ... , 'Vandalism']
        # seq_walks = [0,                        1,                    2, ... ,       13]
        train_seq_walks = [seq_walks[14], ]
        eval_seq_walks = [seq_walks[15], ]
        for npy_dir, dirs, files in seq_walks[1:14]:
            train_num = int(len(files) * rate_of_train_set)

            train_seq_walks.append((npy_dir, dirs, files[:train_num]))
            eval_seq_walks.append((npy_dir, dirs, files[train_num:]))

        self.train_seq_walks = train_seq_walks
        self.eval_seq_walks = eval_seq_walks
        self.seq_walks = seq_walks
        self.data = data

    def get__inp_lab(self, plan_len, is_train=True):
        seq_walks = self.train_seq_walks if is_train else self.eval_seq_walks

        batch_seq = np.empty((plan_len, 2, Inp_dim), dtype=np.float32)
        batch_lab = np.empty((plan_len, 2, 1), dtype=np.float32)
        '''fill normal seq'''
        dir0, dirs, file0s = seq_walks[0]  # normal
        total_len0 = 0

        # seq_num=0
        while total_len0 < plan_len:
            ary0 = np.load('{}/{}'.format(dir0, rd.choice(file0s)))
            len0 = ary0.shape[0]
            add_len0 = min(len0, plan_len - total_len0)

            batch_seq[total_len0:total_len0 + add_len0, 0] = ary0[:add_len0]
            total_len0 += add_len0
            # seq_num += 1
        # print(seq_num)
        batch_lab[:, 0] = 0.00

        '''fill normal mix abnormal seq'''
        seq_id = rd.randint(2, len(self.train_seq_walks))
        total_len = 0
        while total_len < plan_len:
            is_normal = bool(rd.rand() < 0.5)

            dir_path, dirs, files = seq_walks[0] if is_normal \
                else seq_walks[seq_id]

            ary = np.load('{}/{}'.format(dir_path, rd.choice(files), ))
            ary_len = ary.shape[0]
            add_len = min(ary_len, plan_len - total_len)

            batch_seq[total_len:total_len + add_len, 1] = ary[:add_len]
            batch_lab[total_len:total_len + add_len, 1] = 0.00 if is_normal else 6.00
            if is_normal:
                batch_lab[total_len:total_len + add_len, 1] = 0.00
            else:
                batch_lab[total_len:total_len + add_len, 1] = 6.00
                batch_lab[total_len:total_len + 8, 1] = 0.00
            total_len += add_len

        batch_seq = torch.tensor(batch_seq.reshape((-1, 16, Inp_dim)), device=self.device)
        batch_lab = torch.tensor(batch_lab.reshape((-1, 16, Out_dim)), device=self.device)
        return batch_seq, batch_lab

    def get_eva_inp(self, len0, video_id=None):
        dir0, dirs, file0s = self.eval_seq_walks[0]  # Testing_Normal_Videos_Anomaly
        video_id = rd.randint(2, 14) if video_id is None else video_id

        load_len0 = 0
        ary0 = None
        while load_len0 <= len0:
            ary0 = np.load('{}/{}'.format(dir0, rd.choice(file0s)))
            load_len0 = ary0.shape[0]
        ary0 = ary0[:len0]
        seq0 = torch.tensor(ary0, dtype=torch.float32, device=self.device)

        dir1, dirs, file1s = self.eval_seq_walks[video_id]
        ary1 = np.load('{}/{}'.format(dir1, rd.choice(file1s)))
        seq1 = torch.tensor(ary1, dtype=torch.float32, device=self.device)

        seq = torch.cat((seq0, seq1), dim=0)
        seq = seq.view((-1, 1, Inp_dim))

        return seq

    def get_rd_eva_normal_seq(self):
        dir_path, dirs, files = self.eval_seq_walks[0]
        ary = np.load('{}/{}'.format(dir_path, rd.choice(files)))
        ary_len = ary.shape[0]
        rd_l = rd.randint(ary_len // 2, ary_len - 1)
        rd_i = rd.randint(rd_l)
        ary = ary[rd_i: rd_i + rd_l]

        return ary


class I3D:
    def __init__(self):
        mod_path, num_classes = './models/rgb_imagenet.pt', 400
        # mod_path, num_classes = './models/rgb_charades.pt', 157

        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(16)
        torch.manual_seed(1943)

        '''build model'''
        from pytorch_i3d import InceptionI3d
        os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_id)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        i3d = InceptionI3d(num_classes=400, in_channels=3).to(device)
        # i3d.replace_logits(num_classes) # for the pre-training model in charades dataset (indoor video)
        i3d.load_state_dict(torch.load(mod_path, map_location=lambda storage, loc: storage))
        i3d.eval()
        self.net = i3d
        self.device = device

    def extract_features(self, imgs, min_len=33, max_len=2 ** 8):
        imgs_len = imgs.shape[0]
        clip_len = (imgs_len - min_len) // max_len + 1
        outs = list()
        for i in range(clip_len):
            j = i * max_len
            k = j + max_len + 1
            if k <= imgs_len:
                pass
            elif k - j >= min_len:
                k -= (k - 1) % (min_len - 1)
            else:  # elif k - j < min_len:
                continue

            ary = imgs[j:k]
            ary = ary[np.newaxis]
            # ary.shape == (batch, time, high, width, channel)
            ary = ary.transpose((0, 4, 1, 2, 3))
            # ary.shape == (batch, channel, time, high, width)
            inp = torch.tensor(ary, dtype=torch.float32, device=self.device)
            inp /= 128.0
            inp -= 1.0

            out = self.net(inp)
            out = out.cpu().data.numpy()[0]
            out = out.transpose((1, 0))

            outs.append(out)

        # print(imgs_len, clip_len, len(outs))
        outs = np.vstack(outs)
        return outs

    def run_test(self):
        imgs = np.ones((rd.randint(1234, 2345), 224, 224, 3))
        print('Inp.shape', imgs.shape)

        ary = imgs
        time_gap = 33
        max_batch = 8

        ary_len0 = ary.shape[0]
        ary = np.reshape(ary[:ary_len0 - (ary_len0 % time_gap)],
                         (-1, time_gap, 224, 224, 3))
        # ary.shape == (batch, time, high, width, channel)

        # out_list = list()
        ary_len1 = ary.shape[0]
        outs = np.empty((ary_len1, 400, 4), dtype=np.float32)
        for i in range(0, ary_len1, max_batch):
            inp = torch.tensor(ary[i:i + max_batch], dtype=torch.float32, device=self.device)
            inp /= 128
            inp -= 1.0
            # inp.shape == (batch, time, high, width, channel)
            inp = inp.permute(0, 4, 1, 2, 3)
            # inp.shape == (batch, channel, time, high, width)

            out = self.net(inp)
            out = out.cpu().data.numpy()
            # out_list.append(out)
            print(out.shape, i)
            outs[i:i + out.shape[0]] = out

        # outs = np.concatenate(out_list, axis=0)
        print(outs.shape)
        return outs

    def run_demo(self):
        mod_path, num_classes = './models/rgb_imagenet.pt', 400
        # mod_path, num_classes = './models/rgb_charades.pt', 157

        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(16)
        torch.manual_seed(1943)

        '''build model'''
        os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_id)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = self.net
        # from pytorch_i3d import InceptionI3d
        # net = InceptionI3d(num_classes=400, in_channels=3).to(device)
        # net.replace_logits(num_classes) # for the pre-training model in charades dataset (indoor video)

        '''load model'''
        net.load_state_dict(torch.load(mod_path, map_location=lambda storage, loc: storage))

        b, c, t, h, w = 2, 3, 32 + 1, 224, 224
        # batch, channel, time, high, weight
        inp = torch.randn(b, c, t, h, w, dtype=torch.float32, device=device)
        print('inp.size():', inp.size(), '==', (b, c, t, 224, 224))
        out = net(inp)
        print('out.size():', out.size(), '==', (b, num_classes, (t - 1) // 8))


def run_extract_features_from_video_by_i3d():
    #  convert video (mp4) to I3D feature (npy)

    data = DataUCFCrimes(Data_dir)
    mp4_dir_name = data.mp4_dir_name

    i3d = I3D()

    '''loop'''
    npy_dir_name = data.npy_dir_name
    npy_replace_name = '/{}/'.format(npy_dir_name)
    npy_dir = '{}/{}'.format(Data_dir, npy_dir_name)
    os.makedirs(npy_dir, exist_ok=True)

    mp4_walks = data.get_dir_walks(mp4_dir_name)
    for mp4_dir_path, dirs, files in mp4_walks:
        npy_dir_path = mp4_dir_path.replace('/Videos/', npy_replace_name)
        os.makedirs(npy_dir_path, exist_ok=True)

        if len(os.listdir(mp4_dir_path)) == len(os.listdir(npy_dir_path)):
            print('finish :', mp4_dir_path)
            continue
        else:
            print('convert:', mp4_dir_path)

        for file in tqdm(files):
            mp4_path = '{}/{}'.format(mp4_dir_path, file)
            npy_path = mp4_path.replace('/Videos/', npy_replace_name).replace('.mp4', '.npy')
            if os.path.exists(npy_path):
                continue

            imgs = data.mp42npy(mp4_path)

            ary = i3d.extract_features(imgs)
            # print(imgs.shape, ary.shape)
            # if ary is None:
            #     print("Too Long. skip:", mp4_path)
            # else:
            #     np.save(npy_path, ary)
            np.save(npy_path, ary)


def run_reshape_normalization():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bs = BatchSeq(device)

    """get_normalization_mean_std"""
    print("| Run: get_normalization_mean_std")
    save_name = 'mean_std__normalization.npz'
    max_len = 2 ** 14  # memory limit

    if os.path.isfile(save_name):
        print("BatchSeq.get_normalization_mean_std() loading")
        # npz = np.load(save_name, allow_pickle=True)
        # mean, std = [v for v in npz.values()]
    else:
        with torch.no_grad():
            print("BatchSeq.get_normalization_mean_std() running")
            npy_dir, dirs, files = bs.train_seq_walks[0]

            '''numpy. RuntimeWarning overflow, invalid value'''
            # mean = np.zeros((1, Inp_dim), dtype=np.float32)
            # std = np.zeros((1, Inp_dim), dtype=np.float32)
            # file_len = len(files)
            #
            # for file in tqdm(files):
            #     path = '{}/{}'.format(npy_dir, file)
            #     ary = np.load(path, allow_pickle=True)
            #
            #     len_j = min(ary.shape[0], max_len)
            #     len_i = rd.randint(ary.shape[0] - len_j + 1)
            #     ary = ary[len_i:len_i + len_j]
            #
            #     ary = np.nan_to_num(ary)
            #     ary = ary.transpose((0, 2, 1))
            #     ary = ary.reshape((-1, 400))
            #     ary = np.tanh(ary / 256) * 256
            #
            #     mean += ary.mean(axis=0, keepdims=True) / file_len
            #     std += ary.std(axis=0, keepdims=True) / file_len

            '''torch. avoid overflow and faster than numpy'''
            mean = torch.zeros((1, Inp_dim), device=device)
            std = torch.zeros((1, Inp_dim), device=device)
            file_len = len(files)

            for file in tqdm(files):
                path = '{}/{}'.format(npy_dir, file)
                ary = np.load(path, allow_pickle=True)
                # ary = ary.transpose((0, 2, 1))
                # ary = ary.reshape(-1, Inp_dim)
                ary = np.nan_to_num(ary)

                len_j = min(ary.shape[0], max_len)
                len_i = rd.randint(ary.shape[0] - len_j + 1)
                ary = ary[len_i:len_i + len_j]

                ten = torch.tensor(ary, dtype=torch.float32, device=device)
                # ten[ten != ten] = 0  # np.nan_to_num(ary)
                # ten = ten.permute(0, 2, 1)
                # ten = ten.reshape(-1, Inp_dim)
                ten = torch.tanh(ten / 256) * 256

                mean += ten.mean(dim=0, keepdim=True) / file_len
                std += ten.std(dim=0, keepdim=True) / file_len

            mean = mean.cpu().data.numpy()
            std = std.cpu().data.numpy()

            np.savez(save_name, mean, std)
        mean = torch.tensor(mean, dtype=torch.float32, device=device)
        std = torch.tensor(std, dtype=torch.float32, device=device)

        """reshape_and_normalization"""
        print("| Run: reshape_and_normalization")
        for npy_dir, dirs, files in bs.seq_walks:
            # print(npy_dir.split('/')[-1])
            for file in tqdm(files):
                path = '{}/{}'.format(npy_dir, file)
                ary = np.load(path, allow_pickle=True)

                if len(ary.shape) == 2:
                    continue
                ary = ary.transpose((0, 2, 1))
                ary = ary.reshape(-1, Inp_dim)
                ary = np.nan_to_num(ary)

                ten = torch.tensor(ary, dtype=torch.float32, device=device)
                ten = torch.tanh(ten / 256) * 256
                ten = (ten - mean) / std

                ary = ten.cpu().data.numpy()
                np.save(path, ary)
        print("| Done: reshape_and_normalization")
    pass


def run_test():
    import matplotlib.pyplot as plt
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # data = DataUCFCrimes(Data_dir)
    bs = BatchSeq(device)

    # dir_walks = bs.eval_seq_walks

    # i = 0
    # for dir_path, dirs, files in dir_walks:
    #     name = dir_path.split('/')[-1]
    #     print(i, name)
    #     i += 1

    '''net'''
    os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_id)
    torch.set_default_dtype(torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = RegRNN(Inp_dim, Out_dim, Mid_dim, Mid_layers).to(device)
    net.load_state_dict(torch.load('%s/net.pth' % (Mod_dir,),
                                   map_location=lambda storage, loc: storage))

    # type_id = ((8, 18), (0, 118),)[:]
    len0 = 2 ** 12
    from tqdm import trange
    for video_id, type_color in ((None, 'lightcoral'), (0, 'royalblue')):
        for _ in trange(32):
            inp = bs.get_eva_inp(len0, video_id=video_id)
            out = net(inp)
            # out = blur1d_axis0(out, kernel_size=1024)

            out0 = out[:len0]
            out_bias = out0.mean()  # +out0.std()
            # out_bias = (out0).max()
            out = (out - out_bias) / (1 - out_bias)

            ary = out.cpu().data.numpy()
            ary = ary.flatten()
            # print(ary.shape)

            plt.plot(ary, alpha=0.4, color=type_color)

    plt.savefig('{}/plot.png'.format(Mod_dir))
    plt.show()


def run_train():
    batch_len = int(2 ** 16)
    train_epoch = int(2 ** 5 * 2)
    show_gap = 2 ** 4

    '''build model'''
    os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_id)
    torch.set_default_dtype(torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = RegRNN(Inp_dim, Out_dim, Mid_dim, Mid_layers).to(device)
    # net.load_state_dict(torch.load('%s/net.pth' % (Mod_dir,),
    #                                map_location=lambda storage, loc: storage))
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    bs = BatchSeq(device)

    '''training loop'''
    print('Training')
    start_time = show_time = time.time()
    loss_list = list()
    eva_loss_list = list()
    try:
        net.train()
        for epoch in range(train_epoch):
            inp, lab = bs.get__inp_lab(batch_len, is_train=True)
            if inp is None:
                continue

            out = net(inp)
            loss = criterion(out, lab)
            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if time.time() > show_time + show_gap:
                show_time = time.time()

                inp, lab = bs.get__inp_lab(batch_len, is_train=False)
                if inp is None:
                    continue

                net.eval()
                out = net(inp)
                eva_loss = criterion(out, lab)
                eva_loss_list.append(eva_loss.item())
                net.train()

                loss_smooth = np.average(loss_list[-16:])
                eva_loss_smooth = np.average(eva_loss_list[-16:])
                print('Epoch: {:6}    EpoL: {:.2e}    EvaL: {:.2e}'.format(
                    epoch, loss_smooth, eva_loss_smooth))

    except KeyboardInterrupt:
        print("KeyboardInterrupt")
    # except Exception as error:
    #     print("Error:", error)
    finally:
        os.makedirs(Mod_dir, exist_ok=True)
        torch.save(net.state_dict(), '%s/net.pth' % (Mod_dir,))

        print("Saved:", Mod_dir)

    print("Times Used:", int(time.time() - start_time))


def run_eval_fap(h_list):
    is_draw = False

    """init"""
    save_npz_path = '{}/eva_fap__normal.npz'.format(Mod_dir)

    if os.path.exists(save_npz_path):
        ary_list = np.load(save_npz_path, allow_pickle=True)['arr_0']
    else:

        '''build model'''
        os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_id)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = RegRNN(Inp_dim, Out_dim, Mid_dim, Mid_layers).to(device)
        net.eval()

        '''load model'''
        net.load_state_dict(torch.load('%s/net.pth' % (Mod_dir,), map_location=lambda storage, loc: storage))

        '''data'''
        bs = BatchSeq(device)

        '''eval loop'''
        ary_list = list()
        # print('len            Normal_Videos_event:  50', len(bs.seq_walks[0][2]))
        # print('len Training_Normal_Videos_Anomaly: 800', len(bs.train_seq_walks[0][2]))
        # print('len  Testing_Normal_Videos_Anomaly: 150', len(bs.eval_seq_walks[0][2]))
        for seq_walks in (
                # bs.seq_walks[0], bs.train_seq_walks[0],
                bs.eval_seq_walks[0],
        ):
            dir_path, dirs, file0s = seq_walks
            for file in tqdm(file0s):
                ary = np.load('{}/{}'.format(dir_path, file))
                # print(ary.shape[0], file0)
                inp = torch.tensor(ary[:, np.newaxis], dtype=torch.float32, device=device)
                out = net.not_relu6(inp)
                # out = blur1d_axis0(out)

                ary = out.cpu().data.numpy()
                ary = ary.flatten()
                ary_list.append(ary)

        np.savez(save_npz_path, ary_list)

    if is_draw:
        ary_avg_len = int(np.mean([ary.shape[0] for ary in ary_list]))
        alpha = 10 / len(ary_list)
        import matplotlib.pyplot as plt
        for ary in ary_list:
            plt.plot(ary[:ary_avg_len], linewidth=4, alpha=alpha, color='k')

        plt.savefig('{}/plot_normal.png'.format(Mod_dir))
        plt.show()
        exit()

    """evaluate False Alarm Period (FAP)"""
    print('|FAP len(ary0_list):', len(ary_list))
    eva_gap = 128

    eva_total_len = sum([ary.shape[0] for ary in ary_list])
    print("|FAP eva_total_len:", eva_total_len)
    for h in h_list:
        c_false_alarm_events = 0

        for ary in ary_list:
            eva_actions = np.where(ary >= h)[0]

            eva_t_list = list()
            if eva_actions.shape[0] != 0:
                eva_t_list.append(eva_actions[0])

                for eva_t in eva_actions:
                    if eva_t >= eva_t_list[-1] + eva_gap:
                        eva_t_list.append(eva_t)

            c_false_alarm_events += len(eva_t_list)

        real_total_len = eva_total_len - c_false_alarm_events * eva_gap
        false_alarm_period = real_total_len / c_false_alarm_events if c_false_alarm_events \
            else np.inf
        print("h {:6}      FAE  {:6}      FAP  {:.3f}".format(
            h, c_false_alarm_events, false_alarm_period))


def run_eval_add(h_list):
    is_draw = False

    """init"""
    save_npy_path = '{}/eva_add__abnormal.npz'.format(Mod_dir)

    if os.path.exists(save_npy_path):
        ary_list = np.load(save_npy_path, allow_pickle=True)['arr_0']
    else:
        '''build model'''
        os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_id)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = RegRNN(Inp_dim, Out_dim, Mid_dim, Mid_layers).to(device)
        net.eval()

        '''load model'''
        net.load_state_dict(torch.load('%s/net.pth' % (Mod_dir,), map_location=lambda storage, loc: storage))

        '''data'''
        bs = BatchSeq(device)

        '''eval loop'''
        ary_list = list()
        print('sum |train_abnormal_seq_len: 759', sum([len(walks[2]) for walks in bs.train_seq_walks[1:]]))
        print('sum | eval_abnormal_seq_len: 191', sum([len(walks[2]) for walks in bs.eval_seq_walks[1:]]))
        for seq_walks in bs.eval_seq_walks[1:]:
            dir_path, dirs, files = seq_walks
            for file in tqdm(files):
                ary = np.load('{}/{}'.format(dir_path, file))
                # print(ary.shape[0], file0)

                inp = torch.tensor(ary[:, np.newaxis], dtype=torch.float32, device=device)
                out = net.not_relu6(inp)
                # out = blur1d_axis0(out)

                ary = out.cpu().data.numpy()
                ary = ary.flatten()
                ary_list.append(ary)

        np.savez(save_npy_path, ary_list)
        # exit()

    if is_draw:
        ary_avg_len = int(np.mean([ary.shape[0] for ary in ary_list]))
        alpha = 10 / len(ary_list)
        import matplotlib.pyplot as plt
        for ary in ary_list:
            plt.plot(ary[:ary_avg_len], linewidth=4, alpha=alpha, color='k')

        plt.title('plot_abnormal.png')
        plt.savefig('{}/plot_abnormal.png'.format(Mod_dir))
        plt.show()
        exit()

    """evaluate Average Detection Delay(ADD)"""
    print('|ADD len(ary0_list):', len(ary_list))
    eva_tau = 0

    for h in h_list:
        # c_false_alarm_events = 0
        c_det_delays = []

        for ary in ary_list:
            eva_actions = np.where(ary >= h)[0]
            if len(eva_actions) == 0:
                continue
            eva_t = eva_actions[0]

            if eva_t < eva_tau:  # at this point, the online decision is 1 ("stop")
                # c_false_alarm_events += 1
                pass
            else:
                det_delay = eva_t - eva_tau
                c_det_delays.append(det_delay)

        avg_det_delay = np.mean(np.array(c_det_delays)) if len(c_det_delays) > 0 else np.nan
        # avg_false_alarm_rate = c_false_alarm_events / len(ary_list)
        # print("h {:6}      ADD  {:.3f}      FAR  {:.3f}".format(
        #     h, avg_det_delay, avg_false_alarm_rate, ))
        print("h {:6}      ADD  {:.3f}".format(
            h, avg_det_delay, ))


if __name__ == '__main__':
    the_h_list = (
        # 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10,
        # 0.01, 0.05, 0.10,
        # 0.10, 0.105, 0.11, 0.115, 0.12,
        # 0.12, 0.125, 0.130, 0.131, 0.132, 0.133, 0.134, 0.135,
        # 0.14, 0.16, 0.18, 0.2,
        # 0.25, 0.3, 0.35, 0.4, 0.45,
        # 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9,
        # 0.95, 0.99, 0.995, 0.996, 0.997, 0.998, 0.999,
        0.1, 0.2, 0.3, 0.4,
        0.5, 0.6, 0.7, 0.8, 0.9, 0.95,
        # 0.91, 0.92, 0.93, 0.94, 0.95,
        # 0.96, 0.97, 0.98,
        # 0.981, 0.982, 0.983, 0.984,
        # 0.992, 0.993, 0.994, 0.995, 0.996,
        # 0.9965, 0.9967, 0.997,
    )
    # run_extract_features_from_video_by_i3d()
    # run_reshape_normalization()
    run_train()
    run_eval_fap(the_h_list)
    run_eval_add(the_h_list)
