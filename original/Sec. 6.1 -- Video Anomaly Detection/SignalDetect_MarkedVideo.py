import os
import sys
import time

import cv2
import numpy as np
from tqdm import tqdm  # progress bar
import numpy.random as rd

import torch
import torch.nn as nn

GPU_id = 0  # sys.argv[0][-4]
Data_dir = '/mnt/sdb1/weit/datasets/UCF_Crimes'
Mark_road_anomalous = './MARK_road_anomalous__name_clip_beg_end_clip.txt'
Mark_road_normal = './MARK_road_normal__name.txt'
Mod_dir = 'VideoMarkDetect_{}'.format(GPU_id)
Inp_dim = 1024
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


def load_seqs_ano_mark_i():
    npy_dir = "{}/Videos_I3D_npy".format(Data_dir)
    '''
    1 second = 30 frames
    4 frames = 1 t
    1 second = 30/4 t
    '''
    k = 30 / 4

    marks_anomalous = list()
    with open(Mark_road_anomalous, 'r') as f:
        for line in f.readlines():
            l = line[:-1].split()
            name = l[0]
            l[1:] = [int(int(i) * k) if i != '-1' else int(i)
                     for i in l[1:]]
            marks_anomalous.append(l)

    seqs_ano_mark_i = list()
    for name, clip_i, i, j, clip_j in marks_anomalous:
        npy_path = '{}/{}.npy'.format(npy_dir, name)
        if os.path.exists(npy_path):
            i -= clip_i
            seqs_ano_mark_i.append(i)
    return seqs_ano_mark_i


class BatchSeqMarkVersion:
    def __init__(self, device):
        # mp4_dir = "{}/Videos".format(Data_dir)
        npy_dir = "{}/Videos_I3D_npy".format(Data_dir)
        rate_of_training_set = 0.8
        '''
        1 second = 30 frames
        4 frames = 1 t
        1 second = 30/4 t
        '''
        k = 30 / 4

        self.device = device

        '''load anomalous'''
        marks_anomalous = list()
        with open(Mark_road_anomalous, 'r') as f:
            for line in f.readlines():
                l = line[:-1].split()
                name = l[0]
                l[1:] = [int(int(i) * k) if i != '-1' else int(i)
                         for i in l[1:]]
                marks_anomalous.append(l)

        seqs_anomalous = list()
        for name, clip_i, i, j, clip_j in marks_anomalous:
            npy_path = '{}/{}.npy'.format(npy_dir, name)
            if os.path.exists(npy_path):
                ary = np.load(npy_path)
                ary_len = ary.shape[0]

                if clip_j == -1:
                    clip_j = ary_len
                if j == -1:
                    j = ary_len

                ary = ary[clip_i:clip_j]
                i -= clip_i
                j -= clip_i
                seqs_anomalous.append((ary, i, j))
        # self.seq1s = seqs_anomalous

        '''load normal'''
        seqs_normal = list()
        with open(Mark_road_normal, 'r') as f:
            for line in f.readlines():
                name = line[:-1]
                npy_path = '{}/{}.npy'.format(npy_dir, name)
                # print(npy_path)
                if os.path.exists(npy_path):
                    ary = np.load(npy_path)
                    seqs_normal.append(ary)
                    # print(ary.shape[0])
        # self.seq0s = seqs_normal

        training_num0 = int(rate_of_training_set * len(seqs_normal))
        self.seq0_train = seqs_normal[:training_num0]
        self.seq0_eval = seqs_normal[training_num0:]
        training_num1 = int(rate_of_training_set * len(seqs_anomalous))
        self.seq1_train = seqs_anomalous[:training_num1]
        self.seq1_eval = seqs_anomalous[training_num1:]

    @staticmethod
    def random_crop(ary):
        len0 = ary.shape[0]
        clip_i = rd.randint(int(len0 * 0.1))
        clip_j = -rd.randint(int(len0 * 0.1))
        return ary[clip_i:None if clip_j == 0 else clip_j]

    def get__inp_lab(self, plan_len, is_train=True, batch_size=16):
        plan_len -= plan_len % 16

        seq0 = self.seq0_train if is_train else self.seq0_eval
        seq1 = self.seq1_train if is_train else self.seq1_eval

        batch_seq = np.empty((plan_len, 2, Inp_dim), dtype=np.float32)
        batch_lab = np.zeros((plan_len, 2, 1), dtype=np.float32)
        '''fill normal seq'''
        total_len0 = 0

        # seq_num=0
        while total_len0 < plan_len:
            ary0 = self.random_crop(rd.choice(seq0))

            len0 = ary0.shape[0]
            add_len0 = min(len0, plan_len - total_len0)

            batch_seq[total_len0:total_len0 + add_len0, 0] = ary0[:add_len0]
            total_len0 += add_len0
            # seq_num += 1
        # print(seq_num)
        batch_lab[:, 0] = 0.00

        '''fill normal mix abnormal seq'''
        total_len = 0
        while total_len < plan_len:
            is_normal = bool(rd.rand() < 0.5)

            if is_normal:
                ary0 = rd.choice(seq0)

                ary_len = ary0.shape[0]
                add_len = min(ary_len, plan_len - total_len)

                batch_seq[total_len:total_len + add_len, 1] = ary0[:add_len]
            else:
                ary, mark_i, mark_j = seq1[rd.randint(len(seq1))]
                ary = ary[:int(ary.shape[0] * rd.uniform(0.9, 1))]

                ary_len = ary.shape[0]
                add_len = min(ary_len, plan_len - total_len)

                batch_seq[total_len:total_len + add_len, 1] = ary[:add_len]

                beg = min(total_len + mark_i, plan_len)
                end = min(total_len + mark_j, plan_len)
                batch_lab[beg:end, 1] = 6.00
                if beg + 3 <= plan_len:
                    batch_lab[beg:beg + 3, 1, 0] = [1., 3., 5.]
                if end + 3 <= plan_len:
                    # print(plan_len, end+3, batch_lab[end:end + 3, 1, 0].shape)
                    batch_lab[end:end + 3, 1, 0] = [5., 3., 1.]

            total_len += add_len

        batch_seq = torch.tensor(batch_seq.reshape((-1, batch_size, Inp_dim)), device=self.device)
        batch_lab = torch.tensor(batch_lab.reshape((-1, batch_size, Out_dim)), device=self.device)
        return batch_seq, batch_lab

    def run_test(self):
        print([len(l) for l in (self.seq0_train, self.seq0_eval,
                                self.seq1_train, self.seq1_eval,)])


def run_train():
    batch_len = int(2 ** 15)
    train_epoch = int(2 ** 5)
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

    bs = BatchSeqMarkVersion(device)

    '''training loop'''
    print('Training')
    start_time = show_time = time.time()
    loss_list = list()
    eva_loss_list = list()
    try:
        net.train()
        for epoch in range(train_epoch):
            batch_len0 = int(batch_len * 1.03 ** epoch)
            inp, lab = bs.get__inp_lab(batch_len0, is_train=True)
            if inp is None:
                continue

            out = net(inp)
            loss = criterion(out[8:], lab[8:])
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
        save_npz_path = '{}/eva_fap__normal.npz'.format(Mod_dir)
        if os.path.exists(save_npz_path):
            os.remove(save_npz_path)
        save_npz_path = '{}/eva_add__anomalous.npz'.format(Mod_dir)
        if os.path.exists(save_npz_path):
            os.remove(save_npz_path)

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
        bs = BatchSeqMarkVersion(device)

        '''eval loop'''
        ary_list = list()
        # print('len  Testing_Normal_Videos_Anomaly: 150', len(bs.eval_seq_walks[0][2]))
        for ary in tqdm(bs.seq0_eval):
            inp = torch.tensor(ary[:, np.newaxis], dtype=torch.float32, device=device)
            out = net.not_relu6(inp)

            ary = out.cpu().data.numpy()
            ary = ary.flatten()
            ary[:8] = 0.
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
    # eva_gap = 128
    eva_total_len = sum([ary.shape[0] for ary in ary_list])
    print("|Total Normal Frames:", eva_total_len)
    print('|Total Normal Videos:', len(ary_list))
    print('{:>6}    {:>6}    {:>7}    {:>7}'.format('h', 'AE', 'FPR', 'FAP', ))
    for h in h_list:
        total_events = len(ary_list)
        real_total_len = 0
        alarm_events = 0

        for ary in ary_list:
            eva_actions = np.where(ary >= h)[0]
            if eva_actions.shape[0] == 0:
                real_total_len += ary.shape[0]
            else:
                alarm_events += 1
                eva_t = eva_actions[0]
                real_total_len += eva_t

        false_alarm_period = real_total_len / alarm_events if alarm_events \
            else np.inf
        false_positive_rate = alarm_events / total_events
        print("{:6}    {:6}    {:7.3f}    {:7.3f}".format(
            h, alarm_events, false_positive_rate, false_alarm_period))


def run_eval_add(h_list):
    is_draw = False

    """init"""
    save_npy_path = '{}/eva_add__anomalous.npz'.format(Mod_dir)

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
        bs = BatchSeqMarkVersion(device)

        '''eval loop'''
        ary_list = list()
        # print('sum |train_abnormal_seq_len: 759', sum([len(walks[2]) for walks in bs.train_seq_walks[1:]]))
        # print('sum | eval_abnormal_seq_len: 191', sum([len(walks[2]) for walks in bs.eval_seq_walks[1:]]))
        for ary, mark_i, mark_j in tqdm(bs.seq1_eval):
            # print(ary.shape[0], file0)

            inp = torch.tensor(ary[:, np.newaxis], dtype=torch.float32, device=device)
            out = net.not_relu6(inp)

            ary = out.cpu().data.numpy()
            ary = ary.flatten()
            ary[:8] = 0.
            ary_list.append(ary)

        np.savez(save_npy_path, ary_list)

    if is_draw:
        ary_avg_len = int(np.mean([ary.shape[0] for ary in ary_list]))
        alpha = 10 / len(ary_list)
        import matplotlib.pyplot as plt
        for ary in ary_list:
            plt.plot(ary[:ary_avg_len], linewidth=4, alpha=alpha, color='k')

        plt.title('plot_anomalous.png')
        plt.savefig('{}/plot_abnormal.png'.format(Mod_dir))
        plt.show()
        exit()

    """evaluate Average Detection Delay(ADD)"""
    # eva_tau = 0
    eva_total_len = sum([ary.shape[0] for ary in ary_list])
    print("|Total Anomalous Frames:", eva_total_len)
    print('|Total Anomalous Videos:', len(ary_list))
    print('{:>6}    {:>6}    {:>7}    {:>7}'.format('h', 'AE', 'TPR', 'ADD', ))

    seqs_mark_i = load_seqs_ano_mark_i()
    for h in h_list:
        total_events = len(ary_list)
        alarm_events = 0
        c_det_delays = []

        for ary, mark_i in zip(ary_list, seqs_mark_i):
            eva_tau = max(mark_i - 8, 0)

            eva_actions = np.where(ary >= h)[0]
            if eva_actions.shape[0] == 0:
                continue
            else:
                alarm_events += 1

            eva_t = eva_actions[0]

            if eva_t < eva_tau:  # at this point, the online decision is 1 ("stop")
                # c_false_alarm_events += 1
                pass
            else:
                det_delay = eva_t - eva_tau
                c_det_delays.append(det_delay)

        avg_det_delay = np.mean(np.array(c_det_delays)) if len(c_det_delays) > 0 else np.nan
        true_positive_rate = alarm_events / total_events
        print("{:6}    {:6}    {:7.3f}    {:7.3f}".format(
            h, alarm_events, true_positive_rate, avg_det_delay))


if __name__ == '__main__':
    the_h_list = [
        .05, .1, .15, .2,
        .3, .35,
        .4, .45,
        .5, .55,
        .6, .7, .8, .85, .9, .95,
        # .97, .99, .998, 0.999
    ]
    # bs = BatchSeqMarkVersion(None)
    # bs.run_test()

    run_train()
    run_eval_fap(the_h_list)
    run_eval_add(the_h_list)
