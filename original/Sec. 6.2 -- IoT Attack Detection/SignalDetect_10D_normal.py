import os
import sys
import time

from tqdm import tqdm  # progress bar
import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

"""
Github: Yonv1943 DetectRNN_stable_2019_1213
signal change detector
dataset: IoT_botnet_attacks_N_BaIoT
dataset: 10-dim normal distribution
"""

GPU_id = 0  # sys.argv[0][-4]
Mod_dir = 'SignalDetect_{}'.format(GPU_id)
Inp_dim = 10  # 115
Mid_dim = 16  # 64
Out_dim = 1
Mid_layers = 2


class RegRNN(nn.Module):
    def __init__(self, input_size, out_dim, mid_dim, mid_layers):
        super(RegRNN, self).__init__()

        # self.rnn = nn.LSTM(input_size, mid_dim, mid_layers, dropout=0.25)  # rnn
        self.rnn = nn.GRU(input_size, mid_dim, mid_layers, dropout=0.25)  # rnn
        self.reg = nn.Sequential(
            nn.BatchNorm1d(mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, mid_dim),
            nn.BatchNorm1d(mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, out_dim),
            nn.Sigmoid(),
        )  # regression

    def forward(self, x):
        self.rnn.dropout = rd.uniform(0.125, 0.375)
        x = self.rnn(x)[0]

        seq_len, batch_size, hid_dim = x.shape
        x = x.view(-1, hid_dim)
        x = self.reg(x)

        x = x.view(seq_len, batch_size, -1)
        return x

    def output_y_h(self, x, h):
        y, h = self.rnn(x, h)

        seq_len, batch_size, hid_dim = y.size()
        y = y.view(-1, hid_dim)
        y = self.reg(y)
        y = y.view(seq_len, batch_size, -1)
        return y, h


def run_train():
    seq_len = int(2 ** 9 * 1.5)
    batch_size = 2 ** 9
    eval_size = 2 ** 9
    train_epoch = 2 ** 8  # 12
    show_gap = 2 ** 3

    '''build model'''
    os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_id)
    torch.set_default_dtype(torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = RegRNN(Inp_dim, Out_dim, Mid_dim, Mid_layers).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
    criterion = nn.MSELoss()

    '''data'''
    bs = BatchSeq10dNormal(device)

    '''training loop'''
    start_time = show_time = time.time()
    label = bs.get_label(seq_len, batch_size)
    eval_label = bs.get_label(seq_len, eval_size)
    loss = None
    try:
        net.train()
        for epoch in range(train_epoch):
            inp = bs.random_sample(seq_len, batch_size)

            out = net(inp)

            loss = criterion(out, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if time.time() > show_time + show_gap:
                show_time = time.time()
                net.eval()
                eval_inp = bs.random_sample(seq_len, eval_size, is_train=False)
                eval_out = net(eval_inp)
                eval_loss = criterion(eval_out, eval_label)

                print('Epoch: {:6}    EpoL: {:.2e}    EvaL: {:.2e}'.format(
                    epoch, loss.item(), eval_loss.item()))
                net.train()

    except KeyboardInterrupt:
        print("KeyboardInterrupt")
    except Exception as error:
        print("Error:", error)
    finally:
        net.eval()
        eval_inp = bs.random_sample(seq_len, eval_size, is_train=False)
        eval_out = net(eval_inp)
        eval_loss = criterion(eval_out, eval_label)

        print('Epoch: {:6}    EpoL: {:.2e}    EvaL: {:.2e}'.format(
            0, loss.item(), eval_loss.item()))

        os.makedirs(Mod_dir, exist_ok=True)
        torch.save(net.state_dict(), '%s/net.pth' % (Mod_dir,))
        print("Saved:", Mod_dir)

    print("Times Used:", int(time.time() - start_time))


def run_eval():
    h_list = (
        0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10,
        # 0.01, 0.05, 0.10,
        0.12, 0.14, 0.16, 0.18, 0.2,
        0.25, 0.3, 0.35, 0.4, 0.45,
        0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9,
        0.95, 0.99, 0.995, 0.996, 0.997, 0.998, 0.999,
        # 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95,
        # 0.97, 0.98, 0.985, 0.99,
        # 0.992, 0.993, 0.994, 0.995, 0.996,
        # 0.9965, 0.9967, 0.997,
    )

    """init"""

    '''build model'''
    os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = RegRNN(Inp_dim, Out_dim, Mid_dim, Mid_layers).to(device)
    net.eval()

    '''data'''
    bs = BatchSeq10dNormal(device)

    '''load model'''
    net.load_state_dict(torch.load('%s/net.pth' % (Mod_dir,), map_location=lambda storage, loc: storage))

    """evaluate Average Detection Delay(ADD) and False Alarm Period (FAP)"""
    eva_epoch = 2 ** 12  # 14
    eva_size = 2 ** 8
    eva_len = 2 ** 11
    '''
    00000000001111
    <------------>eva_len
              ^eva_tau, eva_tau in uniform(min, max)

    notice!
    SimpleHome_XCS7_1003, benign_traffic.csv 17.6MB, len==19528
    it is too short.
    '''

    from tqdm import trange
    """eval and draw in fixed eva_tau"""
    draw_size = 16
    eva_tau = eva_len - rd.randint(2 ** 8, 2 ** 10)
    eva_out_list = list()
    for i in trange(draw_size):
        eva_inp = bs.random_sample_eva(eva_len, draw_size, eva_tau)
        eva_out = net(eva_inp)
        eva_out = eva_out.cpu().data.numpy()
        eva_out = eva_out[:, :, 0]
        eva_out = np.transpose(eva_out, axes=(1, 0))
        eva_out_list.append(eva_out)

    eva_out_list = np.concatenate(eva_out_list, axis=0)
    draw_action_plot(eva_out_list[:draw_size], eva_tau)
    # exit()

    """eval in changing eva_tau"""
    eva_out_list = list()
    eva_tau_ary = rd.randint(2 ** 8, 2 ** 10, size=eva_epoch)  # rd in the loop

    for i in trange(eva_epoch):
        eva_tau = eva_tau_ary[i]  # rd in the loop

        eva_inp = bs.random_sample_eva(eva_len, 1, eva_tau)
        eva_out = net(eva_inp)
        eva_out = eva_out.cpu().data.numpy()
        eva_out = eva_out[:, :, 0]
        eva_out = np.transpose(eva_out, axes=(1, 0))
        eva_out_list.append(eva_out)

    eva_out_list = np.concatenate(eva_out_list, axis=0)
    # np.save('temp0.npy', eva_out_list)
    # np.save('temp1.npy', eva_tau_ary)
    # exit()
    # eva_out_list = np.load('temp0.npy', allow_pickle=True)
    # eva_tau_ary = np.load('temp1.npy', allow_pickle=True)

    eva_out_list[:, :8] = 0.0  # ignore init
    eva_out_list[:, -1] = 1.0  # force terminal

    for h in h_list:
        c_false_alarm_events = 0
        c_det_delays = []
        c_false_alarm_periods = []
        c_true_alarm_periods = []

        for i in range(eva_epoch):
            eva_out = eva_out_list[i]
            eva_tau = eva_tau_ary[i]

            eva_actions = np.where(eva_out >= h)[0]
            eva_t = eva_actions[0]

            if eva_t < eva_tau:  # at this point, the online decision is 1 ("stop")
                c_false_alarm_events += 1
                c_false_alarm_periods.append(eva_t)
            else:
                det_delay = eva_t - eva_tau
                c_det_delays.append(det_delay)
                c_true_alarm_periods.append(eva_tau)

        c_avg_det_delay = np.mean(c_det_delays) if len(c_det_delays) else 0
        false_alarm_period = sum(c_false_alarm_periods) + sum(c_true_alarm_periods)
        false_alarm_period = false_alarm_period / c_false_alarm_events if c_false_alarm_events else np.inf

        print("h {:6}      ADD  {:8.3f}  FAE  {:6}      FAP  {:8.3f}".format(
            h, c_avg_det_delay, c_false_alarm_events, false_alarm_period))

    # """evaluate False Alarm Period (FAP)"""
    # eva_out_list = list()
    # for _ in trange(2 ** 8):
    #     eva_inp = torch.randn(2 ** 15, 1, Inp_dim)
    #
    #
    # for h in h_list:
    #     c_false_alarm_events = 0
    #     # c_det_delays = []
    #     c_false_alarm_periods = []
    #     c_true_alarm_periods = []
    #     for i in range(eva_epoch):
    #         eva_out = eva_out_list[i]
    #         eva_tau = eva_tau_ary[i]


def run_test():
    seq_len = 512
    batch_size = 64
    train_epoch = 2 ** 10  # 12
    train_size = 2 ** 14

    '''build model'''
    from SignalDetectRNN import RegLSTM
    os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_id)
    torch.set_default_dtype(torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bs = BatchSeq10dNormal(device)
    # train_x = bs.random_sample(seq_len, batch_size)


def draw_action_plot(eva_outs, eva_tau):  # 2018-12-10
    plt.ion()

    fig, axs = plt.subplots(2)

    ax0 = axs[0]
    eva_range = 8
    for eva_out in eva_outs:
        eva_out_slice = eva_out[eva_tau - eva_range:eva_tau + eva_range]
        ax0.plot(eva_out_slice, color='royalblue', label='pred', alpha=0.3)
        ax0.plot([eva_range, eva_range], [0, 1], 'lightcoral', label='change point')
        ax0.set_facecolor('#f8f8f8')
        ax0.grid(color='white', linewidth=1.5)
        # ax0.legend(loc='best')

        ax1 = axs[1]
        ax1.plot(eva_out, color='darkcyan', label='pred', alpha=0.3)
        ax1.plot([eva_tau, eva_tau], [0, 1], color='lightcoral', label='change point')

    plt.savefig('{}/SignalDetectRNN.png'.format(Mod_dir))
    plt.pause(4)
    plt.close()


class BatchSeq10dNormal:
    def __init__(self, device):
        self.device = device

        # mu_ary = np.array([0.6, 0.8, 1.2, 0.9, 0.7, 0.5, 1.1, 0.3, 0.7, 1.1])  # arbitrary mean
        mu_ary = np.ones(Inp_dim)
        mu_ten = torch.tensor(mu_ary, dtype=torch.float32, device=self.device)
        # mu_ten = mu_ten.view((1, 1, Inp_dim))
        self.mu_ten = mu_ten

    def get_mask01(self, seq_len, batch_size):
        part_len = (seq_len - batch_size) // 2
        with torch.no_grad():
            label = torch.zeros((seq_len, batch_size, Out_dim), device=self.device)

            # grad_len = 3
            # grad = np.linspace(0.25, 1.0, grad_len).astype(np.float32)
            # grad = np.power(grad, 0.5)
            # grad = grad[:, np.newaxis]
            # grad = torch.tensor(grad, device=self.device)
            for i in range(batch_size):
                j = i + part_len
                label[j:, i] = 1
                # label[j:j + grad_len, i] = grad
        return label

    def get_label(self, seq_len, batch_size):
        part_len = (seq_len - batch_size) // 2
        with torch.no_grad():
            label = torch.zeros((seq_len, batch_size, Out_dim), device=self.device)

            grad_len = 3
            grad = np.linspace(0.25, 1.0, grad_len).astype(np.float32)
            grad = np.power(grad, 0.5)
            grad = grad[:, np.newaxis]
            grad = torch.tensor(grad, device=self.device)
            for i in range(batch_size):
                j = i + part_len
                label[j:, i] = 1
                label[j:j + grad_len, i] = grad
        return label

    def random_sample(self, seq_len, batch_size, is_train=True):
        part_len = (seq_len - batch_size) // 2
        '''
        left  mid  right
        00000 1111 11111
        00000 0111 11111
        00000 0011 11111
        00000 0001 11111
        <--->part_len  
               <-->batch_size
        <---------------->seq_len
        '''
        with torch.no_grad():
            batch_data = torch.randn((seq_len, batch_size, Inp_dim), device=self.device)
            for i in range(batch_size):
                left_len = part_len + i

                '''fill right batch with data1 (attack data)'''
                batch_data[left_len:, i] += self.mu_ten

        return batch_data

    def random_sample_eva(self, seq_len, batch_size, eva_tau):
        left_len = eva_tau  # benign_len
        """
        00000000000001111
        <--------------->   seq_len
        <----------->       left_len or benign
                     <-->   right_len or attack
                     ^      eva_tau
        """

        with torch.no_grad():
            batch_data = torch.randn((seq_len, batch_size, Inp_dim), device=self.device)
            for i in range(batch_size):

                '''fill right batch with data1 (attack data)'''
                batch_data[left_len:, i] += self.mu_ten

        return batch_data


if __name__ == '__main__':
    run_train()
    run_eval()
    # run_test()
