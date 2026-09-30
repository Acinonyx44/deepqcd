import os
import sys
import time

import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

"""
Github: Yonv1943 DetectRNN_stable_2019_1219
signal change detector
dataset: IoT_botnet_attacks_N_BaIoT
"""

GPU_id = sys.argv[0][-4]
Mod_dir = 'SignalDetect_{}'.format(GPU_id)
Inp_dim = 115
Mid_dim = 64
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
    train_epoch = 2 ** 9  # 12
    show_gap = 2 ** 3

    '''build model'''
    os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_id)
    torch.set_default_dtype(torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = RegRNN(Inp_dim, Out_dim, Mid_dim, Mid_layers).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
    criterion = nn.MSELoss()

    '''data'''
    bs = BatchSeq(device)

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
    bs = BatchSeq(device)

    '''load model'''
    net.load_state_dict(torch.load('%s/net.pth' % (Mod_dir,), map_location=lambda storage, loc: storage))

    """evaluate Average Detection Delay(ADD) and False Alarm Period (FAP)"""
    eva_epoch = 2 ** 12
    eva_size = 2 ** 8
    eva_len = 2 ** 10
    eva_tau = eva_len - rd.randint(2 ** 6, 2 ** 8)  # rd in the loop
    '''
    00000000001111
    <------------>eva_len
              ^eva_tau, eva_tau in uniform(min, max)

    notice!
    SimpleHome_XCS7_1003, benign_traffic.csv 17.6MB, len==19528
    it is too short.
    '''

    eva_out_list = list()
    for i in range(eva_epoch // eva_size):
        # eva_tau = eva_len - rd.randint(2 ** 6, 2 ** 8)

        eva_inp = bs.random_sample_eva(eva_len, eva_size, eva_tau)
        eva_out = net(eva_inp)
        eva_out = eva_out.cpu().data.numpy()
        eva_out = eva_out[:, :, 0]
        eva_out = np.transpose(eva_out, axes=(1, 0))

        eva_out_list.append(eva_out)
        # print(i)

    eva_out_list = np.concatenate(eva_out_list, axis=0)
    draw_action_plot(eva_out_list[:8], eva_tau)
    # np.save('temp.npy', eva_out_list)
    # eva_out_list = np.load('temp.npy', allow_pickle=True)
    print("shape:", eva_out_list.shape)

    eva_out_list[:, -1] = 1.0
    eva_out_list[:, :64] = 0

    for h in h_list:
        c_false_alarm_events = 0
        c_det_delays = []

        for n in range(eva_epoch):
            eva_out = eva_out_list[n]

            eva_actions = np.where(eva_out >= h)[0]
            eva_t = eva_actions[0]

            if eva_t < eva_tau:  # at this point, the online decision is 1 ("stop")
                c_false_alarm_events += 1
            else:
                det_delay = eva_t - eva_tau
                c_det_delays.append(det_delay)

        avg_det_delay = np.mean(np.array(c_det_delays)) if len(c_det_delays) > 0 else np.nan
        avg_false_alarm_rate = c_false_alarm_events / eva_epoch

        print("h {:6}      ADD  {:.3f}      FAR  {:.3f}".format(
            h, avg_det_delay, avg_false_alarm_rate, ))

    """evaluate False Alarm Period (FAP)"""
    eva_gap = 4
    reverse_eva_benign_data = bs.get_reverse_eva_benign_data()
    from torch.nn.utils.rnn import pad_sequence
    eva_pad_seq = pad_sequence(reverse_eva_benign_data)

    eva_out = net(eva_pad_seq)
    eva_out = eva_out.cpu().data.numpy()
    eva_out = eva_out[:, :, 0]
    eva_out = np.transpose(eva_out, axes=(1, 0))
    # print(eva_out.shape)

    eva_total_len = sum([ten.size(0) for ten in reverse_eva_benign_data])
    print("eva_total_len:", eva_total_len)
    for h in h_list:
        c_false_alarm_events = 0
        for iot_dev_id in range(len(eva_out)):
            iot_dev_seq_len = reverse_eva_benign_data[iot_dev_id].size(0)
            eva_seq = eva_out[iot_dev_id, :iot_dev_seq_len]

            eva_actions = np.where(eva_seq >= h)[0]

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

    bs = BatchSeq(device)
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


class BatchSeq:
    def __init__(self, device):
        data = DataSetIotBotNet()
        # data.print_informance()

        '''device'''
        self.device = device
        npz_save_data_paths = data.generate_npz()
        self.iot_train_list, self.iot_eval_list = self.get_train_eval_list(
            npz_save_data_paths, device)

        self.do_normalization()  # important

    @staticmethod
    def get_train_eval_list(npz_save_data_paths, device):
        iot_train_list = list()
        iot_eval_list = list()
        for npz_save_data_path in npz_save_data_paths:
            data_dict = np.load(npz_save_data_path)
            # for k, v in data_dict.items():
            #     print(k, v.shape)

            '''slice_train_set_and_eval_set'''
            rate_of_train_set = 0.5
            with torch.no_grad():
                train_dict = dict()
                eval_dict = dict()

                for k, v in data_dict.items():
                    data_len = v.shape[0]
                    i = int(data_len * rate_of_train_set)
                    train_dict[k] = torch.tensor(v[:i], dtype=torch.float32, device=device)
                    eval_dict[k] = torch.tensor(v[i:], dtype=torch.float32, device=device)

            train_list = [train_dict['benign_traffic.csv'], ]
            train_list += [v for k, v in train_dict.items() if k != 'benign_traffic.csv']
            iot_train_list.append(train_list)

            eval_list = [eval_dict['benign_traffic.csv'], ]
            eval_list += [v for k, v in eval_dict.items() if k != 'benign_traffic.csv']
            iot_eval_list.append(eval_list)

        # iot_train_list[idt_device_id, {0: benign_data, >=1: attack_data}]
        # iot_eval_list[idt_device_id, {0: benign_data, >=1: attack_data}]
        return iot_train_list, iot_eval_list

    @staticmethod
    def get_train_eval_list_in_diff_dev(npz_save_data_paths, device):
        iot_train_list = list()
        for npz_save_data_path in npz_save_data_paths[::2]:
            data_dict = np.load(npz_save_data_path)
            with torch.no_grad():
                train_dict = dict()

                for k, v in data_dict.items():
                    # data_len = v.shape[0]
                    # i = int(data_len * rate_of_train_set)
                    # train_dict[k] = torch.tensor(v[:i], dtype=torch.float32, device=device)
                    # eval_dict[k] = torch.tensor(v[i:], dtype=torch.float32, device=device)
                    train_dict[k] = torch.tensor(v, dtype=torch.float32, device=device)
            train_list = [train_dict['benign_traffic.csv'], ]
            train_list += [v for k, v in train_dict.items() if k != 'benign_traffic.csv']
            iot_train_list.append(train_list)

        iot_eval_list = list()
        for npz_save_data_path in npz_save_data_paths[1::2]:
            data_dict = np.load(npz_save_data_path)
            with torch.no_grad():
                eval_dict = dict()

                for k, v in data_dict.items():
                    # data_len = v.shape[0]
                    # i = int(data_len * rate_of_train_set)
                    # train_dict[k] = torch.tensor(v[:i], dtype=torch.float32, device=device)
                    # eval_dict[k] = torch.tensor(v[i:], dtype=torch.float32, device=device)
                    eval_dict[k] = torch.tensor(v, dtype=torch.float32, device=device)

            eval_list = [eval_dict['benign_traffic.csv'], ]
            eval_list += [v for k, v in eval_dict.items() if k != 'benign_traffic.csv']
            iot_eval_list.append(eval_list)

        return iot_train_list, iot_eval_list

    @staticmethod
    def get_train_eval_list_in_diff_attack(npz_save_data_paths, device):
        iot_train_list = list()
        iot_eval_list = list()
        for npz_save_data_path in npz_save_data_paths:
            data_dict = np.load(npz_save_data_path)
            # for k, v in data_dict.items():
            #     print(k, v.shape)

            '''slice_train_set_and_eval_set'''
            rate_of_train_set = 0.5
            with torch.no_grad():
                train_dict = dict()
                eval_dict = dict()

                # for k, v in data_dict.items():
                benign_key = 'benign_traffic.csv'
                v = data_dict[benign_key]
                data_len = v.shape[0]
                i = int(data_len * rate_of_train_set)
                train_dict[benign_key] = torch.tensor(v[:i], dtype=torch.float32, device=device)
                eval_dict[benign_key] = torch.tensor(v[i:], dtype=torch.float32, device=device)

            attack_keys = list(data_dict.keys() - {benign_key, })

            train_list = [train_dict['benign_traffic.csv'], ]
            # train_list += [v for k, v in train_dict.items() if k != 'benign_traffic.csv']
            train_list += [data_dict[k] for k in attack_keys[::2]]
            iot_train_list.append(train_list)

            eval_list = [eval_dict['benign_traffic.csv'], ]
            # eval_list += [v for k, v in eval_dict.items() if k != 'benign_traffic.csv']
            eval_list += [data_dict[k] for k in attack_keys[1::2]]
            iot_eval_list.append(eval_list)

        return iot_train_list, iot_eval_list

    def do_normalization(self):
        for i in range(len(self.iot_train_list)):
            train_list = self.iot_train_list[i]
            eval_list = self.iot_eval_list[i]

            benign_data = train_list[0]

            assert isinstance(benign_data, torch.Tensor)
            data_mean = benign_data.mean(dim=0, keepdim=True)
            data_std = benign_data.std(dim=0, keepdim=True)

            with torch.no_grad():
                for data_list in (train_list, eval_list):
                    for i in range(len(data_list)):
                        data = data_list[i]

                        data = (data - data_mean) / data_std
                        # data = torch.tanh(data / 6) * 6
                        # data = torch.tanh((data - data_mean) / (data_std * 6)) * 6

                        data_list[i] = data

        # for i in range(len(self.iot_train_list)):
        #     train_list = self.iot_train_list[i]
        #     benign_data = train_list[0]
        #     data_mean = benign_data.mean(dim=0, keepdim=True)
        #     data_std = benign_data.std(dim=0, keepdim=True)
        #     print(data_mean, data_std)

        # assert isinstance(benign_data, np.ndarray)
        # data_mean = benign_data.mean(axis=0, keepdims=True)
        # data_std = benign_data.std(axis=0, keepdims=True) + 1e-5
        # # data = (data - data_mean) / data_std
        # # data = np.tanh(data / 6) * 6
        # data = torch.tanh((data - data_mean) / (data_std * 6)) * 6

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
        # iot_device_id = rd.randint(len(self.iot_train_list))

        if is_train:
            iot_device_id = rd.randint(len(self.iot_train_list))
            train_list = self.iot_train_list[iot_device_id]

            data0 = train_list[0]  # benign_data
            data1_list = train_list[1:]  # attack_data
        else:
            iot_device_id = rd.randint(len(self.iot_eval_list))
            eval_list = self.iot_eval_list[iot_device_id]

            data0 = eval_list[0]  # benign_data
            data1_list = eval_list[1:]  # attack_data

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
        part_len = (seq_len - batch_size) // 2
        data0_len = data0.shape[0]

        data1 = rd.choice(data1_list, 1)[0]
        # data1 = data1_list[rd.randint(len(data1_list))]
        data1_len = data1.shape[0]

        with torch.no_grad():
            batch_data = torch.empty((seq_len, batch_size, Inp_dim), device=self.device)
            for i in range(batch_size):
                left_len = part_len + i
                '''fill left batch with data0 (benign data)'''
                j = rd.randint(0, data0_len - left_len)
                batch_data[:left_len, i] = data0[j:j + left_len]

                '''fill right batch with data1 (attack data)'''
                right_len = seq_len - left_len
                # data1 = rd.choice(data1_list, 1)[0]
                # data1_len = data1.shape[0]
                k = rd.randint(0, data1_len - right_len)
                batch_data[left_len:, i] = data1[k:k + right_len]

        return batch_data

    def random_sample_eva(self, seq_len, batch_size, eva_tau):
        iot_device_id = rd.randint(len(self.iot_train_list))
        eval_list = self.iot_eval_list[iot_device_id]

        data0 = eval_list[0]  # benign_data
        data1_list = eval_list[1:]  # attack_data

        """
        00000000000001111
        <--------------->   seq_len
        <----------->       left_len or benign
                     <-->   right_len or attack
                     ^      eva_tau
        """

        left_len = eva_tau  # benign_len
        right_len = seq_len - eva_tau  # attack_len

        data0_len = data0.shape[0]

        with torch.no_grad():
            batch_data = torch.empty((seq_len, batch_size, Inp_dim), device=self.device)
            for i in range(batch_size):
                '''fill left batch with data0 (benign data)'''
                j = rd.randint(0, data0_len - left_len)
                batch_data[:left_len, i] = data0[j:j + left_len]

                '''fill right batch with data1 (attack data)'''
                data1 = rd.choice(data1_list, 1)[0]
                data1_len = data1.shape[0]
                k = rd.randint(0, data1_len - right_len)
                batch_data[left_len:, i] = data1[k:k + right_len]

        return batch_data

    def get_reverse_eva_benign_data(self):
        data_list = [eval_list[0] for eval_list in self.iot_eval_list]
        data_lens = [data.size(0) for data in data_list]

        data_ary = np.array(data_list)
        data_ary = data_ary[np.argsort(data_lens)[::-1]]  # reverse
        return data_ary


class BatchSeqAllDev:
    def __init__(self, device):
        data = DataSetIotBotNet()
        # data.print_informance()

        '''device'''
        self.device = device
        npz_save_data_paths = data.generate_npz()
        self.iot_train_list, self.iot_eval_list = self.get_train_eval_list(
            npz_save_data_paths, device)

        self.do_normalization()  # important

    @staticmethod
    def get_train_eval_list(npz_save_data_paths, device):
        iot_train_list = list()
        iot_eval_list = list()
        for npz_save_data_path in npz_save_data_paths:
            data_dict = np.load(npz_save_data_path)
            # for k, v in data_dict.items():
            #     print(k, v.shape)

            '''slice_train_set_and_eval_set'''
            rate_of_train_set = 0.5
            with torch.no_grad():
                train_dict = dict()
                eval_dict = dict()

                for k, v in data_dict.items():
                    data_len = v.shape[0]
                    i = int(data_len * rate_of_train_set)
                    train_dict[k] = torch.tensor(v[:i], dtype=torch.float32, device=device)
                    eval_dict[k] = torch.tensor(v[i:], dtype=torch.float32, device=device)

            train_list = [train_dict['benign_traffic.csv'], ]
            train_list += [v for k, v in train_dict.items() if k != 'benign_traffic.csv']
            iot_train_list.append(train_list)

            eval_list = [eval_dict['benign_traffic.csv'], ]
            eval_list += [v for k, v in eval_dict.items() if k != 'benign_traffic.csv']
            iot_eval_list.append(eval_list)

        # iot_train_list[idt_device_id, {0: benign_data, >=1: attack_data}]
        # iot_eval_list[idt_device_id, {0: benign_data, >=1: attack_data}]
        return iot_train_list, iot_eval_list

    @staticmethod
    def get_train_eval_list_in_diff_dev(npz_save_data_paths, device):
        iot_train_list = list()
        for npz_save_data_path in npz_save_data_paths[::2]:
            data_dict = np.load(npz_save_data_path)
            with torch.no_grad():
                train_dict = dict()

                for k, v in data_dict.items():
                    # data_len = v.shape[0]
                    # i = int(data_len * rate_of_train_set)
                    # train_dict[k] = torch.tensor(v[:i], dtype=torch.float32, device=device)
                    # eval_dict[k] = torch.tensor(v[i:], dtype=torch.float32, device=device)
                    train_dict[k] = torch.tensor(v, dtype=torch.float32, device=device)
            train_list = [train_dict['benign_traffic.csv'], ]
            train_list += [v for k, v in train_dict.items() if k != 'benign_traffic.csv']
            iot_train_list.append(train_list)

        iot_eval_list = list()
        for npz_save_data_path in npz_save_data_paths[1::2]:
            data_dict = np.load(npz_save_data_path)
            with torch.no_grad():
                eval_dict = dict()

                for k, v in data_dict.items():
                    # data_len = v.shape[0]
                    # i = int(data_len * rate_of_train_set)
                    # train_dict[k] = torch.tensor(v[:i], dtype=torch.float32, device=device)
                    # eval_dict[k] = torch.tensor(v[i:], dtype=torch.float32, device=device)
                    eval_dict[k] = torch.tensor(v, dtype=torch.float32, device=device)

            eval_list = [eval_dict['benign_traffic.csv'], ]
            eval_list += [v for k, v in eval_dict.items() if k != 'benign_traffic.csv']
            iot_eval_list.append(eval_list)

        return iot_train_list, iot_eval_list

    @staticmethod
    def get_train_eval_list_in_diff_attack(npz_save_data_paths, device):
        iot_train_list = list()
        iot_eval_list = list()
        for npz_save_data_path in npz_save_data_paths:
            data_dict = np.load(npz_save_data_path)
            # for k, v in data_dict.items():
            #     print(k, v.shape)

            '''slice_train_set_and_eval_set'''
            rate_of_train_set = 0.5
            with torch.no_grad():
                train_dict = dict()
                eval_dict = dict()

                # for k, v in data_dict.items():
                benign_key = 'benign_traffic.csv'
                v = data_dict[benign_key]
                data_len = v.shape[0]
                i = int(data_len * rate_of_train_set)
                train_dict[benign_key] = torch.tensor(v[:i], dtype=torch.float32, device=device)
                eval_dict[benign_key] = torch.tensor(v[i:], dtype=torch.float32, device=device)

            attack_keys = list(data_dict.keys() - {benign_key, })

            train_list = [train_dict['benign_traffic.csv'], ]
            # train_list += [v for k, v in train_dict.items() if k != 'benign_traffic.csv']
            train_list += [data_dict[k] for k in attack_keys[::2]]
            iot_train_list.append(train_list)

            eval_list = [eval_dict['benign_traffic.csv'], ]
            # eval_list += [v for k, v in eval_dict.items() if k != 'benign_traffic.csv']
            eval_list += [data_dict[k] for k in attack_keys[1::2]]
            iot_eval_list.append(eval_list)

        return iot_train_list, iot_eval_list

    def do_normalization(self):
        for i in range(len(self.iot_train_list)):
            train_list = self.iot_train_list[i]
            eval_list = self.iot_eval_list[i]

            benign_data = train_list[0]

            assert isinstance(benign_data, torch.Tensor)
            data_mean = benign_data.mean(dim=0, keepdim=True)
            data_std = benign_data.std(dim=0, keepdim=True)

            with torch.no_grad():
                for data_list in (train_list, eval_list):
                    for i in range(len(data_list)):
                        data = data_list[i]

                        data = (data - data_mean) / data_std
                        # data = torch.tanh(data / 6) * 6
                        # data = torch.tanh((data - data_mean) / (data_std * 6)) * 6

                        data_list[i] = data

        # for i in range(len(self.iot_train_list)):
        #     train_list = self.iot_train_list[i]
        #     benign_data = train_list[0]
        #     data_mean = benign_data.mean(dim=0, keepdim=True)
        #     data_std = benign_data.std(dim=0, keepdim=True)
        #     print(data_mean, data_std)

        # assert isinstance(benign_data, np.ndarray)
        # data_mean = benign_data.mean(axis=0, keepdims=True)
        # data_std = benign_data.std(axis=0, keepdims=True) + 1e-5
        # # data = (data - data_mean) / data_std
        # # data = np.tanh(data / 6) * 6
        # data = torch.tanh((data - data_mean) / (data_std * 6)) * 6

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
        iot_device_id = rd.randint(len(self.iot_train_list))

        if is_train:
            train_list = self.iot_train_list[iot_device_id]

            data0 = train_list[0]  # benign_data
            data1_list = train_list[1:]  # attack_data
        else:
            eval_list = self.iot_eval_list[iot_device_id]

            data0 = eval_list[0]  # benign_data
            data1_list = eval_list[1:]  # attack_data

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
        part_len = (seq_len - batch_size) // 2
        data0_len = data0.shape[0]

        data1 = rd.choice(data1_list, 1)[0]
        # data1 = data1_list[rd.randint(len(data1_list))]
        data1_len = data1.shape[0]

        with torch.no_grad():
            batch_data = torch.empty((seq_len, batch_size, Inp_dim), device=self.device)
            for i in range(batch_size):
                left_len = part_len + i
                '''fill left batch with data0 (benign data)'''
                j = rd.randint(0, data0_len - left_len)
                batch_data[:left_len, i] = data0[j:j + left_len]

                '''fill right batch with data1 (attack data)'''
                right_len = seq_len - left_len
                # data1 = rd.choice(data1_list, 1)[0]
                # data1_len = data1.shape[0]
                k = rd.randint(0, data1_len - right_len)
                batch_data[left_len:, i] = data1[k:k + right_len]

        return batch_data

    def random_sample_eva(self, seq_len, batch_size, eva_tau):
        iot_device_id = rd.randint(len(self.iot_train_list))
        eval_list = self.iot_eval_list[iot_device_id]

        data0 = eval_list[0]  # benign_data
        data1_list = eval_list[1:]  # attack_data

        """
        00000000000001111
        <--------------->   seq_len
        <----------->       left_len or benign
                     <-->   right_len or attack
                     ^      eva_tau
        """

        left_len = eva_tau  # benign_len
        right_len = seq_len - eva_tau  # attack_len

        data0_len = data0.shape[0]

        with torch.no_grad():
            batch_data = torch.empty((seq_len, batch_size, Inp_dim), device=self.device)
            for i in range(batch_size):
                '''fill left batch with data0 (benign data)'''
                j = rd.randint(0, data0_len - left_len)
                batch_data[:left_len, i] = data0[j:j + left_len]

                '''fill right batch with data1 (attack data)'''
                data1 = rd.choice(data1_list, 1)[0]
                data1_len = data1.shape[0]
                k = rd.randint(0, data1_len - right_len)
                batch_data[left_len:, i] = data1[k:k + right_len]

        return batch_data

    def get_reverse_eva_benign_data(self):
        data_list = [eval_list[0] for eval_list in self.iot_eval_list]
        data_lens = [data.size(0) for data in data_list]

        data_ary = np.array(data_list)
        data_ary = data_ary[np.argsort(data_lens)[::-1]]  # reverse
        return data_ary


class DataSetIotBotNet:  # todo rename DataIotBotnet
    def __init__(self):
        self.root_dir = './IoT_botnet_attacks_N_BaIoT'
        self.child_dirs = [
            'Danmini_Doorbell',
            'SimpleHome_XCS7_1002_WHT_Security_Camera',
            'SimpleHome_XCS7_1003_WHT_Security_Camera',
            'Provision_PT_737E_Security_Camera',
            'Provision_PT_838_Security_Camera',
            'Samsung_SNH_1011_N_Webcam',
        ]
        '''
        SimpleHome_XCS7_1003_WHT_Security_Camera
        benign_traffic.csv 17.6MB, len==19528
        it is too short.
        '''

        self.child_dir_paths = ["{}/{}".format(self.root_dir, name)
                                for name in self.child_dirs]

        self.gafgyt_attacks = ['combo.csv', 'scan.csv', 'tcp.csv', 'junk.csv', 'udp.csv']
        self.mirai_attacks = ['scan.csv', 'syn.csv', 'ack.csv', 'udpplain.csv', 'udp.csv']
        # self.gafgyt_attacks = ['junk.csv', ]
        # self.mirai_attacks = []

        self.demo_struc = self.read_csv_to_list(
            "{}/demonstrate_structure.csv".format(self.root_dir))[0]

    @staticmethod
    def read_csv_to_list(path):
        with open(path, 'r') as csv_file:
            data = list()
            for l in csv_file.readlines():
                data.append(l[:-1].split(','))
        return data

    def print_information(self):  # todo information
        informance_dict = {
            'root_dir': self.root_dir,
            'len(demo_struc)': len(self.demo_struc),
            'len(child_dirs)': len(self.child_dirs),
            'child_dirs': self.child_dirs,
            'gafgyt_attacks': self.gafgyt_attacks,
            'mirai_attacks': self.mirai_attacks,
            'dataset name': "detection_of_IoT_botnet_attacks_N_BaIoT Data Set",
            'download link': "https://archive.ics.uci.edu/ml/datasets/"
                             "detection_of_IoT_botnet_attacks_N_BaIoT",
        }

        for k, v in informance_dict.items():
            print("{}:\t{}".format(k, v))
        print('\n')

    def search_name_in_child_dirs(self, name):
        results = [child_dir
                   for child_dir in self.child_dirs
                   if child_dir.find(name) >= 0]
        len_results = len(results)
        if len_results == 0:
            print("not found name:", name)
            print('root_dir:', self.root_dir)
            raise FileNotFoundError
        elif len_results == 1:
            child_dir_name = results[0]
            child_dir_path = "{}/{}".format(self.root_dir, child_dir_name)
            return child_dir_path

    def read_child_dir(self, child_dir, save_it=False):

        data_dir = self.search_name_in_child_dirs(child_dir)
        print('data_dir:', data_dir)
        save_path = '{}/data.npz'.format(data_dir, )
        save_dict = dict()

        if os.path.exists(save_path):
            save_dict = np.load(save_path, allow_pickle=True)
            save_dict = dict(save_dict)
            # for k, v in save_dict.items():
            #     print(k, v.shape)
        else:
            start_time = time.time()
            """read_benign_traffic_csv"""
            key_name = 'benign_traffic.csv'
            benign_data = "{}/{}".format(data_dir, key_name)
            benign_data = self.read_csv_to_list(benign_data)
            benign_data = np.array(benign_data[1:], dtype=np.float32)

            save_dict[key_name] = benign_data
            print("key_name: {}\tshape: {}".format(key_name, benign_data.shape))

            """read_attacks data"""
            for attack_data_dir in ('gafgyt_attacks', 'mirai_attacks'):
                attack_paths = "{}/{}".format(data_dir, attack_data_dir)
                attack_data_names = os.listdir(attack_paths) if os.path.isdir(attack_paths) \
                    else list()

                for name in attack_data_names:
                    path = '{}/{}'.format(attack_paths, name)

                    attack_data = self.read_csv_to_list(path)
                    attack_data = np.array(attack_data[1:], dtype=np.float32)
                    key_name = "{}__{}".format(attack_data_dir, name)

                    save_dict[key_name] = attack_data
                    print("key_name: {}\tshape: {}".format(key_name, attack_data.shape))

            print("TimeUsed:", int(time.time() - start_time))

            if save_it:
                np.savez_compressed(save_path, **save_dict)
                print("Saved:", save_path)
        return save_dict, save_path

    def generate_npz(self):
        save_paths = list()
        for child_dir in self.child_dirs:
            save_dict, save_path = self.read_child_dir(child_dir, save_it=True)
            save_paths.append(save_path)
        return save_paths

    # @staticmethod
    # def run_test():
    #     data = DatasetBotnet()
    #     data.print_informance()
    #
    #     # data_dict = data.read_child_dir('Danmini')
    #
    #     # benign_data, attack_data_dict = data.read_child_dir('Provision_PT_737')
    #     # benign_data, attack_data_dict = data.read_child_dir('SimpleHome_XCS7_1002')
    #     # benign_data, attack_data_dict = data.read_child_dir('SimpleHome_XCS7_1003')
    #     #
    #     # np.save('benign_data.npy', benign_data[:12345])
    #     # np.save('attack_data.npy', attack_data_dict['scan.csv'][:12345])
    #
    #     benign_data = np.load('benign_data.npy')
    #     attack_data = np.load('attack_data.npy')
    #
    #     i_beg, i_end = 0, 12345
    #     j_beg, j_end = 64, 96  # 0, 115
    #     benign_data = benign_data[i_beg:i_end, j_beg:j_end]
    #     attack_data = attack_data[i_beg:i_end, j_beg:j_end]
    #     print(benign_data.shape, attack_data.shape)
    #
    #     # normalization
    #     data_mean = benign_data.mean(axis=0, keepdims=True)
    #     data_std = benign_data.std(axis=0, keepdims=True) + 1e-5
    #     benign_data = (benign_data - data_mean) / data_std
    #     attack_data = (attack_data - data_mean) / data_std
    #
    #     # benign_data=benign_data.clip(-6, +6)
    #     # attack_data=attack_data.clip(-6, +6)
    #     benign_data = np.tanh(benign_data / 6) * 6
    #     attack_data = np.tanh(attack_data / 6) * 6
    #
    #     # dislocation
    #     dis_range = j_end - j_beg
    #     dis_array = np.arange(dis_range) * (12 / dis_range)
    #     dis_array = dis_array[np.newaxis, :]
    #     benign_data = benign_data + dis_array
    #     attack_data = attack_data + dis_array
    #
    #     # plt.ion()
    #     plt.plot(benign_data, label='benign', color='royalblue', alpha=0.4)
    #     plt.plot(attack_data, label='attack', color='coral', alpha=0.4)
    #     # plt.legend('best')
    #     # plt.pause(4)
    #     plt.show()


if __name__ == '__main__':
    # run_train()
    run_eval()
    # run_test()
