import os
import sys
import time

import numpy as np
import numpy.random as rd
# import matplotlib.pyplot as plt

import torch
import torch.nn as nn

"""
Github: Yonv1943 DetectRNN_stable_2019_1219
signal change detector
dataset: IoT_botnet_attacks_N_BaIoT
"""

GPU_id = sys.argv[0][-4]
Mod_dir = 'SignalDetect_{}'.format(GPU_id)
Inp_dim = 6
Mid_dim = 64
Out_dim = 1
Mid_layers = 2


def run_train():
    seq_len = int(2 ** 8)
    batch_size = 2 ** 3
    train_epoch = int(2 ** 7)
    show_gap = 2 ** -1

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
    eval_inp, eval_lab = bs.get__eva__inp_lab()
    loss = None
    try:
        net.train()
        for epoch in range(train_epoch):
            seq_len0 = int(seq_len * 1.01 ** epoch)
            batch_size0 = int(batch_size * 1.01 ** epoch)
            inp, lab = bs.get__inp_lab(seq_len0, batch_size0)

            out = net(inp)
            loss = criterion(out, lab)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if time.time() > show_time + show_gap:
                show_time = time.time()
                net.eval()
                eval_out = net(eval_inp)
                eval_loss = criterion(eval_out, eval_lab)

                print('Epoch: {:6}    EpoL: {:.2e}    EvaL: {:.2e}'.format(
                    epoch, loss.item(), eval_loss.item()))
                print(seq_len0, batch_size0)
                net.train()

    except KeyboardInterrupt:
        print("KeyboardInterrupt")
    # except Exception as error:
    #     print("Error:", error)
    # finally:
    net.eval()

    eval_out = net(eval_inp)
    eval_loss = criterion(eval_out, eval_lab)

    print('Epoch: {:6}    EpoL: {:.2e}    EvaL: {:.2e}'.format(
        0, loss.item(), eval_loss.item()))

    os.makedirs(Mod_dir, exist_ok=True)
    torch.save(net.state_dict(), '%s/net.pth' % (Mod_dir,))
    print("Saved:", Mod_dir)

    print("Times Used:", int(time.time() - start_time))


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


class DataSetOccupancy:
    def __init__(self):
        self.root_dir = '/mnt/sdb1/yonv/datasets/occupancy_data'
        # self.root_dir = './occupancy_data'
        self.data_files = [
            'datatest.txt',  # 2665
            'datatest2.txt',  # 9752
            'datatraining.txt',  # 8143
        ]

        self.data_items = ['index', 'date', 'Temperature', 'Humidity',
                           'Light', 'CO2', 'HumidityRatio', 'Occupancy']
        self.npy_items = ['date_month', 'date_hour', 'Temperature', 'Humidity',
                          'Light', 'CO2', 'HumidityRatio', 'Occupancy']

    @staticmethod
    def read_txt_to_list(path):
        with open(path, 'r') as csv_file:
            data = list()
            for line in csv_file.readlines():
                data.append(line[:-1].split(','))
        return data

    def print_information(self):
        information_dict = {
            'root_dir': self.root_dir,
            'dataset name': "ics.uci.edu Occupancy Detection",
            'download link': "https://github.com/LuisM78/Occupancy-detection-data",
        }

        for k, v in information_dict.items():
            print("{}:\t{}".format(k, v))
        print('\n')

    def load_build__npy(self):
        name_data__dict = dict()
        for file in self.data_files:
            load_path = "{}/{}".format(self.root_dir, file)
            save_path = load_path.replace('.txt', '.npy')

            if  os.path.isfile(save_path):
                data = np.load(save_path)
            else:
                data = self.read_txt_to_list(load_path)

                data = data[1:]
                data = [[s.replace('\"', '') for s in line[1:]] for line in data]

                data = np.array(data)
                date_str = data[:, 0]
                date_num = list()
                for time_str in date_str:
                    month_num__hour_num = self.time_to_floats(time_str)
                    date_num.append(month_num__hour_num)
                date_num = np.array(date_num)

                # print(date_num.shape, data.shape)
                data1 = np.hstack((date_num[:, 1:2], data[:, 1:]))
                data1 = data1.astype(np.float32)
                # print(data1.shape)

                np.save(save_path, data1)
                print('Seq_shape: {}. Save in {}'.format(data1.shape, save_path))

            name = save_path[:-4].split('/')[-1]
            name_data__dict[name]=data
        return name_data__dict

    @staticmethod
    def time_to_floats(time_str):
        # time_str = "2019-04-13 10:02:23"

        dt = time_str
        time_ary = time.strptime(dt, "%Y-%m-%d %H:%M:%S")
        # time_int = int(time.mktime(time_ary))

        month_num = (time_ary.tm_mon - 1) + (time_ary.tm_mday - 1) / 31
        hour_num = time_ary.tm_hour + time_ary.tm_min / 60
        return month_num, hour_num


class BatchSeq:
    def __init__(self, device):
        data = DataSetOccupancy()
        # data.print_informance()

        '''device'''
        self.device = device

        train_list = list()
        eval_list = list()
        name_data__dict = data.load_build__npy()
        for name, data in name_data__dict.items():
            data = data[:, np.newaxis, :]
            data = torch.tensor(data, dtype=torch.float32, device=device)
            if name in {'datatraining'}:
                train_list.append(data)
            elif name in {'datatest', 'datatest2'}:
                eval_list.append(data)
            else:
                raise FileNotFoundError
        self.train_list, self.eval_list = train_list, eval_list

        self.do_normalization()  # important

    def do_normalization(self):
        data = self.train_list[0]
        print(';', data.size())
        data = data[:, :, :-1]
        print(';;', data.size())
        data_mean = data.mean(dim=0, keepdim=True)
        data_std = data.std(dim=0, keepdim=True) + 1e-7

        for i, data in enumerate(self.train_list):
            data = (data - data_mean) / data_std
            self.train_list[i][:, :, :-1] = data
        for i, data in enumerate(self.eval_list):
            data = (data - data_mean) / data_std
            self.eval_list[i][:, :, :-1] = data

    def get__inp_lab(self, seq_len, batch_size):
        data = self.train_list[0]  # 'datatraining.txt',  # 8143
        max_len = data.size(0)
        seq_len = min(seq_len, max_len - batch_size * 2)

        inp_lab = list()
        for i in rd.randint(0, max_len - seq_len + 1, batch_size):
            inp_lab.append(data[i:i + seq_len])
        inp_lab = torch.cat(inp_lab, dim=1)

        inp = inp_lab[:, :, :-1]
        lab = inp_lab[:, :, -1:]
        # lab = self.blur_tensor(lab)
        return inp, lab

    @staticmethod
    def blur_tensor(ten, size=2):
        for i in range(1, size + 1):
            ten[i:] += ten[:-i]
        ten[size:] /= size
        for i in range(1, size):
            ten[i] /= (i + 1)
        return ten

    def get__eva__inp_lab(self):
        # 'datatest.txt',  # 2665
        # 'datatest2.txt',  # 9752
        seq_len = 2438
        inp_lab = [
            self.eval_list[0][:seq_len],
            self.eval_list[1][seq_len * 1:seq_len * 2],
            self.eval_list[1][seq_len * 2:seq_len * 3],
            self.eval_list[1][seq_len * 3:seq_len * 4],
        ]
        inp_lab = torch.cat(inp_lab, dim=1)

        print(';', inp_lab.size())
        inp = inp_lab[:, :, :-1]
        lab = inp_lab[:, :, -1:]
        # lab = self.blur_tensor(lab)
        return inp, lab


def run_test():
    data = DataSetOccupancy()


if __name__ == '__main__':
    # run_test()
    run_train()
