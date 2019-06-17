from auto_encoder import AutoEncoder
import researchutils.chainer.links as rulinks
import chainer.functions as F
import chainer.links as L


class LstmPredictionNetwork(AutoEncoder):
    def __init__(self, color_num=3, frame_num=1, action_num=3):
        super(LstmPredictionNetwork, self).__init__()
        self.factor_num = 2048
        with self.init_scope():
            # 210x160 -> 102x78
            self.conv1 = L.Convolution2D(
                in_channels=color_num * frame_num, out_channels=64, ksize=8, stride=2, pad=(0, 1))
            # 102x78 -> 50x38
            self.conv2 = L.Convolution2D(
                in_channels=64, out_channels=128, ksize=6, stride=2, pad=(1, 1))
            # 50x38 -> 24x18
            self.conv3 = L.Convolution2D(
                in_channels=128, out_channels=128, ksize=6, stride=2, pad=(1, 1))
            # 24x18 -> 11x8
            self.conv4 = L.Convolution2D(
                in_channels=128, out_channels=128, ksize=4, stride=2)
            self.lstm = rulinks.GradClipLSTM(in_size=None, out_size=self.factor_num,
                                             clip_min=-0.1, clip_max=0.1)
            self.W_enc = L.Linear(
                in_size=self.factor_num, out_size=self.factor_num, nobias=True)
            self.W_a = L.Linear(in_size=action_num,
                                out_size=self.factor_num, nobias=True)
            self.W_dec = L.Linear(in_size=self.factor_num, out_size=128*11*8)
            self.deconv4 = L.Deconvolution2D(
                in_channels=128, out_channels=128, ksize=4, stride=2)
            self.deconv3 = L.Deconvolution2D(
                in_channels=128, out_channels=128, ksize=6, stride=2, pad=(1, 1))
            self.deconv2 = L.Deconvolution2D(
                in_channels=128, out_channels=64, ksize=6, stride=2, pad=(1, 1))
            self.deconv1 = L.Deconvolution2D(
                in_channels=64, out_channels=3, ksize=8, stride=2, pad=(0, 1))

    def encode(self, x):
        frames, actions = x
        h_enc = F.relu(self.conv1(frames))
        h_enc = F.relu(self.conv2(h_enc))
        h_enc = F.relu(self.conv3(h_enc))
        h_enc = F.relu(self.conv4(h_enc))
        h_enc = F.relu(self.lstm(h_enc))

        h_dec = self.W_dec(self.W_enc(h_enc) * self.W_a(actions))
        return h_dec

    def decode(self, h_dec):
        x_hat = F.reshape(h_dec, shape=(-1, 128, 11, 8))
        x_hat = F.relu(self.deconv4(x_hat))
        x_hat = F.relu(self.deconv3(x_hat))
        x_hat = F.relu(self.deconv2(x_hat))
        x_hat = self.deconv1(x_hat)

        return x_hat

    def reset_state(self):
        self.lstm.reset_state()

    def loss(self, x, y):
        pass
