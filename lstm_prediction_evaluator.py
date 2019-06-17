from researchutils.chainer.functions import average_k_step_squared_error

from chainer.link import Chain
from chainer import reporter
import chainer.functions as F
import chainer

class LstmPredictionEvaluator(Chain):
    def __init__(self, predictor, k_step, init_steps, unroll_steps, loss_fun=average_k_step_squared_error):
        super(LstmPredictionEvaluator, self).__init__()
        self.loss_fun = loss_fun
        self.k_step = k_step
        self.init_steps = init_steps
        self.unroll_steps = unroll_steps
        self.loss = None
        with self.init_scope():
            self.predictor = predictor

    def __call__(self, *args, **kwargs):
        (train_frame, train_actions) = kwargs['input']
        target_frames = kwargs['target']
        if self.k_step == 1:
            # print('train_frame len:{}, target_frames len:{}'.format(len(train_frame[0]), len(target_frames[0])))
            assert len(train_frame[0]) == self.unroll_steps
            assert len(
                target_frames[0]) == self.unroll_steps - self.init_steps + 1
        else:
            assert len(train_frame[0]) == self.init_steps
            assert len(target_frames[0]) == self.k_step
        assert len(train_actions[0]) == self.unroll_steps

        # initialize lstm network
        with chainer.no_backprop_mode():
            for step in range(self.init_steps - 1):
                self.predictor(
                    (train_frame[:, step, :, :, :], train_actions[:, step, :, :]))

        self.loss = 0
        if self.k_step == 1:
            training_steps = self.unroll_steps - self.init_steps + 1
            for step in range(training_steps):
                input_frame = train_frame[:, step +
                                          (self.init_steps - 1), :, :, :]
                input_action = train_actions[:,
                                             step + (self.init_steps - 1), :, :]
                predicted_frame = self.predictor((input_frame, input_action))
                expected_frame = target_frames[:, step, :, :, :]
                self.loss += self.loss_fun(predicted_frame,
                                           expected_frame,
                                           self.k_step)
        else:
            input_frame = train_frame[:, self.init_steps - 1, :, :, :]
            for step in range(self.k_step):
                input_action = train_actions[:,
                                             step + (self.init_steps - 1), :, :]
                predicted_frame = self.predictor((input_frame, input_action))
                expected_frame = target_frames[:, step, :, :, :]
                input_frame = chainer.Variable(predicted_frame.array)
                self.loss += self.loss_fun(predicted_frame,
                                           expected_frame,
                                           self.k_step)
        reporter.report({'loss': self.loss}, self)
        return self.loss

    def reset_state(self):
        self.predictor.reset_state()
