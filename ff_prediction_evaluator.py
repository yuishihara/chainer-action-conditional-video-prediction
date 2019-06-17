from researchutils.chainer.functions import average_k_step_squared_error

from chainer.link import Chain
from chainer import reporter
import chainer
import chainer.functions as F

class FFPredictionEvaluator(Chain):
    def __init__(self, predictor, loss_fun=average_k_step_squared_error, k_step=1):
        super(FFPredictionEvaluator, self).__init__()
        self.loss_fun = loss_fun
        self.k_step = k_step
        self.loss = None
        with self.init_scope():
            self.predictor = predictor
    

    def __call__(self, *args, **kwargs):
        self.loss = 0
        (train_frame, train_actions) = kwargs['input']
        target_frames = kwargs['target']
        for step in range(self.k_step):
            predicted_frame = self.predictor(
                (train_frame, train_actions[:, step, :, :]))
            expected_frame = target_frames[:, step, :, :, :]

            self.loss += self.loss_fun(predicted_frame, expected_frame, self.k_step)
            if not self.k_step == 1:
                predicted_frame = chainer.Variable(predicted_frame.array)
                train_frame = F.concat((train_frame, predicted_frame), axis=1)[
                    :, 3:, :, :]
        reporter.report({'loss':self.loss}, self)
        return self.loss