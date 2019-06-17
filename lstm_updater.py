from chainer import training

class LstmUpdater(training.StandardUpdater):
    def __init__(self, iterator, optimizer, converter, device):
        super(LstmUpdater, self).__init__(iterator, optimizer, converter, device)

    def update_core(self):
        iterator = self.get_iterator('main')
        optimizer = self.get_optimizer('main')

        batch = iterator.next()
        batch = self.converter(batch, self.device)

        loss = optimizer.target(**batch)

        optimizer.target.cleargrads()
        loss.backward()
        loss.unchain_backward()
        optimizer.update()

        optimizer.target.reset_state()




