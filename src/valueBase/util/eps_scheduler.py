import math

import matplotlib.pyplot as plt


class ActEpsilonScheduler(object):
    def __init__(self, epsilon_start=1.0, epsilon_final=0.01, epsilon_decay=30000, method='linear', start_frame=0,
                 decay_zero=None):
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.method = method
        self.start_frame = start_frame
        self.decay_zero = decay_zero

    def get(self, frame_idx):
        if frame_idx < self.start_frame:
            return self.epsilon_start
        if self.method == 'exponential':
            return self.epsilon_final + (self.epsilon_start - self.epsilon_final) * math.exp(
                -1. * (frame_idx - self.start_frame) / self.epsilon_decay)
        else:
            if self.decay_zero == None or self.decay_zero <= self.start_frame + self.epsilon_decay or frame_idx <= self.start_frame + self.epsilon_decay:
                return max(self.epsilon_final, self.epsilon_start + (self.epsilon_final - self.epsilon_start) * (
                            frame_idx - self.start_frame) * 1. / self.epsilon_decay)
            else:
                return max(0, self.epsilon_final * (self.decay_zero - frame_idx) / (
                            self.decay_zero - self.start_frame - self.epsilon_decay))

if __name__ == '__main__':
    a = ActEpsilonScheduler(method="exponential")
    datas = []
    for i in range(100000):
        data = 1
        if i > 80000:
            data = a.get(i)
        if i % 100 == 0:
            datas.append(data)

    plt.plot(datas)
    plt.show()
    print(datas[-1])