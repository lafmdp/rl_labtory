'''
  
  @python version : 3.6.8
  @author : pangjc
  @time : 2019/9/20
'''

import numpy as np

class Buffer(object):

    def append(self, *args):
        pass

    def sample(self, *args):
        pass


class SAS_Buffer(Buffer):

    def __init__(self, s_space=1, a_space=1, size=1000000, batch_size=256):

        self.size = size
        self.batch_size =  batch_size

        self.s_space = s_space
        self.a_space = a_space

        self.buffer = np.array([], dtype=np.float32)

    def append(self, s, a, s_):

        sample_num = self.buffer.shape[0]
        if  sample_num > self.size:
            index = np.random.choice(sample_num, self.size, replace=False).tolist()
            self.buffer = self.buffer[index]
        else:
            pass

        s = np.vstack(s).astype(np.float32)
        a = np.vstack(a).astype(dtype=np.float32)
        s_ = np.vstack(s_).astype(np.float32)
        recorder = np.hstack((s,a,s_))

        if sample_num == 0:
            self.buffer = recorder.copy()
        else:
            self.buffer = np.vstack((self.buffer, recorder))

    def sample(self, size=None):
        size = size if size is not None else self.batch_size

        sample_num = self.buffer.shape[0]
        if sample_num < size:
            sample_index = np.random.choice(sample_num, size, replace=True).tolist()
        else:
            sample_index = np.random.choice(sample_num, size, replace=False).tolist()

        sample = self.buffer[sample_index]

        s = sample[:, :self.s_space]
        a = sample[:, self.s_space:self.s_space+self.a_space]
        s_ = sample[:, -self.s_space:]


        s = np.array(s).astype(np.float32)
        a = np.array(a).astype(np.float32)
        s_ = np.array(s_).astype(np.float32)

        ret = {
            "state":s,
            "action":a,
            "state_":s_
        }

        return ret

    def sample_ss_(self):

        sample_num = self.buffer.shape[0]
        if sample_num < self.batch_size:
            sample_index = np.random.choice(sample_num, self.batch_size, replace=True).tolist()
        else:
            sample_index = np.random.choice(sample_num, self.batch_size, replace=False).tolist()

        sample = self.buffer[sample_index]

        s = sample[:, :self.s_space]
        s_ = sample[:, -self.s_space:]


        s = np.array(s).astype(np.float32)
        s_ = np.array(s_).astype(np.float32)

        ret = {
            "state":s,
            "state_":s_
        }

        return ret

    def traverse(self):

        temp_buffer = np.random.shuffle(self.buffer)

        buffer_size = self.buffer.shape[0]
        start = 0
        end = start + self.batch_size

        while end < buffer_size:
            sample = temp_buffer[start:end]

            s = sample[:, :self.s_space]
            a = sample[:, self.s_space:self.s_space + self.a_space]
            s_ = sample[:, -self.s_space:]

            ret = {
                "state": s,
                "action": a,
                "state_": s_
            }

            start += self.batch_size
            end += self.batch_size

            yield ret

class SS_Buffer(Buffer):

    def __init__(self, s_space=1, size=1000000, batch_size=256):

        self.size = size
        self.batch_size =  batch_size
        self.s_space = s_space

        self.buffer = np.array([], dtype=np.float32)

    def append(self, s, s_):

        sample_num = self.buffer.shape[0]
        if  sample_num > self.size:
            index = np.random.choice(sample_num, self.size, replace=False).tolist()
            self.buffer = self.buffer[index]
        else:
            pass

        s = np.vstack(s).astype(np.float32)
        s_ = np.vstack(s_).astype(np.float32)
        recorder = np.hstack((s, s_))

        if sample_num == 0:
            self.buffer = recorder.copy()
        else:
            self.buffer = np.vstack((self.buffer, recorder))

    def sample(self):

        sample_num = self.buffer.shape[0]
        if sample_num < self.batch_size:
            sample_index = np.random.choice(sample_num, self.batch_size, replace=True).tolist()
        else:
            sample_index = np.random.choice(sample_num, self.batch_size, replace=False).tolist()

        sample = self.buffer[sample_index]

        s = sample[:, :self.s_space]
        s_ = sample[:, -self.s_space:]


        s = np.array(s).astype(np.float32)
        s_ = np.array(s_).astype(np.float32)

        ret = {
            "state":s,
            "state_":s_
        }

        return ret


class SA_Buffer(Buffer):

    def __init__(self, s_space=1, a_space=1, size=1000000, batch_size=256):

        self.size = size
        self.batch_size =  batch_size
        self.s_space = s_space
        self.a_space = a_space

        self.buffer = np.array([], dtype=np.float32)

    def append(self, s, a):

        sample_num = self.buffer.shape[0]
        if  sample_num > self.size:
            index = np.random.choice(sample_num, self.size, replace=False).tolist()
            self.buffer = self.buffer[index]
        else:
            pass

        s = np.vstack(s).astype(np.float32)
        a = np.vstack(a).astype(np.float32)
        recorder = np.hstack((s, a))

        if sample_num == 0:
            self.buffer = recorder.copy()
        else:
            self.buffer = np.vstack((self.buffer, recorder))

    def sample(self):

        sample_num = self.buffer.shape[0]
        if sample_num < self.batch_size:
            sample_index = np.random.choice(sample_num, self.batch_size, replace=True).tolist()
        else:
            sample_index = np.random.choice(sample_num, self.batch_size, replace=False).tolist()

        sample = self.buffer[sample_index]

        s = sample[:, :self.s_space]
        a = sample[:, -self.a_space:]

        s = np.array(s).astype(np.float32)
        a = np.array(a).astype(np.float32)

        ret = {
            "state":s,
            "action":a
        }

        return ret
