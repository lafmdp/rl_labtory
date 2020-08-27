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
        a = np.vstack(a).astype(np.float32)
        s_ = np.vstack(s_).astype(np.float32)

        recorder = np.hstack((s,a,s_))

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
        a = sample[:, self.s_space:self.s_space + self.a_space]
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

    def collect_all_samples(self):
        return self.buffer.copy()


    def specific_sample(self, dataset):
        sample_num = dataset.shape[0]
        if sample_num < self.batch_size:
            sample_index = np.random.choice(sample_num, self.batch_size, replace=True).tolist()
        else:
            sample_index = np.random.choice(sample_num, self.batch_size, replace=False).tolist()

        sample = dataset[sample_index]

        s = sample[:, :self.s_space]
        a = sample[:, self.s_space:self.s_space + self.a_space]
        s_ = sample[:, -self.s_space:]

        s = np.array(s).astype(np.float32)
        a = np.array(a).astype(np.float32)
        s_ = np.array(s_).astype(np.float32)

        ret = {
            "state": s,
            "action": a,
            "state_": s_
        }

        return ret

    def all_sample(self, sample):

        s = sample[:, :self.s_space]
        a = sample[:, self.s_space:self.s_space + self.a_space]
        s_ = sample[:, -self.s_space:]

        s = np.array(s).astype(np.float32)
        a = np.array(a).astype(np.float32)
        s_ = np.array(s_).astype(np.float32)

        ret = {
            "state": s,
            "action": a,
            "state_": s_
        }

        return ret

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

class SSAS_Buffer(Buffer):

    def __init__(self, stack_ipt_dim, s_space=1, a_space=1, size=1000000, batch_size=256):

        self.size = size
        self.batch_size =  batch_size

        self.stack_ipt_dim = stack_ipt_dim
        self.s_space = s_space
        self.a_space = a_space

        self.buffer = np.array([], dtype=np.float32)

    def append(self, stack_s, s, a, s_):

        sample_num = self.buffer.shape[0]
        if  sample_num > self.size:
            index = np.random.choice(sample_num, self.size, replace=False).tolist()
            self.buffer = self.buffer[index]
        else:
            pass

        stack_s = np.vstack(stack_s).astype(np.float32)
        s = np.vstack(s).astype(np.float32)
        a = np.vstack(a).astype(np.float32)
        s_ = np.vstack(s_).astype(np.float32)
        recorder = np.hstack((stack_s, s,a,s_))

        if sample_num == 0:
            self.buffer = recorder.copy()
        else:
            self.buffer = np.vstack((self.buffer, recorder))


    def sample(self):

        return self.specific_sample(self.buffer)

    def collect_all_samples(self):
        return self.buffer.copy()


    def specific_sample(self, dataset):
        sample_num = dataset.shape[0]
        if sample_num < self.batch_size:
            sample_index = np.random.choice(sample_num, self.batch_size, replace=True).tolist()
        else:
            sample_index = np.random.choice(sample_num, self.batch_size, replace=False).tolist()

        sample = dataset[sample_index]

        stack_s = sample[:, :self.stack_ipt_dim]
        s = sample[:, self.stack_ipt_dim:self.stack_ipt_dim + self.s_space]
        a = sample[:, self.stack_ipt_dim + self.s_space: self.stack_ipt_dim + self.s_space + self.a_space]
        s_ = sample[:, -self.s_space:]

        stack_s = np.array(stack_s).astype(np.float32)
        s = np.array(s).astype(np.float32)
        a = np.array(a).astype(np.float32)
        s_ = np.array(s_).astype(np.float32)

        ret = {
            "stack_state": stack_s,
            "state": s,
            "action": a,
            "state_": s_
        }

        return ret

    def all_sample(self, sample):

        stack_s = sample[:, :self.stack_ipt_dim]
        s = sample[:, self.stack_ipt_dim:self.stack_ipt_dim + self.s_space]
        a = sample[:, self.stack_ipt_dim + self.s_space: self.stack_ipt_dim + self.s_space + self.a_space]
        s_ = sample[:, -self.s_space:]

        stack_s = np.array(stack_s).astype(np.float32)
        s = np.array(s).astype(np.float32)
        a = np.array(a).astype(np.float32)
        s_ = np.array(s_).astype(np.float32)

        ret = {
            "stack_state": stack_s,
            "state": s,
            "action": a,
            "state_": s_
        }

        return ret

"""
  Store (s,a,r,v_done,s')
"""
class SAC_Buffer(Buffer):

    def __init__(self, s_space=1, a_space=1, size=1000000, batch_size=256):

        self.size = size
        self.batch_size =  batch_size
        self.s_space = s_space
        self.a_space = a_space

        self.buffer = np.array([], dtype=np.float32)

    def append(self, batch):

        sample_num = self.buffer.shape[0]
        if  sample_num > self.size:
            index = np.random.choice(sample_num, self.size, replace=False).tolist()
            self.buffer = self.buffer[index]
        else:
            pass

        s = np.vstack(batch["state"]).astype(np.float32)
        a = np.hstack(batch["action"]).astype(np.float32).reshape([-1,1])
        r = np.hstack(batch["reward"]).astype(np.float32).reshape([-1,1])
        done = np.hstack(batch["done"]).astype(np.float32).reshape([-1,1])
        s_ = np.vstack(batch["state_"]).astype(np.float32)

        recorder = np.hstack((s, a, r, done, s_))

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
        a = sample[:, self.s_space].squeeze()
        r = sample[:, self.s_space+1].squeeze()
        done = sample[:, self.s_space+2].squeeze()
        s_ = sample[:,-self.s_space:]

        ret = {
            "state":s,
            "action":a,
            "reward":r,
            "done":done,
            "state_":s_
        }

        return ret
