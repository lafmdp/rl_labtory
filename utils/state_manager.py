'''
  To concat last n states and last (n-1) actions.
  If n == 5:
  S_(t+4) = (s_t,s_(t+1),s_(t+2),s_(t+3),s_(t+4),a_t,a_(t+1),a_(t+2),a_(t+3))

  @python version : 3.6.8
  @author : pangjc
  @time : 2020/1/7
'''

import numpy as np
from collections import deque

class state_manager():

    def __init__(self, state_num:int=1, state_dim:int=1, action_dim:int=1):
        """
        在Main函数里有state_manager()的具体用法。
        :param state_num: 拼接过去多少步的状态？
        :param state_dim: 单步状态的维度
        :param action_dim: 单步动作的维度
        """

        self.state_num  = state_num
        self.state_dim  = state_dim
        self.action_dim = action_dim

        self.reset()

    def reset(self):
        """
        When start a new game, call this.
        """

        self.state_list = deque(maxlen=self.state_num)
        self.action_list = deque(maxlen=self.state_num - 1)

        for i in range(self.state_num):
            self.state_list.append([0] * self.state_dim)

        for i in range(self.state_num-1):
            self.action_list.append([0] * self.action_dim)

    def generate_new_state(self, state):
        """
        调用这个函数获取拼接后的状态。
        :param state: 最新一步的单步状态
        :return: 拼接后的过去n步的状态
        """

        if type(state) != list:
            raise TypeError

        if len(state) != self.state_dim:
            raise ValueError

        self.state_list.append(state)

        all_state = np.hstack(self.state_list).reshape([1,-1])
        all_action = np.hstack(self.action_list).reshape([1,-1])

        new_state = np.hstack((all_state, all_action))

        return new_state

    def append_action(self, action):
        """
        当根据拼接后的状态计算出动作（可以是高层策略的输出）之后，调用这个函数添加动作。
        :param action: 最新执行的动作
        :return: None
        """

        if type(action) != list:
            raise  TypeError

        if len(action) != self.action_dim:
            raise ValueError

        self.action_list.append(action)


if __name__ == "__main__":

    def policy(s): # an example of policy. Just replace it
        return np.random.random([3]).tolist()

    high_policy_state_manager = state_manager(state_num=5, state_dim=5, action_dim=3)
    high_policy_state_manager.reset()

    for _ in range(50):

        s1 = np.random.random([5]).tolist()
        S1 = high_policy_state_manager.generate_new_state(s1)
        print("step:",_,S1)
        a1 = policy(S1)
        high_policy_state_manager.append_action(a1)

