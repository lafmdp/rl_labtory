'''
  参数管理器，在class中调用，用于管理该模块需要的参数。

  用法：
  在类初始化时调用args，把被管理的参数传入args，调用时用args.attrname来调用.

  @python version : 3.6.8
  @author : pangjc
  @time : 2019/7/18
'''

class args():

    def __init__(self, **kwargs):

        for key,value in kwargs.items():
            self.__setattr__(key, value)
