'''
  
  @python version : 3.6.8
  @author : pangjc
  @time : 2019/11/21
'''

from matplotlib import pyplot as plt
import seaborn as sns
fig,ax=plt.subplots()

import numpy as np

prey_return = np.load("./prey.npy")
data_return = np.load("dator.npy")

x = range(prey_return.shape[0])
plt.plot(x,prey_return, color='red',  label="Prey", alpha=0.5)

plt.ylabel("Reward during training")
plt.xlabel("Iterations")
plt.legend()
plt.savefig('./reward/prey.pdf')
plt.savefig('./reward/prey.jpg')
plt.close()

plt.plot(x,data_return, color='blue',  label="Predator",alpha=0.5)

plt.ylabel("Reward during training")
plt.xlabel("Iterations")
plt.legend()

plt.savefig('./reward/predator.pdf')
plt.savefig('./reward/predator.jpg')
plt.close()