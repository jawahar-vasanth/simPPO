import matplotlib.pyplot as plt
import pandas as pd
A = pd.read_csv('test_policy2.txt', delimiter = "\t")
plt.subplot(3,1,1)
plt.plot(A['Itr'],A['EpRet'])
plt.title("Average Episode Return")
plt.subplot(3,1,2)
plt.plot(A['Itr'],A['EpLen'])
plt.title("Average Episode Length") 
plt.subplot(3,1,3)
plt.plot(A['Itr'],A['EpColl'])
plt.title("Average Episode Collision")
plt.show()