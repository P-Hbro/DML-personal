import matplotlib.pyplot as plt

time = [15145,11571,10662,10405,10289,10150,9642]
count = [2,3,4,5,6,7,8]
plt.plot(count, time)
plt.xlabel("count")
plt.ylabel("time")
plt.legend()
plt.show()
