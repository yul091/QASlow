import numpy as np
import matplotlib.pyplot as plt


ori_lens = np.array([15.7] * 10)
adv_lens = np.array([29.1, 33.4, 35.6, 36.0, 36.0, 36.0, 36.3, 36.3, 36.3, 36.3])

ori_bleus = np.array([2.36] * 10)
adv_bleus = np.array([2.88, 2.47, 2.19, 2.30, 2.30, 2.30, 2.28, 2.28, 2.28, 2.28])

ori_times = np.array([0.091 ] * 10)
adv_times = np.array([0.149, 0.174, 0.185, 0.183, 0.184, 0.185, 0.187, 0.187, 0.202, 0.214])

eps = np.array(list(range(1, 11)))

plt.figure()
plt.plot(eps, adv_lens, linestyle='--', marker='o', markersize=4, color='red', linewidth=1)
plt.grid(color='silver', linestyle='--', linewidth=0.3)
plt.xlabel('Perturbation times')
plt.ylabel('Output length')
plt.title('Output length vs. Perturbation times')
# plt.legend(title='Evaluation metrics', title_fontsize=12, loc='best')
# plt.show()
plt.savefig('output_lengths.png')


plt.figure()
plt.plot(eps, adv_bleus, linestyle='--', marker='^', markersize=4, color='blue', linewidth=1)
plt.grid(color='silver', linestyle='--', linewidth=0.3)
plt.xlabel('Perturbation times')
plt.ylabel('BLEU')
plt.title('BLEU score vs. Perturbation times')
plt.savefig('BLEUs.png')


plt.figure()
plt.plot(eps, adv_times, linestyle='--', marker='s', markersize=4, color='green', linewidth=1)
plt.grid(color='silver', linestyle='--', linewidth=0.3)
plt.xlabel('Perturbation times')
plt.ylabel('Latency')
plt.title('Decoding Latency vs. Perturbation times')
plt.savefig('latencies.png')
