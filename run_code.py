import matplotlib.pyplot as plt
import numpy as np
import AlgoNLMS as algo

aec = algo.AlgoNLMS()
aec.fs = 16000
total_samples = 16000 * 2  #2 seconds

#storage for plotting
d_arr = np.zeros(total_samples)
e_arr = np.zeros(total_samples)
echo_arr = np.zeros(total_samples)

for i in range(total_samples):
    #phase 2, Step 1: Input Acquisition
    x_n, d_n = aec.genSample()

    #step 2: Buffer Update
    aec.updateBuffer(x_n)

    #step 3: Echo Estimation
    yEst = aec.estEcho()

    #step 4: Error Calculation
    aec.calcError(d_n, yEst)

    #step 5: State Detection (Geigel)
    adapt = aec.checkState(d_n)

    #step 6: Weight Update (only if far-end only)
    if adapt:
        aec.updateWeights()

    #store for plots
    d_arr[i] = d_n
    e_arr[i] = aec.en
    echo_arr[i] = 0.5 * x_n  #known echo component from genSample


#ERLE Calculation 
#ERLE_dB = 10 * log10( sum|echo|^2 / (sum|residual|^2 + eps) )
#Computed over a sliding window

window_len = int(0.2 * aec.fs)  #200 ms window
epsilon = 1e-10
erle_db = np.full(total_samples, np.nan)

residual = echo_arr - (d_arr - e_arr)  #echo - estimated echo

for n in range(window_len, total_samples):
    echo_win = echo_arr[n - window_len:n]
    res_win = residual[n - window_len:n]
    erle_db[n] = 10 * np.log10(np.sum(echo_win**2) / (np.sum(res_win**2) + epsilon))

time_axis = np.arange(total_samples) / aec.fs

plt.figure(figsize=(12, 4))
plt.plot(time_axis, erle_db, linewidth=1.0)
plt.axhline(y=15, color='orange', linestyle='--', label='Acceptable (15 dB)')
plt.axhline(y=25, color='green', linestyle='--', label='Excellent (25 dB)')
plt.xlabel('Time (s)')
plt.ylabel('ERLE (dB)')
plt.title('Echo Return Loss Enhancement (ERLE) vs Time')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


#Signal Overview 
fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

axes[0].plot(time_axis, d_arr, linewidth=0.4, color='tab:blue')
axes[0].set_ylabel('Amplitude')
axes[0].set_title('Microphone Signal d(n)')
axes[0].grid(True, alpha=0.3)

axes[1].plot(time_axis, e_arr, linewidth=0.4, color='tab:red')
axes[1].set_ylabel('Amplitude')
axes[1].set_xlabel('Time (s)')
axes[1].set_title('Error Signal e(n), Output after Echo Cancellation')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()