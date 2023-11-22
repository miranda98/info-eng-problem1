import scipy.integrate as integrate
import scipy.special as special
"""
Find the maximum bit rate with that formula
Somehow quantify the noise and signal

The maximum bit rate (in bits/sec) which could be transmitted through the
rover-orbiter-DSN (Deep Space Network) channel of bandwidth B = 500Hz.
This communication channel is impacted with an additive noise, as shown in
Figure 2 (b).

Maximum bit rate (bits/sec) for transmitting through the rover-orbiter-DSN (Deep Space Network) channel
- Channel bandwidth = 500 Hz
- Communication channel has noise`

Shannon-Hartley Theorem:

C = B*log2(1 + (S/N))

C is the channel capacity
B is the bandwidth = 500 Hz
S is the signal power
N is the noise power

"""

"""
Calculate signal power for DIGITAL SIGNALS

st: signal waveform
Tb: bit period
"""
def signal_power(st, Tb):

    integral = integrate.quad(lambda t: )
    power = 1/Tb*()
    return power
