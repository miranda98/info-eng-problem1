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

Find mutual information ?
Discrete data - discrete entropy
Max bit rate = mutual information between input X and output Y -> I(X,Y) [bits/sec/Hz]
Conditional mutual information
To obtain mutual information, you need joint probability, what is the probability of Y given X
See formula of common information
Need PDF of X and PDF of Y and the joint probability
Can obtain common information from the entropy
"""

def max_bit_rate(h_xy, h_x, h_y):
    """
    Calculate mutual information I(X;Y)

    Parameters:
    - h_xy: entropy of Y given X H(X,Y)
    - h_x: entropy of X H(X)
    - h_y: extropy of Y H(Y)

    Returns:
    - Mutual information I(X;Y)
    """
    mi = h_x + h_y - h_xy
    B = 500     # Hz
    bit_rate = B*mi
    return bit_rate





