import numpy as np
from PIL import Image
from math import log2
from collections import Counter

def parseimg(img):
    pixels = np.array(img.getdata())
    if len(np.shape(pixels)) == 1:
        pixels = pixels[:, None]
    dim = np.shape(pixels)[-1]
    return pixels, dim

def calcpmf(arr, dimension):
    val, cnt = np.unique(arr, return_counts=True, axis=0)
    pmf = cnt / len(arr)
    myarr = np.hstack((val.reshape(-1,dimension),pmf.reshape(-1,1)))
    return myarr

def calcentropy(input):
    entropy = 0
    for i in input:
        entropy -= i[-1]*log2(i[-1])
    return entropy

def entropy_from_img(img):
    pix, dim = parseimg(img)
    pmf = calcpmf(pix, dim)
    entropy = calcentropy(pmf)
    return entropy


def calc_joint_entropy(set_a, set_b):
    # Combine the sets into pairs
    pairs = list(zip(set_a, set_b))

    # Count the occurrences of each pair
    pair_counts = Counter(pairs)

    # Calculate joint probability
    total_pairs = len(pairs)
    joint_probability = {pair: count / total_pairs for pair, count in pair_counts.items()}

    # Calculate joint entropy
    joint_entropy = -sum(probability * log2(probability) for probability in joint_probability.values())

    # Print the joint entropy
    print(f'Joint Entropy: {joint_entropy} bits')


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


if __name__ == "__main__":
    img_gs = Image.open('Images/ISAE_Logo_SEIS_gs.png', 'r')
    img_gs_noisy = Image.open('Images/ISAE_Logo_SEIS_gs_noisy.png', 'r')
    img_clr = Image.open('Images/ISAE_Logo_SEIS_clr.png', 'r')
    img_clr_noisy = Image.open('Images/ISAE_Logo_SEIS_clr_noisy.png', 'r')

    img_gs_parsed = parseimg(img_gs)
    img_gs_noisy_parsed = parseimg(img_gs_noisy)
    img_clr_parsed = parseimg(img_clr)
    img_clr_parsed = parseimg(img_clr_noisy)

    gs_ent = entropy_from_img(img_gs)
    gs_noisy_ent = entropy_from_img(img_gs)
    clr_ent = entropy_from_img(img_gs)
    clr_img_ent = entropy_from_img(img_gs)

    gs_joint_entropy = calc_joint_entropy(img_gs_parsed[:, 0], img_gs_noisy_parsed[:, 0])
    clr_joint_entropy = calc_joint_entropy(clr_gs_parsed[:, 0], clr_gs_noisy_parsed[:, 0])

    gs_max_bit_rate = max_bit_rate(gs_joint_entropy, gs_ent, gs_noisy_ent)
    clr_max_bit_rate = max_bit_rate(clr_joint_entropy, clr_ent, clr_noisy_ent)

    print("Entropy GS: ", gs_ent)
    print("Entropy GS noisy: ", gs_noisy_ent)
    print("Entropy CLR: ", clr_ent)
    print("Entropy CLR noisy: ", clr_noisy_ent)

    print(f"Joint entropy for grey-scale images: {gs_joint_entropy}")
    print(f"Joint entropy for colour images {clr_joint_entropy}")

    print(f"Max bit rate for grey-scale images: {gs_max_bit_rate}")
    print(f"Max bit rate for colour images: {clr_max_bit_rate}")
