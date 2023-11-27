import numpy as np
from PIL import Image
from math import log2
from collections import Counter
from matplotlib import pyplot as plt

def parse_img(img):
    pixels = np.array(img.getdata())
    if len(np.shape(pixels)) == 1:
        pixels = pixels[:, None]
    dim = np.shape(pixels)[-1]
    return pixels.astype(int), dim

def calc_pmf(arr, dimension):
    val, cnt = np.unique(arr, return_counts=True, axis=0)
    pmf = cnt / len(arr)
    myarr = np.hstack((val.reshape(-1,dimension),pmf.reshape(-1,1)))
    values = myarr[:, :-1]  # Extract values from PMF (excluding the last column, which is the probability)
    probabilities = myarr[:, -1]  # Extract probabilities from PMF

    plt.bar(values.flatten(), probabilities, width=1.0)
    plt.title(f'Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Probability')
    plt.show()
    return myarr

def calc_entropy(img):
    pix, dim = parse_img(img)
    pmf = calc_pmf(pix, dim)
    entropy = 0
    for i in pmf:
        entropy -= i[-1]*log2(i[-1])
    return entropy

def calc_joint_entropy(img1, img2):
    set_a, _ = parse_img(img1)
    set_b, _ = parse_img(img2)
    
    pairs = list(zip(set_a[:, 0], set_b[:, 0]))
    pair_counts = Counter(pairs)
    total_pairs = len(pairs)
    joint_probability = {pair: count / total_pairs for pair, count in pair_counts.items()}
    joint_entropy = -sum(probability * log2(probability) for probability in joint_probability.values())
    return joint_entropy

def mutual_info(h_xy, h_x, h_y):
    return h_x + h_y - h_xy

def max_bit_rate(mi):
    B = 500     # Hz
    bit_rate = B*mi
    return bit_rate

def max_compression_ratio(img_type, entropy):
    size = 0
    if img_type == "colour":
        size = 3*8

    if img_type == "gray-scale":
        size = 8

    return size/entropy

if __name__ == "__main__":
    img_gs = Image.open('Images/ISAE_Logo_SEIS_gs.png', 'r')
    img_gs_noisy = Image.open('Images/ISAE_Logo_SEIS_gs_noisy.png', 'r')
    img_clr = Image.open('Images/ISAE_Logo_SEIS_clr.png', 'r')
    img_clr_noisy = Image.open('Images/ISAE_Logo_SEIS_clr_noisy.png', 'r')

    gs_ent = calc_entropy(img_gs)
    gs_noisy_ent = calc_entropy(img_gs_noisy)
    clr_ent = calc_entropy(img_clr)
    clr_noisy_ent = calc_entropy(img_clr_noisy)

    gs_joint_entropy = calc_joint_entropy(img_gs, img_gs_noisy)
    clr_joint_entropy = calc_joint_entropy(img_clr, img_clr_noisy)

    gs_mi = mutual_info(gs_joint_entropy, gs_ent, gs_noisy_ent)
    clr_mi = mutual_info(clr_joint_entropy, clr_ent, clr_noisy_ent)

    gs_max_bit_rate = max_bit_rate(gs_mi)
    clr_max_bit_rate = max_bit_rate(clr_mi)

    gs_cr = max_compression_ratio("gray-scale", gs_ent)
    gs_noisy_cr = max_compression_ratio("gray-scale", gs_noisy_ent)
    clr_cr = max_compression_ratio("colour", clr_ent)
    clr_noisy_cr = max_compression_ratio("colour", clr_noisy_ent)

    print("Entropy GS: ", gs_ent)
    print("Entropy GS noisy: ", gs_noisy_ent)
    print("Entropy CLR: ", clr_ent)
    print("Entropy CLR noisy: ", clr_noisy_ent)

    print(f"Joint entropy for grey-scale images: {gs_joint_entropy}")
    print(f"Joint entropy for colour images {clr_joint_entropy}")

    print(f"Maximum compression ratio for grey-scale images: {gs_cr}")
    print(f"Maximum compression ratio for grey-scale, noisy images: {gs_noisy_cr}")
    print(f"Maximum compression ratio for colour images: {clr_cr}")
    print(f"Maximum compression ratio for colour, noisy images: {clr_noisy_cr}")

    print(f"Max bit rate for grey-scale images: {gs_max_bit_rate}")
    print(f"Max bit rate for colour images: {clr_max_bit_rate}")
