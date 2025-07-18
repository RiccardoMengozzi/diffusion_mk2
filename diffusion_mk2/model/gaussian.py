import numpy as np
import torch
import matplotlib.pyplot as plt

def gaussian_1D_label(label, num_class, sig=4.0):
    label = int(label)
    x = np.arange(num_class)
    y_sig = np.exp(-((x - label) ** 2) / (2 * sig**2))
    return y_sig



if __name__ == "__main__":

    
    idx = 7
    num_class = 15

    label = gaussian_1D_label(idx, num_class, sig=1.0)

    label_tensor = torch.tensor(label).type(torch.FloatTensor)

    print(f'Label for index {idx}: {label}')

    plt.plot(label, marker='o', label=f'Label {idx}')
    plt.tight_layout()
    plt.show()