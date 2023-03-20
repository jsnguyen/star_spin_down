import pickle
import numpy as np
import matplotlib.pyplot as plt
from star_spin_down import imf_kroupa2013_unnorm

def main():
    width = 1e-5
    xs = np.arange(0.07, 150, width)
    v_imf = np.vectorize(imf_kroupa2013_unnorm, otypes=[float])
    norm = np.trapz(v_imf(xs), xs)
    print('Calculated Normalization Kroupa 2013 IMF Normalization\n{}'.format(norm))

    with open('kroupa2013_norm.pickle', 'wb') as f:
        pickle.dump(norm, f)

    

if __name__=='__main__':
    main()
