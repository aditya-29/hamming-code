import math as m
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import hamming as Ham

class BPSK:
    def __init__(self, tot_bits, scale = False):
        self.fc = 4000  #seconds
        self.tot_bits = tot_bits
        if scale:
            self.scale = 8/self.tot_bits
        else:
            self.scale = 2
        self.bit_length = int(100 * self.scale)
        self.tot_len = self.tot_bits * self.bit_length

        self.fc = 4000
        self.t = np.linspace(0,self.tot_len*4,self.tot_len)
        self.fs = 20 * self.fc
        self.amp = 1
        self.carrier = self.amp * np.cos(np.dot(2 * m.pi * self.fc,self.t))


    def modulate(self, bit_arr):
        fin_output = []
        for i in range(len(bit_arr)):
            mod_bit_arr = []
            for j in bit_arr[i]:
                if j:
                    temp = [1] * self.bit_length 
                else:
                    temp = [-1] * self.bit_length 
                mod_bit_arr.extend(temp)
            output = mod_bit_arr * self.carrier   
            fin_output.append(output)
            
        return self.carrier, fin_output
        
    def demodulate(self, waves, plot=False):
        [b11,a11] = signal.ellip(5, 0.5, 60, [2000 * 2 / 80000, 6000 * 2 / 80000], btype = 'bandpass', analog = False, output = 'ba')
        [b12,a12] = signal.ellip(5, 0.5, 60, (2000 * 2 / 80000), btype = 'lowpass', analog = False, output = 'ba')
        output = []

        for wave in waves:
            bandpass_out = signal.filtfilt(b11, a11, wave)
            coherent_demod = bandpass_out * (self.carrier * 2)
            lowpass_out = signal.filtfilt(b12, a12, coherent_demod)
            if plot:
                plt.plot(lowpass_out)
                plt.title("low pass output")
                plt.show()
            
            detection_bpsk = np.zeros(len(self.t), dtype=np.float32)
            flag = [0 for i in range(self.tot_bits)]

            for i in range(self.tot_bits):
                tempF = 0
                for j in range(self.bit_length):
                    tempF = tempF + lowpass_out[i * self.bit_length + j]
                    if tempF > 0:
                        flag[i] = 1
                    else:
                        flag[i] = 0
            output.append(flag)
        return output

class AWGN():
    def awgn1(self,y, snr = 5):
        snr = 10 ** (snr / 10.0)
        output = []
        for i in y:
            i = np.array(i)
            xpower = np.sum(i ** 2) / len(i)
            npower = xpower / snr
            temp = (np.random.randn(len(i)) * np.sqrt(npower) + i)
            output.append(temp)
        return output

    def awgn(self, y, snr = 5):
        snr=10.0**(snr/10.0)
        noise_std = 1/sqrt(2*snr)
        noise_mean = 0
        output = []
        N = len(y[0])
        for i in y:     
            rx_symbol = []   
            for m in range (0, N):
                tx_symbol = i[m]
                noise = random.gauss(noise_mean, noise_std)
                rx_symbol.append(tx_symbol + noise)
            output.append(rx_symbol)
        return output

from numpy import sqrt
from numpy.random import rand, randn
import random

def plot_ber_snr1(orig_signal, snr, blocksize):
    N = len(orig_signal)
    snrindB_range = range(0, snr)
    ber = [None]*snr
    awgn = AWGN()
    bpsk = BPSK(blocksize, scale = False)
    for i in range(len(orig_signal)):
        for n in range (0, snr): 
            no_errors = 0
            output_noised = awgn.awgn([orig_signal[i]], n)
            rx_signal = bpsk.demodulate(output_noised)
            for m in range (0, N):
                tx_symbol = orig_signal[m]
                # noise = random.gauss(noise_mean, noise_std)
                # rx_symbol = tx_symbol + noise
                rx_symbol = float(rx_signal[0][m])
                det_symbol = float(2 * (rx_symbol >= 0) - 1)
                no_errors += 1*(tx_symbol != det_symbol)  
            ber[n] = no_errors / N
        
        print("len of ber :" ,len(ber))

        plt.plot(snrindB_range, ber, 'o-',label='practical')
        plt.axis([0, 10, 0, 0.1])
        plt.xlabel('snr(dB)')
        plt.ylabel('BER')
        plt.grid(True)
        plt.title('BPSK Modulation')
        # plt.legend()
        plt.show()


def plot_ber_snr(input_arr, snr, blocksize):
    N = len(input_arr)
    snrindB_range = range(0, snr)
    ber = [None]*snr
    awgn = AWGN()
    bpsk = BPSK(blocksize)
    H = Ham.Hamming(blocksize, interlacing = False)
    enc_output = H.enc_hamming(input_arr, extended = True)
    carrier, output = bpsk.modulate(enc_output)
    for n in range (0, snr): 
        no_errors = 0
        output_noised = awgn.awgn(output, n)
        demodulated = bpsk.demodulate(output_noised)
        decoded = H.dec_hamming(demodulated, extended = True)
        for i in range(len(input_arr)):
            no_errors += 1*(input_arr[i] != decoded[i])
        print("no errors : ", no_errors)
        ber[n] = no_errors / N
    
    print("len of ber :" ,len(ber))
    print("ber : ", ber)
    
    plt.plot(snrindB_range, ber, 'o-',label='practical')
    # plt.axis([0, 10, 0, 0.1])
    plt.xlabel('snr(dB)')
    plt.ylabel('BER')
    plt.grid(True)
    plt.title('BPSK Modulation')
    # plt.legend()
    plt.show()

if __name__ == "__main__":
    bpsk = BPSK(8)
    bits = [[1,0,1,0,1,0,1,1]]
    carrier, output = bpsk.modulate(bits)
    # plot_ber_snr(bits[0])
    bpsk.demodulate(output)

