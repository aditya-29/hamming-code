import hamming as Ham
import transmission as Tran
import matplotlib.pyplot as plt
input_arr = "10011"
input_arr = list(map(int,input_arr))

print("input array : ", input_arr)

block_size = 8

H = Ham.Hamming(block_size, interlacing = False)

enc_output = H.enc_hamming(input_arr, extended = True)
print("encoded input      : ",enc_output)

bpsk = Tran.BPSK(block_size)
carrier, output = bpsk.modulate(enc_output)
for i in output:
    plt.plot(i)
    plt.show()

awgn = Tran.AWGN()

output_noised = awgn.awgn(output,5)

for i in range(len(output_noised)):
    plt.plot(output_noised[i])
    plt.title("output after adding noise")
    plt.show()


demodulated = bpsk.demodulate(output_noised, plot=True)
print("demodulated signal : ", demodulated)


decoded = H.dec_hamming(demodulated, extended = True)
print("final output       : ",decoded)


print("calculating the BER vs SNR Plot ...")
# Tran.plot_ber_snr(orig_signal = output, snr = 10, blocksize = block_size )
Tran.plot_ber_snr(input_arr = input_arr, snr = 10, blocksize = block_size )
