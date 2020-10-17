import hamming as Ham
import transmission as Tran
import matplotlib.pyplot as plt


def subplots_adjust(left, bottom, right, top, wspace, hspace):
    plt.subplots_adjust(left = left, bottom = bottom, right = right, top = top, wspace = wspace, hspace = hspace)


input_arr = input("enter the bits eg: 10010101 :  ")
# input_arr = "10011"
input_arr = list(map(int,input_arr))

print("input array : ", input_arr)

block_size = int(input("enter the block size eg : 8 : "))
# block_size = 8

H = Ham.Hamming(block_size, interlacing = False)

enc_output = H.enc_hamming(input_arr, extended = True)
print("encoded input      : ",enc_output)

bpsk = Tran.BPSK(block_size)
carrier, output = bpsk.modulate(enc_output)

plt.plot(carrier)
plt.title("carrier wave")
plt.show()

awgn = Tran.AWGN()

output_noised = awgn.awgn(output,5)

for i in range(1,len(output_noised)+1,1):
    subplots_adjust(left  = 0.125, right = 0.9, bottom = 0.1, top = 0.9, wspace = 0.2, hspace = 1)

    plt.subplot(len(output)*2,1,i)
    plt.plot(output[i-1])
    plt.title("output after BPSK Modulation {}/{}".format(i,len(output)))

    plt.subplot(len(output)*2,1,i+len(output))
    plt.plot(output_noised[i-1])
    plt.title("modulated output after adding noise {}/{}".format(i,len(output)))
plt.show()


demodulated = bpsk.demodulate(output_noised, plot=True)
print("demodulated signal : ", demodulated)


decoded = H.dec_hamming(demodulated, extended = True)
print("final output       : ",decoded)


print("calculating the BER vs SNR Plot ...")
# Tran.plot_ber_snr(orig_signal = output, snr = 10, blocksize = block_size )
Tran.plot_ber_snr(input_arr = input_arr, snr = 10, blocksize = block_size )
