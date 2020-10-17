import numpy as np
import math as m
import os

class Hamming():
    def __init__(self,block_size, interlacing = True, print=True):
        self.block_size = block_size
        self.divided_arr = []
        self.interlacing = interlacing
        self.usable_bits = None
        self.cal_r_bits()
        self.print = print
        
    def cal_r_bits(self):
        if self.block_size>>1 == self.block_size-1>>1:
            raise ValueError("the entered block size is not of power of 2")
        r_bits = len(bin(self.block_size))-3
        return r_bits

    def arrange(self,arr):
        output = [0] * self.block_size
        i = 0
        for ind, bit in enumerate(output):
            # if ind in [0,1,2] or (ind>>2 != (ind-1)>>2):
            if ind == 0:
                pass
            else:
                temp = m.log(ind, 2)
                if (int(temp) == m.ceil(temp)):
                    pass
                else:
                    output[ind] = arr[i]
                    i+=1
            if i==len(arr):
                break 
        return output
            
    def interlace(self, reverse = False):
        li = len(self.divided_arr) 
        lj = len(self.divided_arr[0]) 

        temp = [0 for i in range(li*lj)]
        if not reverse:
            for i in range(li):
                temp[i::li] = self.divided_arr[i]
            temp = np.array(temp)
            temp = temp.reshape(li,lj)
            return temp
        else:
            self.divided_arr = np.array(self.divided_arr)
            self.divided_arr = self.divided_arr.reshape(-1)
            output = []
            for i in range(li):
                temp = self.divided_arr[i::li]
                output.append(list(temp))
            return output

    def divide(self,arr):
        r_bits = self.cal_r_bits()
        self.usable_bits = self.block_size - r_bits - 1
        if self.print:
            print("usable bits : ",self.usable_bits,end = "\n\n")
            print("({},{}) => Hamming Notation".format(self.block_size-1, self.usable_bits),end = "\n\n")

        if len(arr) <= self.usable_bits:
            print("no dividing necessary")
            output = self.arrange(arr)
            self.divided_arr.append(output)
        
        else:
            print("Dividing necessary !!")
            output = []
            for ind in range(0, len(arr), self.usable_bits):
                temp = arr[ind:ind+self.usable_bits]
                temp = self.arrange(temp)
                output.append(temp)
            self.divided_arr = output   

            if self.interlacing:
                print("applying interlacing")
                self.interlace(reverse = True)



    def enc_hamming(self, arr, extended = True):
        # val = reduce(lambda x,y:x ^ y, [ind for ind, bit in enumerate(arr) if bit])

        self.divide(arr)
        if self.print:
            print("divided input      : ",self.divided_arr)
        for i in range(len(self.divided_arr)):
            no_1 = 0
            val = 0
            for ind, bit in enumerate(self.divided_arr[i]):
                if bit:
                    val = val^ind

            for ind, bit in enumerate(bin(val)):
                if bit=="1":
                    temp = bit + "0"*(len(bin(val)) -1 -ind)
                    self.divided_arr[i][int(temp,2)] = int(not self.divided_arr[i][int(temp,2)])

            for ind, bit in enumerate(self.divided_arr[i]):
                if bit:
                    no_1+=1

            if extended:
                if no_1%2 == 0:
                    self.divided_arr[i][0] = 0
                else:
                    self.divided_arr[i][0] = 1

            for ind, bit in enumerate(self.divided_arr[i]):
                if bit:
                    no_1+=1
                    val = val^ind

        return self.divided_arr

    def dec_hamming(self, arr, extended = True):
        # val = reduce(lambda x,y:x ^ y, [ind for ind, bit in enumerate(arr) if bit])
        if self.interlacing:
            print("reversing interlacing")
            self.interlace(reverse = True)
        output = []
        # r_bits = int(np.log2(len(arr[0])))
        for i in range(len(arr)):
            no_1 = 0
            val = 0
            for ind, bit in enumerate(arr[i]):
                if bit:
                    no_1+=1
                    val = val^ind
            if val==0 and no_1%2==1:
                arr[i][0] = int(not arr[i][0])
            elif val!=0 and no_1%2==1:
                pos = int(bin(val)[2:],2)
                print("Error detected at : [{}][{}]".format(i,pos))
                arr[i][val] = int(not arr[i][val])
                output.append(arr[i])
            elif val==0:
                output.append(arr[i])
            else:
                print("More than 1 error detected !!!")
                # return None
                output.append(arr[i])
        if self.print:
            print("decoded output     : ",output)

        final_output = []
        for i in range(len(arr)):
            for j in range(len(arr[0])):
                if j==0:
                    pass
                else:
                    temp = m.log(j,2)
                    if temp != m.ceil(temp):
                        final_output.append(arr[i][j])
                    else:
                        pass
        return final_output


#SAMPLE PROGRAM

if __name__ == "__main__":
    np.random.seed(5)
    arr = np.random.randint(0,2, size = 20)

    arr = "01010011110001000100"
    arr = list(map(int,arr))
    block_size = 16

    print("input array : ",arr,end = "\n\n")

    H = Hamming(block_size = 8, interlacing = False)

    enc_output = H.enc_hamming(arr, extended =True)
    print("encoded output     : " ,enc_output, end = "\n\n")

    i,j = 2,3
    enc_output[i][j] = int(not enc_output[i][j])

    i,j = 1,3
    enc_output[i][j] = int(not enc_output[i][j])

    i,j = 1,5
    enc_output[i][j] = int(not enc_output[i][j])



    print("changed enc_output : ",enc_output,end = "\n\n")
    dec_output = H.dec_hamming(enc_output, extended = True)
    print("decoded output     : ", dec_output,end = "\n\n")

 