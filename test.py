import numpy as np

def my_fft(a):
    N = len(a)
    
    # Step 1: Pair elements (0, 16), (1, 17), ..., (15, 31)
    twiddle_1 = np.exp(-2j * np.pi * np.arange(16) / N) 
    first_half = a[:16] + twiddle_1 *  a[16:]
    second_half = a[:16] - twiddle_1 * a[16:]
    
    first_pass = np.concatenate((first_half, second_half), axis=-1)

    c = np.split(first_pass, 2)
    return np.concatenate([np.fft.fft(c[i]) for i in range(2)], axis=-1)
    
    # Step 2: Pair elements (0, 8), (1, 9), ..., (7, 15) and (16, 24), (17, 25), ..., (23, 31)
    twiddle_2 = np.exp(-2j * np.pi * np.arange(8) / (N // 2)) 
    b = np.array([(first_pass[i] , first_pass[i+8]) for i in range(8)])
    c = np.array([(first_pass[i+16] , first_pass[i+24]) for i in range(8)]) 
  
    first_quarter = b[:,0] + twiddle_2 * b[:,1]
    second_quarter = b[:,0] - twiddle_2 * b[:,1]
    third_quarter = c[:,0] + twiddle_2 * c[:,1]
    fourth_quarter = c[:,0] - twiddle_2 * c[:,1]
    
    second_pass = np.concatenate((first_quarter, second_quarter, third_quarter, fourth_quarter), axis=-1)

    # Step 3: Now the second_pass array should be split into 4 smaller FFTs of size 8
    c = np.split(second_pass, 4)
    
    # Calculate the FFT of each of these smaller arrays (of size 8) using numpy's FFT
    res = np.concatenate([np.fft.fft(c[i]) for i in range(4)], axis=-1)

    return res

count = 0
def recursive_fft(x):
    N = len(x)
    if N == 1:
        return x
    else:
        global count 
        print(  f"Recursion level: {count}")
        print(f"Input: {x}")
        X_even = recursive_fft(x[::2])
        X_odd = recursive_fft(x[1::2])
        factor = np.exp(-2j * np.pi * np.arange(N) / N)
        print(f"EVEN: {X_even}")
        print(f"ODD: {X_odd}")
        count = count +  1
        return np.concatenate([X_even + factor[:N//2] * X_odd, X_even + factor[N//2:] * X_odd])

a = np.arange(32)

res = my_fft(a)
np_res = np.fft.fft(a)
res_recursive = recursive_fft(a)
print(f"My FFT result:\n{res}")
print(f"NumPy FFT result:\n{np_res}")
print(f"Absolute difference: {np.abs(res - np_res).sum()}")
print(f"Relative difference: {np.abs(res - np_res).sum() / np.abs(np_res).sum()}")

print(f"My FFT result (recursive)")
print(f"Absolute difference: {np.abs(res_recursive - np_res).sum()}")
print(f"Relative difference: {np.abs(res_recursive - np_res).sum() / np.abs(np_res).sum()}")

