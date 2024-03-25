from phe import paillier
import time
import numpy as np


if __name__ == "__main__":
    
    public_key, private_key = paillier.generate_paillier_keypair(n_length=2048)

    wx1 = list(np.random.uniform(low=-1, high=1, size=(40, )))
    wx2 = list(np.random.uniform(low=0, high=1, size=(40, )))
    cipher_val = 7

    start_time = time.time()
    # encrypt twice
    enc_wx1 = list(map(lambda x: public_key.encrypt(x), wx1))
    time_per_enc = (time.time()-start_time) / 320
    print("encrypt cost time : ", time_per_enc)

    start_time = time.time()
    dec_wx1 = list(map(lambda x: private_key.decrypt(x), enc_wx1))
    time_per_dec = (time.time()-start_time) / 320
    print("decrypt cost time : ", time_per_dec)

    start_time = time.time()
    val = list(map(lambda x: x * 0.12353483424, enc_wx1))
    time_per_cmp = (time.time()-start_time) / 320
    print("compute on ciphertext cost time : ", time_per_cmp)
    sample_size = 40000
    feature_size = 400
    enc_times = 2*sample_size*time_per_enc
    dec_times = feature_size*time_per_dec
    compute_times = (2+feature_size)*sample_size*time_per_cmp

    print("total addtional time: ", enc_times+dec_times+compute_times+27.523)