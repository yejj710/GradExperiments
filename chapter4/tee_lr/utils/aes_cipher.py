import binascii
import base64
from Crypto.Cipher import AES
from cryptography.hazmat.primitives import padding
import numpy as np
import time
import os
import pickle


def encrypt_aes256gcm(key, ciphertext, iv):
    # cipher = AES.new(key, AES.MODE_GCM, iv)
    # ed = cipher.encrypt(ciphertext.encode())
    ed, auth_tag = key.encrypt_and_digest(ciphertext.encode())
    return binascii.hexlify(iv + ed + auth_tag).decode()  # 如果不加auth_tag,则少32位；有的是进行base64编码


def decrypt_aes256gcm(key, ciphertext):
    hex_ciphertext = binascii.unhexlify(ciphertext)
    iv = hex_ciphertext[:12]
    
    data = hex_ciphertext[12:-16]
    
    auth_tag = hex_ciphertext[-16:]
    
    # cipher = AES.new(key, AES.MODE_GCM, iv)
     
    dd = key.decrypt_and_verify(data, auth_tag)
    # print(dd)
    return dd.decode()


def encrypt_aesgcm(cipher, plaintext):
    # cipher = AES.new(key, AES.MODE_GCM, iv)

    ed, auth_tag = cipher.encrypt_and_digest(plaintext)
    return binascii.hexlify(iv + ed + auth_tag).decode()


def decrypt_aesgcm(key, ciphertext):
    hex_ciphertext = binascii.unhexlify(ciphertext)
    iv = hex_ciphertext[:12]
    data = hex_ciphertext[12:-16]
    auth_tag = hex_ciphertext[-16:]
    cipher = AES.new(key, AES.MODE_GCM, iv)
    plain_text = cipher.decrypt_and_verify(data, auth_tag)
    # print(plain_text)
    return plain_text


if __name__ == "__main__":
    # init key
    key = '45/nyAIYm52yBAbNlENJ1A=='
    # data = 'd2e3c581e860036321e4c63c55e57fea9b1e5961bf5fcc118c114ecf8aaf99496d2c70e5b790357554ad8f20428f1a16da5e0c53537e5bb778ecbfca7ec1246ff0b18cbf686c204bb4d66175f64d626d'
    key = base64.b64decode(key.encode())
    iv = os.urandom(12)
    start_time = time.time()
    # init data
    wx1 = np.random.uniform(low=-1, high=1, size=(40000, ))
    wx1_bytes = pickle.dumps(wx1)
    # print(binascii.hexlify(iv))
    
    
    cipher = AES.new(key, AES.MODE_GCM, iv)
    text = '{"siteId":"924958456908492800","platform":"Android"}'

    endata = encrypt_aesgcm(cipher, wx1_bytes)
    # print(data)
    # print(endata)
    decdata = decrypt_aesgcm(key, endata)
    # print(pickle.loads(decdata))
    print("cost time : ", (time.time()-start_time)*2)