import binascii
import base64
from Crypto.Cipher import AES

#加密函数
def encrypt_aes256gcm(key, ciphertext, iv):
    cipher = AES.new(key, AES.MODE_GCM, iv)
    # ed = cipher.encrypt(ciphertext.encode())
    ed, auth_tag = cipher.encrypt_and_digest(ciphertext.encode())
    return binascii.hexlify(iv + ed + auth_tag).decode()  # 如果不加auth_tag,则少32位；有的是进行base64编码

#解密函数
def decrypt_aes256gcm(key, ciphertext):
    hex_ciphertext = binascii.unhexlify(ciphertext)
    iv = hex_ciphertext[:12]
    print(iv)
    data = hex_ciphertext[12:-16]
    print(data)
    auth_tag = hex_ciphertext[-16:]
    print(auth_tag)
    cipher = AES.new(key, AES.MODE_GCM, iv)
    # dd = cipher.decrypt(data)  和下面结果相同
    dd = cipher.decrypt_and_verify(data, auth_tag)
    print(dd)
    return dd.decode()


if __name__ == "__main__":
    key = '45/nyAIYm52yBAbNlENJ1A=='
    data = 'd2e3c581e860036321e4c63c55e57fea9b1e5961bf5fcc118c114ecf8aaf99496d2c70e5b790357554ad8f20428f1a16da5e0c53537e5bb778ecbfca7ec1246ff0b18cbf686c204bb4d66175f64d626d'
    key = base64.b64decode(key.encode())
    print(key)
    hex_data = binascii.unhexlify(data)
    print(hex_data)
    # iv为密文的前12byte，前24位
    iv = hex_data[:12]
    print(binascii.hexlify(iv))
    print(iv)
    text = '{"siteId":"924958456908492800","platform":"Android"}'
    endata = encrypt_aes256gcm(key, text, iv)
    print(data)
    print(endata)
    decrypt_aes256gcm(key, data)