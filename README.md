import random
import time

# ------------------------
# DES tables (standard)
# ------------------------

IP = [
    58,50,42,34,26,18,10,2,60,52,44,36,28,20,12,4,
    62,54,46,38,30,22,14,6,64,56,48,40,32,24,16,8,
    57,49,41,33,25,17,9,1,59,51,43,35,27,19,11,3,
    61,53,45,37,29,21,13,5,63,55,47,39,31,23,15,7
]

FP = [
    40,8,48,16,56,24,64,32,39,7,47,15,55,23,63,31,
    38,6,46,14,54,22,62,30,37,5,45,13,53,21,61,29,
    36,4,44,12,52,20,60,28,35,3,43,11,51,19,59,27,
    34,2,42,10,50,18,58,26,33,1,41,9,49,17,57,25
]

PC1 = [
 57,49,41,33,25,17,9,1,58,50,42,34,26,18,
 10,2,59,51,43,35,27,19,11,3,60,52,44,36,
 63,55,47,39,31,23,15,7,62,54,46,38,30,22,
 14,6,61,53,45,37,29,21,13,5,28,20,12,4
]

PC2 = [
 14,17,11,24,1,5,3,28,15,6,21,10,
 23,19,12,4,26,8,16,7,27,20,13,2,
 41,52,31,37,47,55,30,40,51,45,33,48,
 44,49,39,56,34,53,46,42,50,36,29,32
]

E = [
 32,1,2,3,4,5,4,5,6,7,8,9,
 8,9,10,11,12,13,12,13,14,15,16,17,
 16,17,18,19,20,21,20,21,22,23,24,25,
 24,25,26,27,28,29,28,29,30,31,32,1
]

P = [
 16,7,20,21,29,12,28,17,1,15,23,26,
 5,18,31,10,2,8,24,14,32,27,3,9,
 19,13,30,6,22,11,4,25
]

SHIFT_SCHEDULE = [1,1,2,2,2,2,2,2,1,2,2,2,2,2,2,1]

# Full DES S-boxes (8 boxes, each 4x16)
S_BOXES = [
    # S1
    [
        [14,4,13,1,2,15,11,8,3,10,6,12,5,9,0,7],
        [0,15,7,4,14,2,13,1,10,6,12,11,9,5,3,8],
        [4,1,14,8,13,6,2,11,15,12,9,7,3,10,5,0],
        [15,12,8,2,4,9,1,7,5,11,3,14,10,0,6,13],
    ],
    # S2
    [
        [15,1,8,14,6,11,3,4,9,7,2,13,12,0,5,10],
        [3,13,4,7,15,2,8,14,12,0,1,10,6,9,11,5],
        [0,14,7,11,10,4,13,1,5,8,12,6,9,3,2,15],
        [13,8,10,1,3,15,4,2,11,6,7,12,0,5,14,9],
    ],
    # S3
    [
        [10,0,9,14,6,3,15,5,1,13,12,7,11,4,2,8],
        [13,7,0,9,3,4,6,10,2,8,5,14,12,11,15,1],
        [13,6,4,9,8,15,3,0,11,1,2,12,5,10,14,7],
        [1,10,13,0,6,9,8,7,4,15,14,3,11,5,2,12],
    ],
    # S4
    [
        [7,13,14,3,0,6,9,10,1,2,8,5,11,12,4,15],
        [13,8,11,5,6,15,0,3,4,7,2,12,1,10,14,9],
        [10,6,9,0,12,11,7,13,15,1,3,14,5,2,8,4],
        [3,15,0,6,10,1,13,8,9,4,5,11,12,7,2,14],
    ],
    # S5
    [
        [2,12,4,1,7,10,11,6,8,5,3,15,13,0,14,9],
        [14,11,2,12,4,7,13,1,5,0,15,10,3,9,8,6],
        [4,2,1,11,10,13,7,8,15,9,12,5,6,3,0,14],
        [11,8,12,7,1,14,2,13,6,15,0,9,10,4,5,3],
    ],
    # S6
    [
        [12,1,10,15,9,2,6,8,0,13,3,4,14,7,5,11],
        [10,15,4,2,7,12,9,5,6,1,13,14,0,11,3,8],
        [9,14,15,5,2,8,12,3,7,0,4,10,1,13,11,6],
        [4,3,2,12,9,5,15,10,11,14,1,7,6,0,8,13],
    ],
    # S7
    [
        [4,11,2,14,15,0,8,13,3,12,9,7,5,10,6,1],
        [13,0,11,7,4,9,1,10,14,3,5,12,2,15,8,6],
        [1,4,11,13,12,3,7,14,10,15,6,8,0,5,9,2],
        [6,11,13,8,1,4,10,7,9,5,0,15,14,2,3,12],
    ],
    # S8
    [
        [13,2,8,4,6,15,11,1,10,9,3,14,5,0,12,7],
        [1,15,13,8,10,3,7,4,12,5,6,11,0,14,9,2],
        [7,11,4,1,9,12,14,2,0,6,10,13,15,3,5,8],
        [2,1,14,7,4,10,8,13,15,12,9,0,3,5,6,11],
    ],
]


# ------------------------
# Helper utility functions
# ------------------------

def hex_to_bin64(hex_str):
    """Convert 16-hex-digit string to 64-bit binary string."""
    return format(int(hex_str, 16), "064b")

def bin_to_hex64(bin_str):
    """Convert 64-bit binary string to 16-hex-digit uppercase string."""
    return format(int(bin_str, 2), "016X")

def permute(bits, table):
    """Apply a permutation table (1-based indices) to bits string."""
    return "".join(bits[i - 1] for i in table)

def left_rotate(bits, n):
    """Rotate a string left by n positions."""
    return bits[n:] + bits[:n]

def xor_bits(a, b):
    """XOR two bit-strings of equal length and return resulting bit-string."""
    return "".join("0" if a[i] == b[i] else "1" for i in range(len(a)))


# ------------------------
# Key schedule generation
# ------------------------

def generate_round_keys(key_bin64):
    """
    key_bin64: 64-bit binary string (as returned by hex_to_bin64)
    returns: list of 16 round keys (each 48-bit binary strings)
    """
    key56 = permute(key_bin64, PC1)   # drop parity bits -> 56 bits
    C = key56[:28]
    D = key56[28:]
    round_keys = []

    for shift in SHIFT_SCHEDULE:
        C = left_rotate(C, shift)
        D = left_rotate(D, shift)
        combined = C + D
        round_key = permute(combined, PC2)  # 48 bits
        round_keys.append(round_key)

    return round_keys


# ------------------------
# Feistel (round) functions
# ------------------------

def s_box_substitution(bits48):
    """bits48: 48-bit string split into 8 blocks of 6 bits each.
       Returns 32-bit string after S-box substitution."""
    out = []
    for i in range(8):
        block = bits48[i*6:(i+1)*6]
        row = int(block[0] + block[5], 2)      # first and last bit
        col = int(block[1:5], 2)              # middle 4 bits
        val = S_BOXES[i][row][col]
        out.append(format(val, "04b"))
    return "".join(out)


def feistel(R, K):
    """One Feistel function: expand, xor with key, s-box, then permute P."""
    expanded = permute(R, E)           # 48 bits
    xored = xor_bits(expanded, K)      # 48 bits
    after_s = s_box_substitution(xored)  # 32 bits
    return permute(after_s, P)         # 32 bits permuted


# ------------------------
# Block encrypt/decrypt
# ------------------------

def des_encrypt_block(pt_hex16, key_hex16):
    """Encrypt a single 64-bit block (hex string length 16)."""
    bits = permute(hex_to_bin64(pt_hex16), IP)
    L, R = bits[:32], bits[32:]
    round_keys = generate_round_keys(hex_to_bin64(key_hex16))

    for i in range(16):
        new_L = R
        new_R = xor_bits(L, feistel(R, round_keys[i]))
        L, R = new_L, new_R

    preoutput = R + L  # note swap before final permute
    return bin_to_hex64(permute(preoutput, FP))


def des_decrypt_block(ct_hex16, key_hex16):
    """Decrypt a single 64-bit block (hex string length 16)."""
    bits = permute(hex_to_bin64(ct_hex16), IP)
    L, R = bits[:32], bits[32:]
    round_keys = generate_round_keys(hex_to_bin64(key_hex16))

    # apply keys in reverse order
    for i in range(15, -1, -1):
        new_L = R
        new_R = xor_bits(L, feistel(R, round_keys[i]))
        L, R = new_L, new_R

    preoutput = R + L
    return bin_to_hex64(permute(preoutput, FP))


# ------------------------
# Simple demo main()
# ------------------------

def main():
    pt = "0123456789ABCDEF"    # sample plaintext (64-bit / 16 hex chars)
    key = "133457799BBCDFF1"   # sample key (64-bit / 16 hex chars)

    cipher = des_encrypt_block(pt, key)
    back = des_decrypt_block(cipher, key)

    print("Plaintext:", pt)
    print("Key      :", key)
    print("Cipher   :", cipher)
    print("Decrypted:", back)


if __name__ == "__main__":
    main()
