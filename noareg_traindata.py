import torch


def mask_we_dont_care(x, eps=1e-6):
    """
    Return a boolean mask of positions in x that are 'trivial' — 
    i.e., all elements are ~0 or all elements are ~1.
    
    x: tensor of shape [B, L, E]
    returns: mask of shape [B, L], True where we want to ignore
    """
    mask_ones = (x >= 1.0 - eps).all(dim=-1)
    mask_zeros = (x <= eps).all(dim=-1)
    
    mask = mask_ones | mask_zeros
    return mask

def int_to_bits(integers, add_batch_dim: bool = False) -> torch.Tensor:
    matrix = [[(c >> i) & 1 for i in range(32)] for c in integers]  # LSB → MSB
    t = torch.tensor(matrix, dtype=torch.float32)
    return t.unsqueeze(0) if add_batch_dim else t

def int_to_bits_tensor(int_tensor: torch.Tensor, num_bits: torch.uint8 = 32):  
    # int_tensor: (N,) or (batch, N)
    device = int_tensor.device
    shifts = torch.arange(num_bits, device=device, dtype=torch.uint8)
    return ((int_tensor.to(int).unsqueeze(-1) >> shifts.to(int)) & 1).to(torch.uint8)

def bits_to_int(bits: torch.Tensor) -> int:
    bits = bits.flatten().to(torch.int64)
    weights = (1 << torch.arange(bits.numel(), dtype=torch.int64))
    return int((bits * weights).sum().item())

def pad_ints(ints, length):
    while len(ints) < length:
        ints += [0]
    return ints


def hashtronhash(n: int, s: int, max_val: int) -> int:
	# Mixing stage, mix input with salt using subtraction
	m = (n - s) & 0xFFFFFFFF
	# Hashing stage, use xor shift with prime coefficients
	m ^= (m << 2) & 0xFFFFFFFF
	m ^= (m << 3) & 0xFFFFFFFF
	m ^= (m >> 5) & 0xFFFFFFFF
	m ^= (m >> 7) & 0xFFFFFFFF
	m ^= (m << 11) & 0xFFFFFFFF
	m ^= (m << 13) & 0xFFFFFFFF
	m ^= (m >> 17) & 0xFFFFFFFF
	m ^= (m << 19) & 0xFFFFFFFF
	# Mixing stage 2, mix input with salt using addition
	m += s
	m &= 0xFFFFFFFF
	# Modular stage using Lemire's fast alternative to modulo reduction
	return ((m * max_val) >> 32) & 0xFFFFFFFF

def hash_string(string: str) -> int:
	h = 0xFFFFFFFF
	x = string.encode(encoding="utf-8")
	for c in x:
		h = 1 + hashtronhash(h, int(c) & 0xFFFFFFFF, 0xFFFFFFFD)
	return h

padspace = hash_string("_")

def load_tsv_individual(path, seq_len=10, limit=10):
    # --- Training data ---
    in_strings, out_strings = [], []
    possible = {}
    import csv
    pairs = []
    random_in = []
    random_out = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) < 2:
                continue  # skip bad lines

            src = row[0].strip()
            tgt = row[1].strip()

            random_in += [src]
            if len(random_in) > 100:
                random_in = random_in[1:]
            random_out += [tgt.replace(' ', '')]
            if len(random_out) > 100:
                random_out = random_out[1:]

            src = src.split(' ')
            tgt = tgt.split(' ')

            if len(src) != len(tgt):
                continue

            pds = []
            for i in range(len(tgt)):
                if tgt[i].endswith("_"):
                    pds += [1]
                elif tgt[i].startswith("_"):
                    pds += [0]
                else:
                    pds += [-1]
            for i in range(seq_len):
                pds += [-1]

            for i in range(len(src)):
                hsrci = hash_string(src[i])
                htgti = hash_string(tgt[i])
                if not hsrci in possible:
                    possible[hsrci] = {}
                possible[hsrci][htgti] = tgt[i]
                src[i] = hsrci
                tgt[i] = htgti

            while len(src) > 0:

                in_strings += [pad_ints(src, seq_len)[:seq_len]]
                out_strings += [pad_ints(tgt, seq_len)[:seq_len]]

                limit-=1
                if limit <= 0:
                    return in_strings, out_strings, possible, random_in, random_out

                found = seq_len
                for j in range(min(len(tgt), seq_len)):
                    if tgt[j] == padspace:
                        found = j+1
                        break
                    if pds[j] >= 0:
                        found = j + pds[j]
                        if found > 0:
                            break
                        else:
                            found = seq_len

                src = pad_ints(src, found)[found:]
                tgt = pad_ints(tgt, found)[found:]
                pds = pds[found:]

        return in_strings, out_strings, possible, random_in, random_out


def clean_default_choices(in_strings, out_strings, possible):
    real_in_strings, real_out_strings = [], []
    for i in range(len(in_strings)):
        all_ones = True
        for j in range(len(in_strings[i])):
            v = in_strings[i][j]
            if v == 0:
                continue
            if len(possible[v]) > 1:
                all_ones = False
            else:
                out_strings[i][j] = 0xFFFFFFFF
            
        if all_ones:
            continue
        real_in_strings += [in_strings[i]]
        real_out_strings += [out_strings[i]]
    return real_in_strings, real_out_strings

