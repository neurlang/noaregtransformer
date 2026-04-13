import csv
import tqdm
from noareg_traindata import hashtronhash, int_to_bits, bits_to_int, pad_ints


def hash_string_small(string: str) -> int:
    h = 0xFFFF
    x = string.encode(encoding="utf-8")
    for c in x:
        h = 1 + hashtronhash(h, int(c) & 0xFFFF, 0xFFFD)
    return h

def load_lexicon(path: str) -> dict:
    """
    Returns map[word] -> list of IPA strings.
    Words with more than one IPA entry are homographs.
    """
    lexicon = {}
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) < 2:
                continue
            word = row[0].strip()
            ipa  = row[1].strip()
            if not word:
                continue
            if word not in lexicon:
                lexicon[word] = []
            if ipa not in lexicon[word]:
                lexicon[word].append(ipa)
    return lexicon


def load_multi(path: str, lexicon: dict, seq_len: int = 32):
    """
    Load multi.tsv and produce (inputs_32bit, outputs_32bit) lists.

    Each row in multi.tsv is:
        word1 word2 ...  TAB  ipa1 ipa2 ...
    where '_' in the IPA column means "non-homograph, fill from lexicon".

    Returns:
        inputs  : list of int lists, length seq_len, each int is 32-bit
        outputs : list of int lists, length seq_len, each int is 0 or 0xffffffff
    """
    inputs_all  = []
    outputs_all = []

    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in tqdm.tqdm(reader):
            if len(row) < 2:
                continue
            words = row[0].strip().split()
            ipas  = row[1].strip().split()

            if len(words) != len(ipas):
                continue

            # Resolve '_' blanks from lexicon (these are guaranteed non-homographs)
            resolved_ipas = []
            for w, ipa in zip(words, ipas):
                if ipa == '_':
                    candidates = lexicon.get(w, [])
                    assert len(candidates) == 1, \
                        f"CRASH: '_' mapped to homograph or unknown word '{w}' with candidates {candidates}"
                    resolved_ipas.append(candidates[0])
                else:
                    resolved_ipas.append(ipa)

            # Build expanded sequence
            # homographs expand to N slots (one per candidate), non-homographs stay as 1 slot
            expanded_words = []   # word string per slot
            expanded_ipa   = []   # ipa string per slot
            expanded_label = []   # 0 or 0xffffffff per slot

            have_homograph = False

            for w, correct_ipa in zip(words, resolved_ipas):
                candidates = lexicon.get(w, [correct_ipa])
                if len(candidates) <= 1:
                    # non-homograph
                    expanded_words.append(w)
                    expanded_ipa.append(correct_ipa)
                    expanded_label.append(0xAAAAAAAA) # Doesn't matter (0xAAAAAAAA)
                elif len(candidates) > 1:
                    have_homograph = True
                    # homograph: one slot per candidate
                    for cand in candidates:
                        expanded_words.append(w)
                        expanded_ipa.append(cand)
                        expanded_label.append(0xFFFFFFFF if cand == correct_ipa else 0)

            if not have_homograph:
                continue

            # Hash each slot: word -> 16-bit, ipa -> 16-bit, combine to 32-bit
            combined = []
            for w, ipa in zip(expanded_words, expanded_ipa):
                w_tok   = hash_string_small(w)   & 0xFFFF
                ipa_tok = hash_string_small(ipa) & 0xFFFF
                combined.append(w_tok | (ipa_tok << 16))

            # Pad / truncate to seq_len
            combined_padded = pad_ints(combined, seq_len)[:seq_len]
            labels_padded   = pad_ints(expanded_label, seq_len)[:seq_len]

            inputs_all.append(combined_padded)
            outputs_all.append(labels_padded)

    return inputs_all, outputs_all


if __name__ == '__main__':
    import torch

    lexicon = load_lexicon('toy_dataset/lexicon.tsv')
    print("Lexicon:")
    for w, ipas in lexicon.items():
        tag = "(homograph)" if len(ipas) > 1 else ""
        print(f"  {w:12s} -> {ipas} {tag}")

    inputs, outputs = load_multi('toy_dataset/multi.tsv', lexicon, seq_len=16)

    print(f"\nLoaded {len(inputs)} training samples\n")
    for i, (inp, out) in enumerate(zip(inputs, outputs)):
        print(f"Sample {i}:")
        print(f"  input  (hex): {[hex(v) for v in inp]}")
        print(f"  output (hex): {[hex(v) for v in out]}")

        # Show as bit tensors (what the transformer sees)
        x = int_to_bits(inp)
        y = int_to_bits(out)
        print(f"  input  tensor shape: {x.shape}")
        print(f"  output tensor shape: {y.shape}")
