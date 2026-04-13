import csv
import tqdm
from noareg_traindata import hashtronhash, bits_to_int, pad_ints


def hash_string_small(string: str) -> int:
    h = 0xFFFF
    x = string.encode(encoding="utf-8")
    for c in x:
        h = 1 + hashtronhash(h, int(c) & 0xFFFF, 0xFFFD)
    return h


def load_lexicon(path: str) -> dict:
    lexicon = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 2:
                continue
            word = row[0].strip()
            ipa = row[1].strip()
            if not word:
                continue
            if word not in lexicon:
                lexicon[word] = []
            if ipa not in lexicon[word]:
                lexicon[word].append(ipa)
    return lexicon


def load_multi(path: str, lexicon: dict, seq_len: int = 32):
    inputs_all = []
    outputs_all = []

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in tqdm.tqdm(reader):
            if len(row) < 2:
                continue
            words = row[0].strip().split()
            ipas = row[1].strip().split()

            if len(words) != len(ipas):
                continue

            resolved_ipas = []
            for w, ipa in zip(words, ipas):
                if ipa == "_":
                    candidates = lexicon.get(w, [])
                    assert len(candidates) == 1
                    resolved_ipas.append(candidates[0])
                else:
                    resolved_ipas.append(ipa)

            expanded_words = []
            expanded_ipa = []
            expanded_label = []
            expanded_wayness = []

            have_homograph = False

            for w, correct_ipa in zip(words, resolved_ipas):
                candidates = lexicon.get(w, [correct_ipa])
                if len(candidates) <= 1:
                    expanded_words.append(w)
                    expanded_ipa.append(correct_ipa)
                    expanded_label.append(0xAAAAAAAA)
                    expanded_wayness.append(1)
                elif len(candidates) > 1:
                    have_homograph = True
                    # Sort candidates by hash for determinism (matches Go behavior)
                    sorted_candidates = sorted(candidates, key=hash_string_small)
                    for cand in sorted_candidates:
                        expanded_words.append(w)
                        expanded_ipa.append(cand)
                        expanded_label.append(0xFFFFFFFF if cand == correct_ipa else 0)
                        expanded_wayness.append(len(sorted_candidates))

            if not have_homograph:
                continue

            # Build combined with galloping-aware padding
            combined = []
            labels = []

            i = 0
            while i < len(expanded_words):
                pos_in_window = len(combined) % seq_len
                remaining_in_window = seq_len - pos_in_window
                wayness = expanded_wayness[i]

                if wayness > remaining_in_window:
                    # Pad to complete current window with zeros for boundary-spanning homographs
                    padding_needed = remaining_in_window
                    combined.extend([0] * padding_needed)
                    labels.extend([0xAAAAAAAA] * padding_needed)
                    # Don't increment i - same word will be processed in next window iteration
                else:
                    w_tok = hash_string_small(expanded_words[i]) & 0xFFFF
                    ipa_tok = hash_string_small(expanded_ipa[i]) & 0xFFFF
                    combined.append(w_tok | (ipa_tok << 16))
                    labels.append(expanded_label[i])
                    i += 1

            # Pad final to make length divisible by seq_len with zeros
            while len(combined) % seq_len != 0:
                combined.append(0)
                labels.append(0xAAAAAAAA)

            # Chunk into windows of seq_len
            start = 0
            while start < len(combined):
                end = start + seq_len

                chunk_combined = pad_ints(combined[start:end], seq_len)[:seq_len]
                chunk_labels = pad_ints(labels[start:end], seq_len)[:seq_len]

                inputs_all.append(chunk_combined)
                outputs_all.append(chunk_labels)

                start += seq_len

    return inputs_all, outputs_all


if __name__ == "__main__":
    import torch

    lexicon = load_lexicon("toy_dataset/lexicon.tsv")
    print("Lexicon:")
    for w, ipas in lexicon.items():
        tag = "(homograph)" if len(ipas) > 1 else ""
        print(f"  {w:12s} -> {ipas} {tag}")

    inputs, outputs = load_multi("toy_dataset/multi.tsv", lexicon, seq_len=16)

    print(f"\nLoaded {len(inputs)} training samples\n")
    for i, (inp, out) in enumerate(zip(inputs, outputs)):
        print(f"Sample {i}:")
        print(f"  input  (hex): {[hex(v) for v in inp]}")
        print(f"  output (hex): {[hex(v) for v in out]}")

        x = int_to_bits(inp)
        y = int_to_bits(out)
        print(f"  input  tensor shape: {x.shape}")
        print(f"  output tensor shape: {y.shape}")
