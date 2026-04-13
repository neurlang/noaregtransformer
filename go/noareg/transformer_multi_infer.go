package noareg

import (
	"sort"
	"strings"
)

const multiSeqLen = 32

// hashStringSmall mirrors Python's hash_string_small: 16-bit hash
func hashStringSmall(str string) uint32 {
	h := uint32(0xFFFF)
	for i := 0; i < len(str); i++ {
		h = 1 + hash(h, uint32(str[i]), 0xFFFD)
	}
	return h
}

// buildSlots expands inputs into a flat token list.
// Each entry in inputs is [word, ipa_candidate...].
// Non-homographs (single "_") produce 1 slot with ipaHash=0.
// Homographs produce N slots, IPA deduped and sorted by hash (matches Python trainer).
func buildSlots(inputs [][]string) (tokens []uint32, slotWord []int, slotIPA []string) {
	for wordIdx, wordEntry := range inputs {
		if len(wordEntry) < 2 {
			continue
		}
		word := wordEntry[0]
		wordHash := hashStringSmall(word) & 0xFFFF

		candidates := wordEntry[1:]

		if len(candidates) == 1 && candidates[0] == "_" {
			// non-homograph: single slot, ipa bits = 0
			tokens = append(tokens, wordHash)
			slotWord = append(slotWord, wordIdx)
			slotIPA = append(slotIPA, "_")
			continue
		}

		// Dedup IPA candidates
		seen := make(map[string]struct{})
		var unique []string
		for _, ipa := range candidates {
			if _, ok := seen[ipa]; !ok {
				seen[ipa] = struct{}{}
				unique = append(unique, ipa)
			}
		}

		// Sort by IPA hash for determinism — matches Python trainer sort
		sort.Slice(unique, func(i, j int) bool {
			return hashStringSmall(unique[i]) < hashStringSmall(unique[j])
		})

		for _, ipa := range unique {
			ipaHash := hashStringSmall(ipa) & 0xFFFF
			tokens = append(tokens, wordHash|(ipaHash<<16))
			slotWord = append(slotWord, wordIdx)
			slotIPA = append(slotIPA, ipa)
		}
	}
	return
}

type ipaCandidate struct {
	ipa   string
	score int
}

// MultiWordInferFull disambiguates homographs in a sentence.
// inputs: each entry is [word, ipa_candidate...].
//   - [word, "_"]          → non-homograph, pass-through
//   - [word, ipa1, ipa2…]  → homograph, model picks best
//
// Returns space-joined IPA string for the whole sentence.
func MultiWordInferFull(transformer *NoaregTransformer, inputs [][]string) (string, error) {
	tokens, slotWord, slotIPA := buildSlots(inputs)

	best := make(map[int]ipaCandidate) // wordIdx -> best candidate so far

	// Galop through token sequence in windows of multiSeqLen
	for start := 0; start < len(tokens); start += multiSeqLen {
		end := start + multiSeqLen
		if end > len(tokens) {
			end = len(tokens)
		}
		window := tokens[start:end]

		padded := make([]uint32, multiSeqLen)
		copy(padded, window)

		output, err := TransformerInfer(transformer, padded)
		if err != nil {
			return "", err
		}

		for i, tok := range window {
			if tok == 0 {
				continue // padding slot
			}
			wordIdx := slotWord[start+i]
			ipa := slotIPA[start+i]
			if ipa == "_" {
				continue // non-homograph, handled below
			}
			score := onesCount32(output[i])
			if prev, ok := best[wordIdx]; !ok || score > prev.score {
				best[wordIdx] = ipaCandidate{ipa, score}
			}
		}
	}

	// Build result
	result := make([]string, len(inputs))
	for i, wordEntry := range inputs {
		if len(wordEntry) < 2 {
			result[i] = wordEntry[0]
			continue
		}
		if c, ok := best[i]; ok {
			result[i] = c.ipa
		} else {
			// non-homograph: use the provided IPA (not "_")
			ipa := wordEntry[1]
			if ipa == "_" {
				result[i] = wordEntry[0] // unknown word fallback
			} else {
				result[i] = ipa
			}
		}
	}

	return strings.Join(result, " "), nil
}

func onesCount32(v uint32) int {
	count := 0
	for i := 0; i < 32; i++ {
		if (v>>i)&1 == 1 {
			count++
		}
	}
	return count
}
