package noareg

import "strings"

func create2DFloat32FromTokens(tokens []uint32) (out [][]float32) {
	out = make([][]float32, len(tokens), len(tokens))
	for i := range tokens {
		out[i] = make([]float32, 32, 32)
		for j := 0; j < 32; j++ {
			if (tokens[i]>>j)&1 != 0 {
				out[i][j] = 1.
			}
		}
	}
	return out
}

func createTokensFrom2DFloat32(in [][]float32) (tokens []uint32) {
	for i := range in {
		var tok uint32
		for j := 0; j < 32; j++ {
			if in[i][j] > 0.5 {
				tok |= 1 << (j)
			}
		}
		tokens = append(tokens, tok)
	}
	return tokens
}

func TransformerInfer(transformer *NoaregTransformer, input []uint32) (output []uint32, err error) {
	input_floats := create2DFloat32FromTokens(input)

	output_floats, err := transformer.Forward(input_floats)
	if err != nil {
		return nil, err
	}

	return createTokensFrom2DFloat32(output_floats), nil
}

func TransformerInferFull(transformer *NoaregTransformer, detokenizer map[uint32]map[[2]uint32]string, try_word string) (out string, garbage [2]string, err error) {

	const limit = 16

	pseudo_data, garbage := Tokenize(detokenizer, try_word, limit)
	var move = limit
	for i := 0; i+limit < len(pseudo_data); i += move {

		//fmt.Println(pseudo_data[i:i+limit])
		output_data, err := TransformerInfer(transformer, pseudo_data[i:i+limit])
		if err != nil {
			return "", [2]string{}, err
		}

		//fmt.Println(output_data)
		for j := range output_data {
			if pseudo_data[i : i+limit][j] == 0 {
				move = limit
				break
			}
			var part = GetDetokenizerChoice(detokenizer, pseudo_data[i : i+limit][j], output_data[j])
			if strings.HasPrefix(part, "_") {
				if j == 0 {
					move = j + 1
				} else {
					move = j
				}
				break
			}
			out += part
			if strings.HasSuffix(part, "_") {
				move = j + 1
				break
			}
		}
	}
	return out, garbage, nil
}
