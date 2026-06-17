package noareg

func tokenizePossibilities(str string) (out []uint32) {
	h := ^uint32(0)
	for i := 0; i < len(str); i++ {
		h = 1 + hash(h, uint32(str[i]), ^uint32(2))
		out = append(out, h)
	}
	return out
}

func Tokenize(detokenizer map[uint32]map[[2]uint32]string, str string, pad int) (out []uint32, punct [2]string) {
	out, _, punct = Tokenize2(detokenizer, str, pad)
	return out, punct
}

func Tokenize2(detokenizer map[uint32]map[[2]uint32]string, str string, pad int) (out []uint32, chunks []string, punct [2]string) {
	var half = len(str) / 2
main:
	for len(str) > 0 {
		var row = tokenizePossibilities(str)
		for i := range row {
			if _, ok := detokenizer[row[len(row)-i-1]]; ok {
				out = append(out, row[len(row)-i-1])
				chunks = append(chunks, str[:len(str)-i])
				str = str[len(str)-i:]
				continue main
			}
		}
		if len(str) > half {
			punct[0] += string([]rune(str)[0])
		} else {
			punct[1] += string([]rune(str)[0])
		}
		str = string([]rune(str)[1:])
	}
	for i := 0; i < pad; i++ {
		out = append(out, 0)
	}
	return out, chunks, punct
}
