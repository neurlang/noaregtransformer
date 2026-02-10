package noareg

import "math/bits"

func hash(n uint32, s uint32, max_val uint32) uint32 {
	m := (n - s)
	m ^= (m << 2)
	m ^= (m << 3)
	m ^= (m >> 5)
	m ^= (m >> 7)
	m ^= (m << 11)
	m ^= (m << 13)
	m ^= (m >> 17)
	m ^= (m << 19)
	m += s
	return uint32((uint64(m) * uint64(max_val)) >> uint64(32))
}

func hashString(str string) uint32 {
	h := ^uint32(0)
	for i := 0; i < len(str); i++ {
		h = 1 + hash(h, uint32(str[i]), ^uint32(2))
	}
	return h
}

func MakeDetokenizer(data map[string][]string) (detokenizer map[uint32]map[[2]uint32]string) {
	detokenizer = make(map[uint32]map[[2]uint32]string)
	for k, w := range data {
		if len(w) == 0 {
			continue
		}
		hsk := hashString(k)
		detokenizer[hsk] = make(map[[2]uint32]string)
		if len(w) > 1 {
			for i, v := range w {
				detokenizer[hsk][[2]uint32{uint32(i), hashString(v)}] = v
			}
		} else {
			detokenizer[hsk][[2]uint32{0, ^uint32(0)}] = w[0]
		}
	}
	return detokenizer
}

func GetDetokenizerChoice(data map[uint32]map[[2]uint32]string, k, v uint32) (ret string) {
	if def, ok := data[k][[2]uint32{0, ^uint32(0)}]; ok {
		return def
	} else {
		var mind = ^uint32(0)
		var mini = ^uint32(0)
		for key, val := range data[k] {
			var d = uint32(bits.OnesCount32(key[1] ^ v))
			var i = key[0]
			if d < mind || ((d == mind) && (i < mini)) {
				mind = d
				mini = i
				ret = val
			}
		}
		return ret
	}
}
