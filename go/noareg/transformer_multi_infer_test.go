package noareg

import (
	"compress/zlib"
	"os"
	"strings"
	"testing"
)

func countHomographSlots(words [][]string) int {
	count := 0
	for _, wordEntry := range words {
		if len(wordEntry) > 2 {
			count++
		}
	}
	return count
}

func calcWrongHomographs(result string, tc struct {
	name     string
	input    [][]string
	expected []string
}) int {
	resultWords := strings.Fields(result)
	wrong := 0
	ri := 0
	for _, wordEntry := range tc.input {
		if len(wordEntry) < 2 {
			continue
		}
		if ri >= len(resultWords) {
			break
		}
		// only score homograph slots
		if len(wordEntry) > 2 {
			exp := wordEntry[len(wordEntry)-1] // last element = correct IPA
			if resultWords[ri] != exp {
				wrong++
			}
		}
		ri++
	}
	return wrong
}

func loadTransformerForTest(t *testing.T) *NoaregTransformer {
	t.Helper()
	// Try relative path from workspace root
	paths := []string{
		"../../weights_multi.bin.zlib",
		"weights_multi.bin.zlib",
	}
	var file *os.File
	var err error
	for _, p := range paths {
		file, err = os.Open(p)
		if err == nil {
			break
		}
	}
	if err != nil {
		t.Skip("weights_multi.bin.zlib not found, skipping")
	}
	defer file.Close()

	zlibReader, err := zlib.NewReader(file)
	if err != nil {
		t.Fatal(err)
	}
	defer zlibReader.Close()

	tensors, err := ReadTensors(zlibReader)
	if err != nil {
		t.Fatal(err)
	}

	transformer := NewNoaregTransformer(32, 16, 100, 4)
	if err := LoadTransformerFile(transformer, tensors); err != nil {
		t.Fatal(err)
	}
	return transformer
}

// TestMultiInfer_IPA_Dedup verifies that duplicate IPA candidates are collapsed.
// FOO -> BAR, BAR must give same result as FOO -> BAR (single candidate).
func TestMultiInfer_IPA_Dedup(t *testing.T) {
	transformer := loadTransformerForTest(t)

	// שב has candidates ʃˈev, ʃˈav — run with duplicate ʃˈev
	withDup := [][]string{
		{"זהו", "zˈehu"},
		{"שב", "ʃˈev", "ʃˈav", "ʃˈev"}, // ʃˈev duplicated
	}
	withoutDup := [][]string{
		{"זהו", "zˈehu"},
		{"שב", "ʃˈev", "ʃˈav"},
	}

	r1, err := MultiWordInferFull(transformer, withDup)
	if err != nil {
		t.Fatal(err)
	}
	r2, err := MultiWordInferFull(transformer, withoutDup)
	if err != nil {
		t.Fatal(err)
	}
	if r1 != r2 {
		t.Errorf("dedup sensitivity: with-dup=%q without-dup=%q", r1, r2)
	} else {
		t.Logf("dedup OK: %q == %q", r1, r2)
	}
}

// TestMultiInfer_IPA_OrderIndependence verifies that candidate order doesn't affect result.
// FOO -> BAR, BAZ must give same result as FOO -> BAZ, BAR.
func TestMultiInfer_IPA_OrderIndependence(t *testing.T) {
	transformer := loadTransformerForTest(t)

	orderA := [][]string{
		{"זהו", "zˈehu"},
		{"שב", "ʃˈev", "ʃˈav"},
	}
	orderB := [][]string{
		{"זהו", "zˈehu"},
		{"שב", "ʃˈav", "ʃˈev"}, // reversed
	}

	r1, err := MultiWordInferFull(transformer, orderA)
	if err != nil {
		t.Fatal(err)
	}
	r2, err := MultiWordInferFull(transformer, orderB)
	if err != nil {
		t.Fatal(err)
	}
	if r1 != r2 {
		t.Errorf("order sensitivity: A=%q B=%q", r1, r2)
	} else {
		t.Logf("order-independent OK: %q == %q", r1, r2)
	}
}

// TestMultiInfer_SeqLen verifies seq_len=32 matches trainer (not 16).
// A sentence with >16 expanded slots must still produce output.
func TestMultiInfer_SeqLen(t *testing.T) {
	transformer := loadTransformerForTest(t)

	// Build a sentence with many homographs to exceed 16 slots
	// Each homograph expands to 2 slots; 9 homographs = 18 slots > 16
	input := [][]string{
		{"שב", "ʃˈev", "ʃˈav"},
		{"כהן", "kohˈen", "χohˈen"},
		{"את", "ʔˈet", "ʔˈat"},
		{"על", "ʔˈal", "ʔˈol"},
		{"של", "ʃˈel", "ʃˈal"},
		{"אל", "ʔˈel", "ʔˈal"},
		{"לא", "lˈo", "lˈa"},
		{"אבל", "ʔavˈal", "ʔˈevel", "ʔavˈel"},
		{"בדרך", "bedˈeʁeχ", "badˈeʁeχ", "vadˈeʁeχ", "badaʁˈeχ"},
	}
	result, err := MultiWordInferFull(transformer, input)
	if err != nil {
		t.Fatalf("seq_len test failed: %v", err)
	}
	words := strings.Fields(result)
	if len(words) != len(input) {
		t.Errorf("expected %d output words, got %d: %q", len(input), len(words), result)
	} else {
		t.Logf("seq_len OK: %q", result)
	}
}

// TestMultiInfer_Galloping verifies that sentences with expanded slots > multiSeqLen
// are handled by galloping (sliding window), not truncated.
func TestMultiInfer_Galloping(t *testing.T) {
	transformer := loadTransformerForTest(t)

	// 20 homographs × 2 slots = 40 slots > 32 — requires galloping
	homograph := []string{"שב", "ʃˈev", "ʃˈav"}
	input := make([][]string, 20)
	for i := range input {
		input[i] = homograph
	}

	result, err := MultiWordInferFull(transformer, input)
	if err != nil {
		t.Fatalf("galloping test failed: %v", err)
	}
	words := strings.Fields(result)
	if len(words) != 20 {
		t.Errorf("expected 20 output words, got %d: %q", len(words), result)
	} else {
		t.Logf("galloping OK: %d words", len(words))
	}
}

// TestMultiInfer_Hebrew runs the 10 Hebrew sentences and reports homograph WER.
func TestMultiInfer_Hebrew(t *testing.T) {
	transformer := loadTransformerForTest(t)

	// Each entry: [word, ipa_candidates..., correct_ipa_last]
	// For non-homographs: [word, single_ipa]
	// For homographs: [word, cand1, cand2, ..., correct] — correct is last
	// NOTE: correct IPA is the last element; all candidates including correct are passed to inference.
	type testCase struct {
		name    string
		input   [][]string // [word, cand..., correct] — correct = last
		correct []string   // expected IPA per word (for WER)
	}

	cases := []testCase{
		{
			// זהו זחאלקה שב במקומך תודה רבה
			// _ _ ʃˈev bimkomχˈa todˈa ʁabˈa
			name: "s1_shav",
			input: [][]string{
				{"זהו", "zˈehu"},
				{"זחאלקה", "zaχalkˈa"},
				{"שב", "ʃˈav", "ʃˈev"},               // correct: ʃˈev
				{"במקומך", "bimkomˈeχ", "bimkomχˈa"}, // correct: bimkomχˈa
				{"תודה", "todˈe", "todˈa"},           // correct: todˈa
				{"רבה", "ʁˈava", "ʁabˈa"},            // correct: ʁabˈa
			},
			correct: []string{"zˈehu", "zaχalkˈa", "ʃˈev", "bimkomχˈa", "todˈa", "ʁabˈa"},
		},
		{
			// אחמד תביא גם איזה פסוק — all _ (no homographs after dedup)
			name: "s2_no_homograph",
			input: [][]string{
				{"אחמד", "ʔaχmˈad"},
				{"תביא", "tavˈi"},
				{"גם", "gˈam"},
				{"איזה", "ʔˈejze"},
				{"פסוק", "pasˈuk"},
			},
			correct: []string{"ʔaχmˈad", "tavˈi", "gˈam", "ʔˈejze", "pasˈuk"},
		},
		{
			// אחמד תביא איזה פסוק — all _ (no homographs)
			name: "s3_no_homograph",
			input: [][]string{
				{"אחמד", "ʔaχmˈad"},
				{"תביא", "tavˈi"},
				{"איזה", "ʔˈejze"},
				{"פסוק", "pasˈuk"},
			},
			correct: []string{"ʔaχmˈad", "tavˈi", "ʔˈejze", "pasˈuk"},
		},
		{
			// יש ביניהם מלחמה בנושאים האלה
			// _ _ _ banosʔˈim haʔˈele
			name: "s4_banosaim_haele",
			input: [][]string{
				{"יש", "jˈeʃ"},
				{"ביניהם", "benehˈem"},
				{"מלחמה", "milχamˈa"},
				{"בנושאים", "benosʔˈim", "banosʔˈim"},     // correct: banosʔˈim
				{"האלה", "haʔelˈa", "haʔalˈa", "haʔˈele"}, // correct: haʔˈele
			},
			correct: []string{"jˈeʃ", "benehˈem", "milχamˈa", "banosʔˈim", "haʔˈele"},
		},
		{
			// בדרך כלל מנהלים משא ומתן
			// bedˈeʁeχ klˈal _ _ _
			name: "s5_bederech_klal",
			input: [][]string{
				{"בדרך", "badaʁˈeχ", "vadˈeʁeχ", "badˈeʁeχ", "bedˈeʁeχ"}, // correct: bedˈeʁeχ
				{"כלל", "kalˈal", "klˈal"},                               // correct: klˈal
				{"מנהלים", "menahalˈim"},
				{"משא", "masˈa"},
				{"ומתן", "umatˈan"},
			},
			correct: []string{"bedˈeʁeχ", "klˈal", "menahalˈim", "masˈa", "umatˈan"},
		},
		{
			// גם אימא יהודייה אבל גם אימא פלסטינית
			// _ _ _ ʔavˈal _ _ falastˈinit
			name: "s6_ima_falastin",
			input: [][]string{
				{"גם", "gˈam"},
				{"אימא", "ʔimˈa"},
				{"יהודייה", "jehudijˈa"},
				{"אבל", "ʔˈevel", "ʔavˈel", "ʔavˈal"}, // correct: ʔavˈal
				{"גם", "gˈam"},
				{"אימא", "ʔimˈa"},
				{"פלסטינית", "falastinˈit", "falastˈinit"}, // correct: falastˈinit
			},
			correct: []string{"gˈam", "ʔimˈa", "jehudijˈa", "ʔavˈal", "gˈam", "ʔimˈa", "falastˈinit"},
		},
		{
			// אבל אל תשללו מאחרים את אותה הזכות לתבוע את אותה דרישה
			// ʔavˈal ʔˈal _ meʔaχeʁˈim ʔˈet _ _ _ ʔˈet _ _
			name: "s7_aval_al_et",
			input: [][]string{
				{"אבל", "ʔˈevel", "ʔavˈel", "ʔavˈal"}, // correct: ʔavˈal
				{"אל", "ʔˈel", "ʔˈal"},                // correct: ʔˈal
				{"תשללו", "tiʃllˈu"},
				{"מאחרים", "meʔaχaʁˈim", "meʔaχeʁˈim"}, // correct: meʔaχeʁˈim
				{"את", "ʔˈat", "ʔˈet"},                 // correct: ʔˈet
				{"אותה", "ʔotˈa"},
				{"הזכות", "hazχˈut"},
				{"לתבוע", "litbˈoa"},
				{"את", "ʔˈat", "ʔˈet"}, // correct: ʔˈet
				{"אותה", "ʔotˈa"},
				{"דרישה", "dʁiʃˈa"},
			},
			correct: []string{"ʔavˈal", "ʔˈal", "tiʃllˈu", "meʔaχeʁˈim", "ʔˈet", "ʔotˈa", "hazχˈut", "litbˈoa", "ʔˈet", "ʔotˈa", "dʁiʃˈa"},
		},
		{
			// אתה יכול לנסות להשמיע טיעונים
			// ʔatˈa _ _ _ _
			name: "s8_ata_yachol",
			input: [][]string{
				{"אתה", "ʔatˈa"},
				{"יכול", "jaχˈol"},
				{"לנסות", "lenasˈot"},
				{"להשמיע", "lehaʃmˈia"},
				{"טיעונים", "tiʔunˈim"},
			},
			correct: []string{"ʔatˈa", "jaχˈol", "lenasˈot", "lehaʃmˈia", "tiʔunˈim"},
		},
		{
			// אף אחד מאיתנו לא אומר שכל משפחת כהן הם רוצחים למרות שיש המון פושעים במשפחת כהן
			// _ ʔeχˈad _ lˈo ʔomˈeʁ ʃekˈol miʃpˈaχat kohˈen _ _ lamʁˈot ʃejˈeʃ _ _ _ kohˈen
			name: "s9_cohen_long",
			input: [][]string{
				{"אף", "ʔˈaf"},
				{"אחד", "ʔeχˈad"},
				{"מאיתנו", "meʔitˈanu"},
				{"לא", "lˈa", "lˈo"}, // correct: lˈo
				{"אומר", "ʔomˈeʁ"},
				{"שכל", "sˈeχel", "ʃaχˈal", "ʃekˈol"}, // correct: ʃekˈol
				{"משפחת", "miʃpχˈot", "miʃpˈaχat"},    // correct: miʃpˈaχat
				{"כהן", "χohˈen", "kohˈen"},           // correct: kohˈen
				{"הם", "hˈem"},
				{"רוצחים", "ʁotsχˈim"},
				{"למרות", "lemaʁˈut", "lamʁˈot"},      // correct: lamʁˈot
				{"שיש", "ʃˈajiʃ", "ʃajˈiʃ", "ʃejˈeʃ"}, // correct: ʃejˈeʃ
				{"המון", "hamˈon"},
				{"פושעים", "poʃʔˈim"},
				{"במשפחת", "bemiʃpˈaχat"},
				{"כהן", "χohˈen", "kohˈen"}, // correct: kohˈen
			},
			correct: []string{"ʔˈaf", "ʔeχˈad", "meʔitˈanu", "lˈo", "ʔomˈeʁ", "ʃekˈol", "miʃpˈaχat", "kohˈen", "hˈem", "ʁotsχˈim", "lamʁˈot", "ʃejˈeʃ", "hamˈon", "poʃʔˈim", "bemiʃpˈaχat", "kohˈen"},
		},
		{
			// תגיד משהו על הרצח של השוטרים
			// _ mˈaʃehu ʔˈal _ ʃˈel _
			name: "s10_tagid_shel",
			input: [][]string{
				{"תגיד", "tagˈid"},
				{"משהו", "mˈaʃehu"},
				{"על", "ʔˈol", "ʔˈal"}, // correct: ʔˈal
				{"הרצח", "haʁˈetsaχ"},
				{"של", "ʃˈal", "ʃˈel"}, // correct: ʃˈel
				{"השוטרים", "haʃotʁˈim"},
			},
			correct: []string{"tagˈid", "mˈaʃehu", "ʔˈal", "haʁˈetsaχ", "ʃˈel", "haʃotʁˈim"},
		},
	}

	totalHomographs := 0
	totalWrong := 0

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			result, err := MultiWordInferFull(transformer, tc.input)
			if err != nil {
				t.Fatalf("error: %v", err)
			}
			got := strings.Fields(result)

			wrong := 0
			homographs := 0
			for i, wordEntry := range tc.input {
				isHomograph := len(wordEntry) > 2
				if isHomograph {
					homographs++
					if i < len(got) && i < len(tc.correct) {
						if got[i] != tc.correct[i] {
							wrong++
							t.Logf("  WRONG [%s]: got %q want %q", wordEntry[0], got[i], tc.correct[i])
						}
					}
				}
			}
			totalHomographs += homographs
			totalWrong += wrong

			t.Logf("%s: result=%q wrong=%d/%d homographs", tc.name, result, wrong, homographs)
		})
	}

	wer := 0.0
	if totalHomographs > 0 {
		wer = float64(totalHomographs-totalWrong) / float64(totalHomographs) * 100
	}
	t.Logf("=== Homograph accuracy: %d/%d correct (%.1f%%)", totalHomographs-totalWrong, totalHomographs, wer)
}
