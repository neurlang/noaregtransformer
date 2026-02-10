package noareg

import (
	"errors"
	"fmt"
	"math"
)

// ActivationType represents the type of activation function
type ActivationType int

const (
	ReLU ActivationType = iota
	Sigmoid
)

// PositionalEncoding implements sinusoidal positional encoding
type PositionalEncoding struct {
	pe [][]float32
}

// NewPositionalEncoding creates a new positional encoding matrix
func NewPositionalEncoding(embedDim, maxLen int) *PositionalEncoding {
	pe := make([][]float32, maxLen)
	for i := range pe {
		pe[i] = make([]float32, embedDim)
	}

	pos := 0
	for ; pos < maxLen; pos++ {
		for i := 0; i < embedDim; i += 2 {
			// Sin for even indices
			pe[pos][i] = float32(math.Sin(float64(pos) / math.Pow(10000, float64(i)/float64(embedDim))))

			// Cos for odd indices
			if i+1 < embedDim {
				pe[pos][i+1] = float32(math.Cos(float64(pos) / math.Pow(10000, float64(i)/float64(embedDim))))
			}
		}
	}

	return &PositionalEncoding{pe: pe}
}

// Apply adds positional encoding to the input
func (p *PositionalEncoding) Apply(x [][]float32) [][]float32 {
	seqLen := len(x)
	maxPos := len(p.pe)
	result := make([][]float32, seqLen)
	for i := range result {
		result[i] = make([]float32, len(x[0]))
		for j := range result[i] {
			if i < maxPos {
				result[i][j] = x[i][j] + p.pe[i][j]
			} else {
				// if sequence longer than maxLen, just copy (or you could wrap/clip)
				result[i][j] = x[i][j]
			}
		}
	}
	return result
}

// AttentionLayer represents a single attention layer
type AttentionLayer struct {
	// Combined QKV projection weights (32x96)
	inProjWeight [][]float32
	// Combined QKV projection bias (96)
	inProjBias []float32
	// Output projection weights (32x32)
	outProjWeight [][]float32
	// Output projection bias (32)
	outProjBias []float32
	headCount   int
	embedDim    int
}

// NewAttentionLayer creates a new attention layer
func NewAttentionLayer(inProjWeight, outProjWeight [][]float32,
	inProjBias, outProjBias []float32, headCount int) *AttentionLayer {
	return &AttentionLayer{
		inProjWeight:  inProjWeight,
		outProjWeight: outProjWeight,
		inProjBias:    inProjBias,
		outProjBias:   outProjBias,
		headCount:     headCount,
		embedDim:      len(outProjWeight),
	}
}

// splitHeads splits the input into multiple heads - FIXED VERSION
func (a *AttentionLayer) splitHeads(x [][]float32) [][][][]float32 {
	L := len(x)      // sequence length (16)
	E := a.embedDim  // embed dimension (32)
	H := a.headCount // number of heads (16)
	d := E / H       // dimension per head (2)
	B := 1           // batch size (fixed at 1)

	// Reshape: [L, E] -> [B, H, L, d]
	result := make([][][][]float32, B)
	for b := 0; b < B; b++ {
		result[b] = make([][][]float32, H)
		for h := 0; h < H; h++ {
			result[b][h] = make([][]float32, L)
			for pos := 0; pos < L; pos++ {
				result[b][h][pos] = make([]float32, d)
				// Copy data from input
				for i := 0; i < d; i++ {
					result[b][h][pos][i] = x[pos][h*d+i]
				}
			}
		}
	}
	return result
}

// combineHeads combines multiple heads back
func (a *AttentionLayer) combineHeads(heads [][][][]float32) [][]float32 {
	B := len(heads)
	H := a.headCount
	L := len(heads[0][0])
	d := a.embedDim / H
	E := a.embedDim

	result := make([][]float32, B*L)
	for i := range result {
		result[i] = make([]float32, E)
	}

	for b := 0; b < B; b++ {
		for pos := 0; pos < L; pos++ {
			for h := 0; h < H; h++ {
				for i := 0; i < d; i++ {
					result[b*L+pos][h*d+i] = heads[b][h][pos][i]
				}
			}
		}
	}
	return result
}

// scaledDotProductAttention computes attention - FIXED VERSION
func (a *AttentionLayer) scaledDotProductAttention(Q, K, V [][][][]float32) [][][][]float32 {
	B := len(Q)
	H := a.headCount
	L := len(Q[0][0]) // sequence length
	d := a.embedDim / a.headCount

	output := make([][][][]float32, B)
	for b := range output {
		output[b] = make([][][]float32, H)
		for h := range output[b] {
			output[b][h] = make([][]float32, L)
			for i := range output[b][h] {
				output[b][h][i] = make([]float32, d)
			}
		}
	}

	// Compute attention for each batch and head
	for b := 0; b < B; b++ {
		for h := 0; h < H; h++ {
			// Compute QK^T / sqrt(d)
			scores := make([][]float32, L)
			for i := range scores {
				scores[i] = make([]float32, L)
			}

			// Matrix multiplication: Q * K^T
			for i := 0; i < L; i++ {
				for j := 0; j < L; j++ {
					var sum float32
					for k := 0; k < d; k++ {
						sum += Q[b][h][i][k] * K[b][h][j][k] // FIXED: Added [b] index
					}
					scores[i][j] = sum / float32(math.Sqrt(float64(d)))
				}
			}

			// Apply softmax
			for i := 0; i < L; i++ {
				// Find max for numerical stability
				maxVal := scores[i][0]
				for j := 1; j < L; j++ {
					if scores[i][j] > maxVal {
						maxVal = scores[i][j]
					}
				}

				// Compute exp and sum
				var sum float32
				exps := make([]float32, L)
				for j := 0; j < L; j++ {
					exps[j] = float32(math.Exp(float64(scores[i][j] - maxVal)))
					sum += exps[j]
				}

				// Normalize
				for j := 0; j < L; j++ {
					scores[i][j] = exps[j] / sum
				}
			}

			// Apply attention to values
			for i := 0; i < L; i++ {
				for k := 0; k < d; k++ {
					var sum float32
					for j := 0; j < L; j++ {
						sum += scores[i][j] * V[b][h][j][k] // FIXED: Added [b] index
					}
					output[b][h][i][k] = sum
				}
			}
		}
	}

	return output
}

// Forward performs the attention forward pass - FIXED VERSION
func (a *AttentionLayer) Forward(x [][]float32) ([][]float32, error) {
	L := len(x) // Sequence length (should be 16)
	E := a.embedDim

	// Apply in_proj (combined QKV projection)
	inProj := make([][]float32, len(x))
	for i := range inProj {
		inProj[i] = make([]float32, 3*E)
		for j := 0; j < 3*E; j++ {
			var sum float32
			for k := 0; k < E; k++ {
				sum += x[i][k] * a.inProjWeight[k][j]
			}
			inProj[i][j] = sum + a.inProjBias[j]
		}
	}

	// Split into Q, K, V (each of size [L, E])
	Q := make([][]float32, L)
	K := make([][]float32, L)
	V := make([][]float32, L)
	for i := range Q {
		Q[i] = make([]float32, E)
		K[i] = make([]float32, E)
		V[i] = make([]float32, E)

		for j := 0; j < E; j++ {
			Q[i][j] = inProj[i][j]
			K[i][j] = inProj[i][E+j]
			V[i][j] = inProj[i][2*E+j]
		}
	}

	// Split heads
	QHeads := a.splitHeads(Q)
	KHeads := a.splitHeads(K)
	VHeads := a.splitHeads(V)

	// Compute scaled dot-product attention - FIXED: Pass the full arrays
	attnOutputHeads := a.scaledDotProductAttention(QHeads, KHeads, VHeads)

	// Combine heads
	attnOutput := a.combineHeads(attnOutputHeads)

	// Apply out_proj
	out := make([][]float32, len(attnOutput))
	for i := range out {
		out[i] = make([]float32, E)
		for j := 0; j < E; j++ {
			var sum float32
			for k := 0; k < E; k++ {
				sum += attnOutput[i][k] * a.outProjWeight[j][k]
			}
			out[i][j] = sum + a.outProjBias[j]
		}
	}

	return out, nil
}

// NoaregTransformer is the Go implementation of the transformer
type NoaregTransformer struct {
	embedDim  int
	headCount int
	numLayers int
	maxLen    int

	// Input projection
	inputProjWeight [][]float32 // 32x32 (onnx::MatMul_423)
	inputProjBias   []float32   // 32

	// Positional encoding
	posEncoder *PositionalEncoding

	// Attention layers
	attnLayers []*AttentionLayer

	// Activation types for each layer
	activations []ActivationType
}

// NewNoaregTransformer creates a new transformer instance
func NewNoaregTransformer(embedDim, headCount, maxLen, numLayers int) *NoaregTransformer {
	return &NoaregTransformer{
		embedDim:    embedDim,
		headCount:   headCount,
		numLayers:   numLayers,
		maxLen:      maxLen,
		posEncoder:  NewPositionalEncoding(embedDim, maxLen),
		attnLayers:  make([]*AttentionLayer, numLayers),
		activations: make([]ActivationType, numLayers),
	}
}

// InitializeParameters initializes the transformer with trained parameters
func (t *NoaregTransformer) InitializeParameters(
	inputProjWeight []float32, // Flattened 32x32 matrix
	inputProjBias []float32,
	attnInProjWeights [][][]float32, // [4][32][96]
	attnOutProjWeights [][][]float32, // [4][32][32]
	attnInProjBiases [][]float32, // [4][96]
	attnOutProjBiases [][]float32) error { // [4][32]

	// Initialize input projection bias
	if len(inputProjBias) != t.embedDim {
		return errors.New("input_proj.bias size mismatch")
	}
	t.inputProjBias = inputProjBias

	// Initialize input projection weight (reshape from flattened)
	if len(inputProjWeight) != t.embedDim*t.embedDim {
		return errors.New("input_proj.weight size mismatch")
	}
	t.inputProjWeight = make([][]float32, t.embedDim)
	for i := range t.inputProjWeight {
		t.inputProjWeight[i] = make([]float32, t.embedDim)
		for j := range t.inputProjWeight[i] {
			t.inputProjWeight[i][j] = inputProjWeight[i*t.embedDim+j]
		}
	}

	// Initialize attention layers
	for i := 0; i < t.numLayers; i++ {
		if i == t.numLayers-1 {
			t.activations[i] = Sigmoid
		} else {
			t.activations[i] = ReLU
		}

		// Verify sizes
		if len(attnInProjWeights[i]) != t.embedDim || len(attnInProjWeights[i][0]) != 3*t.embedDim {
			return fmt.Errorf("attn_layers.%d.in_proj_weight size mismatch", i)
		}
		if len(attnOutProjWeights[i]) != t.embedDim || len(attnOutProjWeights[i][0]) != t.embedDim {
			return fmt.Errorf("attn_layers.%d.out_proj_weight size mismatch", i)
		}
		if len(attnInProjBiases[i]) != 3*t.embedDim {
			return fmt.Errorf("attn_layers.%d.in_proj_bias size mismatch", i)
		}
		if len(attnOutProjBiases[i]) != t.embedDim {
			return fmt.Errorf("attn_layers.%d.out_proj_bias size mismatch", i)
		}

		// Create attention layer
		t.attnLayers[i] = NewAttentionLayer(
			attnInProjWeights[i],  // 32x96
			attnOutProjWeights[i], // 32x32
			attnInProjBiases[i],   // 96
			attnOutProjBiases[i],  // 32
			t.headCount,
		)
	}

	return nil
}

// applyActivation applies the activation function
func applyActivation(x [][]float32, activation ActivationType) [][]float32 {
	result := make([][]float32, len(x))
	for i := range result {
		result[i] = make([]float32, len(x[0]))
		for j := range result[i] {
			switch activation {
			case ReLU:
				if x[i][j] > 0 {
					result[i][j] = x[i][j]
				} else {
					result[i][j] = 0
				}
			case Sigmoid:
				result[i][j] = float32(1.0 / (1.0 + math.Exp(float64(-x[i][j]))))
			}
		}
	}
	return result
}

// matrixAdd adds two matrices
func matrixAdd(a, b [][]float32) [][]float32 {
	result := make([][]float32, len(a))
	for i := range result {
		result[i] = make([]float32, len(a[0]))
		for j := range result[i] {
			result[i][j] = a[i][j] + b[i][j]
		}
	}
	return result
}

// Forward performs the transformer forward pass
func (t *NoaregTransformer) Forward(x [][]float32) ([][]float32, error) {
	B := len(x)
	E := t.embedDim

	// Input projection + activation
	proj := make([][]float32, B)
	for i := range proj {
		proj[i] = make([]float32, E)
		for j := 0; j < E; j++ {
			var sum float32
			for k := 0; k < E; k++ {
				sum += x[i][k] * t.inputProjWeight[k][j]
			}
			proj[i][j] = sum + t.inputProjBias[j]
		}
	}

	// Apply ReLU activation
	proj = applyActivation(proj, ReLU)

	// Add positional encoding
	proj = t.posEncoder.Apply(proj)

	// Apply attention layers
	current := proj
	for i := 0; i < t.numLayers; i++ {
		// Attention
		attnOut, err := t.attnLayers[i].Forward(current)
		if err != nil {
			return nil, fmt.Errorf("attention layer %d failed: %w", i, err)
		}

		// Residual connection
		current = matrixAdd(current, attnOut)

		// Activation
		current = applyActivation(current, t.activations[i])
	}

	return current, nil
}
