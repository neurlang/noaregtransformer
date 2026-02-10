# Non-Autoregressive Transformer for Phonetic Transcription

A lightweight transformer model that performs non-autoregressive sequence-to-sequence prediction, specifically designed for phonetic transcription tasks (e.g., Japanese text to IPA phonetics).

## Overview

This project implements a custom transformer architecture that predicts entire output sequences in parallel rather than autoregressively. The model uses multi-head attention with residual connections and can be trained in Python (PyTorch) and deployed in Go for fast inference.

**Key Features:**
- Non-autoregressive parallel prediction
- Multi-head self-attention (16 heads, 32-dim embeddings)
- Compact binary model format with zlib compression
- Pure Go inference implementation (no external ML dependencies)
- Hash-based tokenization for efficient vocabulary handling

## Architecture

- **Input/Output:** 16-token sequences encoded as 32-bit integers
- **Embedding:** 32 dimensions
- **Attention:** 16 heads across 4 transformer layers
- **Activation:** ReLU for intermediate layers, Sigmoid for output
- **Training:** Binary cross-entropy loss with masked positions

The model converts input tokens to bit vectors (32 bits per token), processes them through attention layers with positional encoding, and outputs the proposed token for each position directly.

## Project Structure

```
.
├── noareg_main.py              # Training script
├── noareg_transformer.py       # PyTorch model definition
├── noareg_traindata.py         # Data loading and preprocessing
└── go/
    ├── cmd/main/main.go        # Go inference example
    └── noareg/
        ├── transformer.go       # Core transformer implementation
        ├── transformer_file.go  # Binary model loader
        ├── transformer_infer.go # Inference utilities
        ├── tokenizer.go         # Input tokenization
        ├── detokenizer.go       # Output decoding
        └── tensor_file.go       # Binary format reader
```

## Usage

### Training (Python)

```python
from noareg_transformer import NoaregTransformer
import torch

# Create model
model = NoaregTransformer(
    embed_dimension=32,
    head_count=16,
    max_len=100,
    num_layers=4
)

# Train on your data
# ... training loop ...

# Export to binary format
model.export_to_binary("weights.bin")
```

The training script (`noareg_main.py`) expects TSV data with paired input/output sequences and trains until convergence, automatically exporting the best model.

### Inference (Go)

```go
package main

import (
    "github.com/neurlang/noaregtransformer/go/noareg"
)

func main() {
    // Load model from binary file
    tensors, _ := noareg.ReadTensors(file)
    transformer := noareg.NewNoaregTransformer(32, 16, 100, 4)
    noareg.LoadTransformerFile(transformer, tensors)
    
    // Create detokenizer with vocabulary
    detok := noareg.MakeDetokenizer(vocabulary)
    
    // Run inference
    output, _, _ := noareg.TransformerInferFull(
        transformer, detok, "input text")
}
```

## Binary Format

The model uses a custom binary format for efficient loading:
- Magic bytes: `GOBF`
- Version: int32
- Tensors: name, dtype, shape, raw float32 data
- Optional zlib compression (`.bin.zlib`)

This format allows the Go implementation to load models without external dependencies.

## Requirements

**Python:**
- PyTorch
- NumPy

**Go:**
- Go 1.22.4+
- No external dependencies (pure Go)

## Performance

The non-autoregressive design enables parallel prediction of all output tokens simultaneously, making inference significantly faster than autoregressive models for fixed-length sequences. The Go implementation provides additional performance benefits for production deployment.

## License

See project license file for details.
