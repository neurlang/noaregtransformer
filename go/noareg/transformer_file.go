package noareg

import "errors"

func LoadTransformerFile(transformer *NoaregTransformer, tensors []TensorInfo) error {
	const endian = true
	if transformer == nil {
		return errors.New("no_transformer")
	}
	if len(tensors) == 0 {
		return errors.New("no_tensors")
	}

	// Create a map for easy tensor access
	tensorMap := make(map[string]*TensorInfo)
	for i := range tensors {
		tensorMap[tensors[i].Name] = &tensors[i]
	}

	// Extract and prepare tensors
	// 1. Input projection
	inputProjBias := bytesToFloat32Slice(tensorMap["input_proj.bias"].Data, endian)
	inputProjWeight := bytesToFloat32Slice(tensorMap["val_0"].Data, endian)

	// 2. Prepare attention layer parameters
	attnInProjWeights := make([][][]float32, 4)
	attnOutProjWeights := make([][][]float32, 4)
	attnInProjBiases := make([][]float32, 4)
	attnOutProjBiases := make([][]float32, 4)

	// Attention layer 0
	attnInProjWeights[0] = create2DFloat32FromBytes(tensorMap["val_15"].Data, 32, 96, endian)
	attnOutProjWeights[0] = create2DFloat32FromBytes(tensorMap["attn_layers.0.out_proj.weight"].Data, 32, 32, endian)
	attnInProjBiases[0] = bytesToFloat32Slice(tensorMap["attn_layers.0.in_proj_bias"].Data, endian)
	attnOutProjBiases[0] = bytesToFloat32Slice(tensorMap["attn_layers.0.out_proj.bias"].Data, endian)

	// Attention layer 1
	attnInProjWeights[1] = create2DFloat32FromBytes(tensorMap["val_51"].Data, 32, 96, endian)
	attnOutProjWeights[1] = create2DFloat32FromBytes(tensorMap["attn_layers.1.out_proj.weight"].Data, 32, 32, endian)
	attnInProjBiases[1] = bytesToFloat32Slice(tensorMap["attn_layers.1.in_proj_bias"].Data, endian)
	attnOutProjBiases[1] = bytesToFloat32Slice(tensorMap["attn_layers.1.out_proj.bias"].Data, endian)

	// Attention layer 2
	attnInProjWeights[2] = create2DFloat32FromBytes(tensorMap["val_83"].Data, 32, 96, endian)
	attnOutProjWeights[2] = create2DFloat32FromBytes(tensorMap["attn_layers.2.out_proj.weight"].Data, 32, 32, endian)
	attnInProjBiases[2] = bytesToFloat32Slice(tensorMap["attn_layers.2.in_proj_bias"].Data, endian)
	attnOutProjBiases[2] = bytesToFloat32Slice(tensorMap["attn_layers.2.out_proj.bias"].Data, endian)

	// Attention layer 3
	attnInProjWeights[3] = create2DFloat32FromBytes(tensorMap["val_115"].Data, 32, 96, endian)
	attnOutProjWeights[3] = create2DFloat32FromBytes(tensorMap["attn_layers.3.out_proj.weight"].Data, 32, 32, endian)
	attnInProjBiases[3] = bytesToFloat32Slice(tensorMap["attn_layers.3.in_proj_bias"].Data, endian)
	attnOutProjBiases[3] = bytesToFloat32Slice(tensorMap["attn_layers.3.out_proj.bias"].Data, endian)

	/*
		fmt.Println("inputProjWeight", inputProjWeight[0], inputProjWeight[1], )
		fmt.Println("inputProjBias", inputProjBias[0], inputProjBias[1], )
		fmt.Println("attnInProjWeights", attnInProjWeights[0][0][1], attnInProjWeights[0][1][0], )
		fmt.Println("attnOutProjWeights", attnOutProjWeights[0][0][1], attnOutProjWeights[0][1][0], )
		fmt.Println("attnInProjBiases", attnInProjBiases[0][0], attnInProjBiases[0][1], )
		fmt.Println("attnOutProjBiases", attnOutProjBiases[0][0], attnOutProjBiases[0][1], )
	*/

	// Initialize transformer parameters
	err := transformer.InitializeParameters(
		inputProjWeight,
		inputProjBias,
		attnInProjWeights,
		attnOutProjWeights,
		attnInProjBiases,
		attnOutProjBiases,
	)
	if err != nil {
		return err
	}
	return nil
}
