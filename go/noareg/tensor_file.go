package noareg

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
)

type TensorInfo struct {
	Name       string
	DataType   int32
	Dimensions []int64
	Data       []byte
}

func ReadTensors(reader io.Reader) ([]TensorInfo, error) {
	file := reader

	magic := make([]byte, 4)
	if _, err := io.ReadFull(file, magic); err != nil {
		return nil, err
	}
	if string(magic) != "GOBF" {
		return nil, fmt.Errorf("invalid magic: %s", magic)
	}

	var version int32
	if err := binary.Read(file, binary.LittleEndian, &version); err != nil {
		return nil, err
	}
	if version != 1 {
		return nil, fmt.Errorf("unsupported version: %d", version)
	}

	var numTensors int32
	if err := binary.Read(file, binary.LittleEndian, &numTensors); err != nil {
		return nil, err
	}

	tensors := make([]TensorInfo, numTensors)

	for i := range tensors {
		var nameLen int32
		if err := binary.Read(file, binary.LittleEndian, &nameLen); err != nil {
			return nil, err
		}
		nameBytes := make([]byte, nameLen)
		if _, err := io.ReadFull(file, nameBytes); err != nil {
			return nil, err
		}
		tensors[i].Name = string(nameBytes)

		if err := binary.Read(file, binary.LittleEndian, &tensors[i].DataType); err != nil {
			return nil, err
		}

		var numDims int32
		if err := binary.Read(file, binary.LittleEndian, &numDims); err != nil {
			return nil, err
		}

		tensors[i].Dimensions = make([]int64, numDims)
		for j := range tensors[i].Dimensions {
			if err := binary.Read(file, binary.LittleEndian, &tensors[i].Dimensions[j]); err != nil {
				return nil, err
			}
		}

		var dataLen int32
		if err := binary.Read(file, binary.LittleEndian, &dataLen); err != nil {
			return nil, err
		}
		tensors[i].Data = make([]byte, dataLen)
		if _, err := io.ReadFull(file, tensors[i].Data); err != nil {
			return nil, err
		}
	}

	return tensors, nil
}

func bytesToFloat32Slice(data []byte, endian bool) []float32 {
	result := make([]float32, len(data)/4)
	for i := range result {
		if endian {
			result[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
		} else {
			result[i] = math.Float32frombits(binary.BigEndian.Uint32(data[i*4:]))
		}
	}
	return result
}

func create2DFloat32FromBytes(data []byte, rows, cols int, endian bool) [][]float32 {
	floatData := bytesToFloat32Slice(data, endian)
	result := make([][]float32, rows)
	for i := range result {
		result[i] = make([]float32, cols)
		for j := range result[i] {
			if i*cols+j < len(floatData) {
				result[i][j] = floatData[i*cols+j]
			}
		}
	}
	return result
}
