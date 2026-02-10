import torch
import torch.nn as nn
import copy
import struct
import numpy as np
import zlib
from typing import Dict, List

class NoaregTransformer(nn.Module):
    def __init__(self, embed_dimension=32, head_count=16, max_len=100, num_layers=4):
        super().__init__()

        self.embed_dim = embed_dimension
        self.heads = head_count
        self.inference = False
        self.num_layers = num_layers

        # Input projection
        self.input_proj = nn.Linear(embed_dimension, embed_dimension)
        self.input_act = nn.ReLU()

        # Positional encoding
        self.pos_encoder = PositionalEncoding(embed_dimension, max_len=max_len)

        # Create attention layers as nn.ModuleList so they're properly registered
        self.attn_layers = nn.ModuleList()
        self.activation_layers = nn.ModuleList()
        
        for i in range(num_layers):
            # Create attention layer
            attn_layer = nn.MultiheadAttention(
                embed_dim=embed_dimension,
                num_heads=head_count,
                batch_first=True
            )
            self.attn_layers.append(attn_layer)
            
            # Create activation (Sigmoid for last layer, ReLU for others)
            if i == num_layers - 1:
                self.activation_layers.append(nn.Sigmoid())
            else:
                self.activation_layers.append(nn.ReLU())

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x):
        B, L, E = x.shape

        # Input projection + activation
        x = self.input_act(self.input_proj(x))
        x = self.pos_encoder(x)

        # Apply attention layers
        for i in range(self.num_layers):
            attn_out, _ = self.attn_layers[i](x, x, x)
            x = x + attn_out  # Residual connection
            x = self.activation_layers[i](x)

        return x

    def copy(self):
        return copy.deepcopy(self)


    def generate_tokenized(self, strs, seq_len, int_to_bits, bits_to_int, hash_string, possible, alt_model=None):
        strs = copy.deepcopy(strs)
        for j in range(len(strs)):
            strs[j] = strs[j].split(' ')
            for i in range(len(strs[j])):
                strs[j][i] = hash_string(strs[j][i])
            while len(strs[j]) < seq_len:
                strs[j] += [0]
            strs[j] = strs[j][:seq_len]

        #print(strs)

        x_in = torch.stack([int_to_bits(s) for s in strs])

        if alt_model is None:
            # Set to evaluation mode
            was_training = self.training
            self.eval()
            with torch.no_grad():
                next_tokens = self(x_in)
            if was_training:
                self.train()
        else:
            next_tokens = alt_model(x_in)

        next_tokens = (next_tokens > 0.5).float()

        # Convert back to strings
        ret = []
        for b in range(next_tokens.size(0)):
            ret_str = ""
            for l in range(next_tokens.size(1)):
                val = bits_to_int(next_tokens[b][l])
                #print(val)
                w = ''
                p = 33
                possible_key = strs[b][l]
                if possible_key == 0:
                    break
                if not possible_key in possible:
                    break
                if len(possible[possible_key]) == 1:
                    for k in possible[possible_key]:
                        w = possible[possible_key][k]
                        break
                else:
                    for k in possible[possible_key]:
                        cnt = (k ^ val).bit_count()
                        if cnt < p:
                            p = cnt
                            w = possible[possible_key][k]
                ret_str += w
            ret.append(ret_str)

        return ret


    def export_to_binary(self, output_path: str, force_transpose_weights: bool = False) -> None:
        """
        Export model weights directly to binary format matching the Go code.
        Binary format (little endian):
        - Magic bytes: 'GOBF' (4 bytes)
        - Version: int32 (4 bytes)
        - Number of tensors: int32 (4 bytes)
        - For each tensor:
            - Name length: int32 (4 bytes)
            - Name: bytes (variable)
            - Data type: int32 (4 bytes, 1=float32)
            - Number of dimensions: int32 (4 bytes)
            - Dimensions: list[int64] (8 bytes each)
            - Data length: int32 (4 bytes)
            - Raw data: bytes (variable, float32 values)
        """
        # Get state dict
        state_dict = self.state_dict()
        
        # Map state dict keys to expected Go tensor names
        tensor_map = self._create_tensor_map(state_dict, force_transpose_weights)
        
        #print(f"Exporting {len(tensor_map)} tensors:")
        #for name, tensor in tensor_map.items():
        #    print(f"  {name}: shape {tensor.shape}, dtype {tensor.dtype}")
        
        # Write to binary file
        with open(output_path, 'wb') as f:
            # Write magic and version (little endian)
            f.write(b'GOBF')  # Magic bytes
            f.write(struct.pack('<I', 1))  # Version 1 (unsigned int, little endian)
            
            # Write number of tensors
            num_tensors = len(tensor_map)
            f.write(struct.pack('<I', num_tensors))
            #print(f"\nWriting {num_tensors} tensors...")
            
            # Write each tensor
            for tensor_name in sorted(tensor_map.keys()):
                self._write_tensor(f, tensor_name, tensor_map[tensor_name])
        
        #print(f"\nExported model to {output_path}")
        
        # Also save compressed version with zlib
        with open(output_path, 'rb') as f:
            binary_data = f.read()
        
        compressed_data = zlib.compress(binary_data)
        compressed_path = output_path + '.zlib'
        
        with open(compressed_path, 'wb') as f:
            f.write(compressed_data)
        
        #print(f"Exported compressed model to {compressed_path}")
            
    def _create_tensor_map(self, state_dict: Dict, force_transpose: bool = True) -> Dict[str, np.ndarray]:
        """
        Map PyTorch state dict keys to the expected tensor names in Go code.
        
        Note: PyTorch stores linear layer weights as [out_features, in_features].
        The Go code might expect them as [in_features, out_features] or vice versa.
        """
        tensor_map = {}
        
        #print("\nProcessing PyTorch state dict:")
        #for key, value in state_dict.items():
        #    print(f"  {key}: shape {tuple(value.shape)}")
        
        # ATTENTION THIS ONE IS TRANSPOSED IN ONNX, SO NOT TRANSPOSING IF TRANSPOSING
        # 1. Input projection
        # PyTorch: input_proj.weight shape [32, 32] (out_features, in_features)
        # Go: expects val_0 with shape [32, 32] (but might need transpose)
        input_proj_weight = state_dict["input_proj.weight"].cpu().numpy()
        if not force_transpose:
            input_proj_weight = input_proj_weight.T  # [32, 32] -> [32, 32] (but transposed)
        
        tensor_map["val_0"] = input_proj_weight.astype(np.float32)
        tensor_map["input_proj.bias"] = state_dict["input_proj.bias"].cpu().numpy().astype(np.float32)
        
        # 2. Attention layers
        for i in range(self.num_layers):
            # Map layer index to val_* keys used in Go code
            val_key_map = {0: 15, 1: 51, 2: 83, 3: 115}
            val_key = val_key_map[i]
            
            # Get attention layer
            attn_layer = self.attn_layers[i]
            
            # Attention in_proj weights (combined Q, K, V projection)
            # PyTorch shape: [3*embed_dim, embed_dim] = [96, 32]
            # Go expects: [32, 96] (based on create2DFloat32FromBytes(tensorMap["val_15"].Data, 32, 96))
            in_proj_weight = state_dict[f"attn_layers.{i}.in_proj_weight"].cpu().numpy()
            
            if force_transpose:
                # Transpose to [32, 96] for Go
                in_proj_weight = in_proj_weight
            else:
                # Keep as [96, 32] but reshape differently?
                # The Go code reshapes to [32, 96], so we might need to reshape
                in_proj_weight = in_proj_weight.T  # Actually, let's transpose for consistency
            
            tensor_map[f"val_{val_key}"] = in_proj_weight.astype(np.float32)
            
            # Attention in_proj bias
            # PyTorch shape: [3*embed_dim] = [96]
            # Go expects: [96]
            in_proj_bias = state_dict[f"attn_layers.{i}.in_proj_bias"].cpu().numpy()
            tensor_map[f"attn_layers.{i}.in_proj_bias"] = in_proj_bias.astype(np.float32)
            
            # Attention out_proj weights
            # PyTorch shape: [embed_dim, embed_dim] = [32, 32]
            # Go expects: [32, 32] (based on create2DFloat32FromBytes(tensorMap["attn_layers.0.out_proj.weight"].Data, 32, 32))
            out_proj_weight = state_dict[f"attn_layers.{i}.out_proj.weight"].cpu().numpy()
            
            if force_transpose:
                # Transpose to [32, 32] (but already [32, 32])
                out_proj_weight = out_proj_weight.T
            
            tensor_map[f"attn_layers.{i}.out_proj.weight"] = out_proj_weight.astype(np.float32)
            
            # Attention out_proj bias
            # PyTorch shape: [embed_dim] = [32]
            # Go expects: [32]
            out_proj_bias = state_dict[f"attn_layers.{i}.out_proj.bias"].cpu().numpy()
            tensor_map[f"attn_layers.{i}.out_proj.bias"] = out_proj_bias.astype(np.float32)
        
        return tensor_map
    
    def _write_tensor(self, file, name: str, tensor: np.ndarray) -> None:
        """
        Write a single tensor to the binary file (little endian).
        """
        # Ensure tensor is C-contiguous and float32
        if tensor.dtype != np.float32:
            tensor = tensor.astype(np.float32)
        
        tensor = np.ascontiguousarray(tensor)
        
        # Write name (UTF-8 encoded)
        name_bytes = name.encode('utf-8')
        name_len = len(name_bytes)
        
        # Write name length (int32, little endian)
        file.write(struct.pack('<I', name_len))
        # Write name bytes
        file.write(name_bytes)
        
        # Write data type (1 = float32, int32, little endian)
        file.write(struct.pack('<I', 1))
        
        # Write shape/dimensions
        shape = tensor.shape
        num_dims = len(shape)
        
        # Write number of dimensions (int32, little endian)
        file.write(struct.pack('<I', num_dims))
        
        # Write each dimension (int64, little endian)
        for dim in shape:
            file.write(struct.pack('<q', dim))  # 'q' = signed long long (8 bytes)
        
        # Write raw data
        raw_data = tensor.tobytes()
        data_len = len(raw_data)
        
        # Write data length (int32, little endian)
        file.write(struct.pack('<I', data_len))
        # Write raw data
        file.write(raw_data)
    



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]
