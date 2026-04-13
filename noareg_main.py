from noareg_traindata import load_tsv_individual, int_to_bits_tensor, bits_to_int, hash_string, clean_default_choices, mask_we_dont_care
from noareg_transformer import NoaregTransformer
import time
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

def parse_args():
    parser = argparse.ArgumentParser(description="Process TSV and generate weights file.")

    parser.add_argument(
        "--input-tsv-file",
        type=str,
        default="~/goruut/dicts/japanese/clean.tsv",
        help="Path to input TSV file"
    )

    parser.add_argument(
        "--output-train-file",
        type=str,
        default="~/goruut/dicts/japanese/weights8.bin",
        help="Path to output weights file"
    )

    parser.add_argument(
        "--seq-len",
        type=int,
        default=16,
        help="Sequence length"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=999999999999,
        help="Limit number of rows"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=41,
        help="Random seed"
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=600,
        help="Patience in seconds"
    )

    return parser.parse_args()


args = parse_args()

# parameters
input_tsv_file = os.path.expanduser(args.input_tsv_file)
output_train_file = os.path.expanduser(args.output_train_file)
seq_len = args.seq_len
limit = args.limit
seed = args.seed
patience = args.patience

# --- Check for GPU availability ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# --- Training ---
torch.manual_seed(seed)  # reproducibility
model = NoaregTransformer().to(device)
criterion = nn.BCELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


in_ints, out_ints, possible, random_in, random_out = load_tsv_individual(
    input_tsv_file, seq_len=seq_len, limit=limit)

in_ints, out_ints = clean_default_choices(in_ints, out_ints, possible)

print("Training Dataset of Size:", len(in_ints))
#print(possible, random_in, random_out)

train_inputs = int_to_bits_tensor(
    torch.tensor(inputs_ints, dtype=torch.uint32, device=device)
)

train_targets = int_to_bits_tensor(
    torch.tensor(outputs_ints, dtype=torch.uint32, device=device)
)

#print(train_inputs.size())
#print(train_targets.size())

best_loss = 99999999.
best = None
best_arr = []
best_success = 0

dataset = TensorDataset(train_inputs, train_targets)
loader = DataLoader(dataset, batch_size=10000, shuffle=True)

step = 0
epoch_time = int(time.time())

for epoch in range(100000):
    #step = epoch
    #batch_inputs = train_inputs
    #batch_targets = train_targets

    patience_time = int(time.time())
    print("clock", patience_time - epoch_time)
    if patience_time - epoch_time > patience:
        break

    for batch_inputs, batch_targets in loader:


        optimizer.zero_grad()

        # mask we don't care
        mask = ~mask_we_dont_care(batch_targets)

        # Forward pass
        y_pred = model(batch_inputs)

        # Compute loss
        loss = criterion(y_pred[mask], batch_targets[mask])

        # Backward + step
        loss.backward()
        optimizer.step()

        loss_item = loss.item()

        step+=1


        if step % 1 == 0:
          print(f"Step {step},\t Loss: {best_loss:.8f}\t Loss actual: {loss_item:.8f}")

          if loss_item < best_loss:
            best_loss = loss_item
            best = model.copy().to('cpu')
            #new_best = best.generate(["ahoj", "svet", "čau", "abeceda"], seq_len, string_to_bits, bits_to_char)
            new_best = best.generate_tokenized(random_in,
						seq_len, int_to_bits, bits_to_int, hash_string, possible)
            if new_best != best_arr:
                best_arr = new_best
                success = 0
                mistake_in = ''
                mistake_out = ''
                for i in range(len(random_in)):
                    if new_best[i] == random_out[i]:
                        success+=1
                    elif mistake_in == '':
                        mistake_in = random_in[i].replace(' ', '')
                        mistake_out = new_best[i]
                print(success, "%", mistake_in, mistake_out)

            if success > best_success:
                epoch_time = int(time.time())
                best_success = success
                # Create example inputs for exporting the model. The inputs should be a tuple of tensors.
                example_inputs = (torch.randn(1, seq_len, 32),)
                best.eval()
                best.export_to_binary(output_train_file)

    print(f"Epoch {epoch},\t Loss: {best_loss:.8f}")
