method = "SimVP"

# Dataset
dataname = "fus_npz"
manifest_path = "C:/Users/ESPCI/Documents/GitHub/fUSPredict/derivatives/preprocessing/splits_multi.json"
frames_key = "frames"
pre_seq_length = 20
aft_seq_length = 20
total_length = pre_seq_length + aft_seq_length
in_shape = [pre_seq_length, 1, 128, 128]  # [T, C, H, W], adjust H/W for your data
stride = 1
seed = 42
data_root = "./data"  # ignored when manifest_path is absolute

# Model
spatio_kernel_enc = 3
spatio_kernel_dec = 3
hid_S = 64
hid_T = 512
N_T = 8
N_S = 4

# Training
epoch = 100
lr = 1e-3
batch_size = 8
val_batch_size = 8
drop_path = 0.0
sched = "onecycle"
