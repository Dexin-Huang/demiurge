#!/bin/bash
# One-time pod setup for DEMIURGE
# Run this on a fresh RunPod pod with the network volume attached.
# After first run, data persists on /runpod-volume across pod recreations.
#
# Data sources: HuggingFace (https://huggingface.co/collections/quentinll/lewm)
# No more Google Drive quota issues.
set -e

VOLUME=/runpod-volume
WORKSPACE=/workspace/demiurge
VENV=$WORKSPACE/baselines/lewm/.venv
CKPT_DIR=$VOLUME/checkpoints
DATA_DIR=$VOLUME/data

echo "=== DEMIURGE Pod Setup ==="

# 1. Clone repo (or pull if exists)
if [ -d "$WORKSPACE" ]; then
    echo "Repo exists, pulling..."
    cd $WORKSPACE && git pull && git checkout v0.2-hybrid-simulator
else
    echo "Cloning repo..."
    cd /workspace && git clone --recurse-submodules https://github.com/Dexin-Huang/demiurge.git
    cd $WORKSPACE && git checkout v0.2-hybrid-simulator
fi

# 2. Python environment
if [ -d "$VENV" ]; then
    echo "Venv exists, activating..."
else
    echo "Creating venv..."
    pip install uv 2>/dev/null
    cd $WORKSPACE/baselines/lewm
    uv venv --python=3.10
fi
source $VENV/bin/activate

# Check if packages installed
python -c "import stable_worldmodel; import torch; print('Packages OK')" 2>/dev/null || {
    echo "Installing packages..."
    uv pip install 'stable-worldmodel[train,env]'
    uv pip install 'torch==2.4.0+cu124' 'torchvision==0.19.0+cu124' --index-url https://download.pytorch.org/whl/cu124
    uv pip install 'datasets>=2.18,<3.0' scipy scikit-learn huggingface_hub
    # Fix torchvision transforms compatibility
    python3 -c "
import re
path = '$VENV/lib/python3.10/site-packages/stable_pretraining/data/transforms.py'
with open(path, 'r') as f: content = f.read()
patched = content.replace('self.transform(', 'self._transform(')
with open(path, 'w') as f: f.write(patched)
print('Patched transforms')
"
}

# 3. Data from HuggingFace (replaces Google Drive)
mkdir -p $DATA_DIR $CKPT_DIR

# LeWM pretrained checkpoint from HuggingFace
if [ -f "$CKPT_DIR/lejepa_object.ckpt" ]; then
    echo "LeWM checkpoint exists"
else
    echo "Downloading LeWM checkpoint from HuggingFace..."
    python3 -c "
from huggingface_hub import hf_hub_download
import zstandard, tarfile, os

# Download Push-T checkpoint from LeWM HF collection
path = hf_hub_download(
    repo_id='quentinll/lewm-pusht',
    filename='ckpt/lejepa.tar.zst',
    repo_type='dataset',
)
print(f'Downloaded: {path}')

# Extract
with open(path, 'rb') as f:
    dctx = zstandard.ZstdDecompressor()
    with dctx.stream_reader(f) as reader:
        with tarfile.open(fileobj=reader, mode='r|') as tar:
            tar.extractall('$CKPT_DIR/')

# Move to expected location
for name in ['lejepa_object.ckpt', 'lejepa_weights.ckpt']:
    src = os.path.join('$CKPT_DIR', 'pusht', name)
    dst = os.path.join('$CKPT_DIR', name)
    if os.path.exists(src):
        os.rename(src, dst)

# Cleanup
import shutil
pusht_dir = os.path.join('$CKPT_DIR', 'pusht')
if os.path.isdir(pusht_dir):
    shutil.rmtree(pusht_dir)
print('Checkpoint ready')
"
fi

# Push-T dataset from HuggingFace
if [ -f "$DATA_DIR/pusht_expert_train.h5" ]; then
    echo "Push-T dataset exists"
else
    echo "Downloading Push-T dataset from HuggingFace..."
    python3 -c "
from huggingface_hub import hf_hub_download
import zstandard, os

# Download dataset
path = hf_hub_download(
    repo_id='quentinll/lewm-pusht',
    filename='dataset/pusht_expert_train.h5.zst',
    repo_type='dataset',
)
print(f'Downloaded: {path}')
print('Decompressing (this takes a few minutes)...')

# Decompress directly to volume
with open(path, 'rb') as fin:
    dctx = zstandard.ZstdDecompressor()
    with open('$DATA_DIR/pusht_expert_train.h5', 'wb') as fout:
        dctx.copy_stream(fin, fout, read_size=8*1024*1024, write_size=8*1024*1024)

size = os.path.getsize('$DATA_DIR/pusht_expert_train.h5') / 1e9
print(f'Dataset ready: {size:.1f} GB')
"
fi

# Symlink for stable_worldmodel
mkdir -p /root/.stable_worldmodel
ln -sf $DATA_DIR/pusht_expert_train.h5 /root/.stable_worldmodel/pusht_expert_train.h5 2>/dev/null || true

# 4. Verify
echo ""
echo "=== Verification ==="
python -c "
import torch
print(f'torch: {torch.__version__}, cuda: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"
ls -lh $CKPT_DIR/lejepa_object.ckpt 2>/dev/null && echo "Checkpoint: OK" || echo "Checkpoint: MISSING"
ls -lh $DATA_DIR/pusht_expert_train.h5 2>/dev/null && echo "Dataset: OK" || echo "Dataset: MISSING"

echo ""
echo "=== Setup Complete ==="
echo "To train:"
echo "  source $VENV/bin/activate"
echo "  export PYTHONPATH=$WORKSPACE/baselines/lewm:\$PYTHONPATH"
echo "  python $WORKSPACE/experiments/train_temporal.py --checkpoint $CKPT_DIR/lejepa_object.ckpt --data_dir $DATA_DIR"
