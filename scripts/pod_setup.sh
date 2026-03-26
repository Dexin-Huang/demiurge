#!/bin/bash
# One-time pod setup for DEMIURGE
# Run this on a fresh RunPod pod with the network volume attached.
# After first run, data persists on /runpod-volume across pod recreations.
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

# 2. Python environment (install into volume so it persists)
if [ -d "$VENV" ]; then
    echo "Venv exists, activating..."
else
    echo "Creating venv..."
    cd $WORKSPACE/baselines/lewm
    uv venv --python=3.10
fi
source $VENV/bin/activate

# Check if packages installed
python -c "import stable_worldmodel; import torch; print('Packages OK')" 2>/dev/null || {
    echo "Installing packages..."
    uv pip install 'stable-worldmodel[train,env]'
    uv pip install 'torch==2.4.0+cu124' 'torchvision==0.19.0+cu124' --index-url https://download.pytorch.org/whl/cu124
    uv pip install 'datasets>=2.18,<3.0' scipy scikit-learn
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

# 3. Data (on persistent volume)
mkdir -p $DATA_DIR $CKPT_DIR

# LeWM pretrained checkpoint
if [ -f "$CKPT_DIR/lejepa_object.ckpt" ]; then
    echo "LeWM checkpoint exists"
else
    echo "Downloading LeWM checkpoint..."
    cd $CKPT_DIR
    gdown 1CagjbwPOovHlmcvot07eWvq7fGswdYtI -O lejepa.tar.zst
    python3 -c "
import zstandard, tarfile, io
with open('lejepa.tar.zst', 'rb') as f:
    dctx = zstandard.ZstdDecompressor()
    with dctx.stream_reader(f) as reader:
        with tarfile.open(fileobj=reader, mode='r|') as tar:
            tar.extractall('$CKPT_DIR/')
print('Extracted')
"
    # Move checkpoint to expected location
    mv $CKPT_DIR/pusht/lejepa_object.ckpt $CKPT_DIR/ 2>/dev/null || true
    mv $CKPT_DIR/pusht/lejepa_weights.ckpt $CKPT_DIR/ 2>/dev/null || true
    rm -f lejepa.tar.zst
    rm -rf $CKPT_DIR/pusht
fi

# Push-T dataset
if [ -f "$DATA_DIR/pusht_expert_train.h5" ]; then
    echo "Push-T dataset exists"
else
    if [ -f "$VOLUME/pusht_expert_train.h5.zst" ]; then
        echo "Decompressing dataset..."
        python3 -c "
import zstandard, os
with open('$VOLUME/pusht_expert_train.h5.zst', 'rb') as fin:
    dctx = zstandard.ZstdDecompressor()
    with open('$DATA_DIR/pusht_expert_train.h5', 'wb') as fout:
        dctx.copy_stream(fin, fout, read_size=8*1024*1024, write_size=8*1024*1024)
print(f'Decompressed: {os.path.getsize(\"$DATA_DIR/pusht_expert_train.h5\")/1e9:.1f} GB')
"
        rm -f $VOLUME/pusht_expert_train.h5.zst
    else
        echo "ERROR: Dataset not found. Upload pusht_expert_train.h5.zst to $VOLUME/"
        echo "  scp -P <port> pusht_expert_train.h5.zst root@<ip>:$VOLUME/"
    fi
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
echo "To train: source $VENV/bin/activate && PYTHONPATH=$WORKSPACE/baselines/lewm python $WORKSPACE/experiments/train_shield.py --checkpoint $CKPT_DIR/lejepa_object.ckpt --data_dir $DATA_DIR"
