# Installation Guide - Drone Video Processing System

Complete installation instructions for Ubuntu/Linux and Windows platforms.

## Table of Contents
- [System Requirements](#system-requirements)
- [Ubuntu/Linux Installation](#ubuntulinux-installation)
- [Windows Installation](#windows-installation)
- [Verification](#verification)
- [Model Weights](#model-weights)
- [Troubleshooting](#troubleshooting)

---

## System Requirements

### Minimum Requirements
- **OS**: Ubuntu 20.04+ / Windows 10/11
- **Python**: 3.8, 3.9, 3.10, or 3.11
- **RAM**: 16GB
- **Storage**: 10GB free space
- **GPU**: NVIDIA GPU with 4GB VRAM (GTX 1650 or better)
- **CUDA**: 11.8 or 12.1

### Recommended Requirements
- **OS**: Ubuntu 22.04 / Windows 11
- **Python**: 3.10
- **RAM**: 32GB+
- **Storage**: 50GB+ SSD
- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 3060 or better)
- **CUDA**: 11.8 or 12.1

### Tested Configurations
- ✅ NVIDIA Jetson Orin (16GB) - Ubuntu 20.04, JetPack 5.x
- ✅ RTX 3060 (12GB) - Windows 11, CUDA 11.8
- ✅ RTX 4090 (24GB) - Ubuntu 22.04, CUDA 12.1
- ✅ RTX 3080 (10GB) - Windows 10, CUDA 11.8

---

## Ubuntu/Linux Installation

### Step 1: System Dependencies

```bash
# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install Python and development tools
sudo apt-get install -y python3-pip python3-dev python3-venv
sudo apt-get install -y build-essential cmake git wget

# Install OpenCV dependencies
sudo apt-get install -y libopencv-dev libopencv-contrib-dev

# Install ffmpeg (optional, for video compression)
sudo apt-get install -y ffmpeg
```

### Step 2: Install CUDA Toolkit

#### For CUDA 11.8 (Recommended for compatibility):
```bash
# Download and install CUDA keyring
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb

# Update and install CUDA
sudo apt-get update
sudo apt-get -y install cuda-11-8

# Add to PATH (add to ~/.bashrc for persistence)
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
```

#### For CUDA 12.1:
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-12-1

export PATH=/usr/local/cuda-12.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH
```

### Step 3: Install cuDNN (Optional but recommended)
```bash
sudo apt-get install -y libcudnn8 libcudnn8-dev
```

### Step 4: Create Virtual Environment
```bash
# Navigate to project directory
cd /path/to/project

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

### Step 5: Install PyTorch

#### For CUDA 11.8:
```bash
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
```

#### For CUDA 12.1:
```bash
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
```

#### For CPU only (NOT RECOMMENDED):
```bash
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu
```

### Step 6: Install Project Dependencies
```bash
pip install -r requirements-ubuntu.txt
```

### Step 7: Verify Installation
```bash
# Run verification script
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python3 -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python3 -c "import kornia; print(f'Kornia: {kornia.__version__}')"

# Test GPU
python3 -c "import torch; t = torch.randn(100, 100).cuda(); print('✓ GPU test passed!')"
```

---

## Windows Installation

### Step 1: Install Python

1. Download Python 3.10 from: https://www.python.org/downloads/
2. Run installer
3. **Important**: Check these boxes:
   - ✅ Add Python to PATH
   - ✅ Install pip
4. Click "Install Now"

### Step 2: Install CUDA Toolkit

#### For CUDA 11.8 (Recommended):
1. Download from: https://developer.nvidia.com/cuda-11-8-0-download-archive
2. Select: Windows → x86_64 → 10/11 → exe (local)
3. Run installer with default settings
4. Restart computer

#### For CUDA 12.1:
1. Download from: https://developer.nvidia.com/cuda-12-1-0-download-archive
2. Follow same steps as above

### Step 3: Install cuDNN

1. Download from: https://developer.nvidia.com/cudnn (requires NVIDIA account)
2. Extract ZIP file
3. Copy files to CUDA directory:
   ```
   Copy bin\*.dll → C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin
   Copy include\*.h → C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\include
   Copy lib\*.lib → C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\lib\x64
   ```

### Step 4: Install Visual Studio Build Tools

1. Download from: https://visualstudio.microsoft.com/downloads/
2. Install "Desktop development with C++" workload
3. Restart computer

### Step 5: Create Virtual Environment

Open Command Prompt or PowerShell:
```cmd
# Navigate to project directory
cd C:\path\to\project

# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Upgrade pip
python -m pip install --upgrade pip setuptools wheel
```

### Step 6: Install PyTorch

#### For CUDA 11.8:
```cmd
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
```

#### For CUDA 12.1:
```cmd
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
```

### Step 7: Install Project Dependencies
```cmd
pip install -r requirements-windows.txt
```

### Step 8: Install ffmpeg (Optional)

1. Download from: https://ffmpeg.org/download.html
2. Extract to `C:\ffmpeg`
3. Add to PATH:
   - Open System Properties → Environment Variables
   - Edit "Path" variable
   - Add: `C:\ffmpeg\bin`
4. Restart Command Prompt

### Step 9: Verify Installation
```cmd
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "import kornia; print(f'Kornia: {kornia.__version__}')"

# Test GPU
python -c "import torch; t = torch.randn(100, 100).cuda(); print('✓ GPU test passed!')"
```

---

## Verification

### Complete Verification Script

Create a file `verify_installation.py`:

```python
import sys

def verify_installation():
    print("=" * 70)
    print("INSTALLATION VERIFICATION")
    print("=" * 70)
    
    # Python version
    print(f"\n✓ Python: {sys.version.split()[0]}")
    
    # PyTorch
    try:
        import torch
        print(f"✓ PyTorch: {torch.__version__}")
        print(f"  CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA Version: {torch.version.cuda}")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    except ImportError as e:
        print(f"✗ PyTorch: NOT INSTALLED - {e}")
        return False
    
    # OpenCV
    try:
        import cv2
        print(f"✓ OpenCV: {cv2.__version__}")
    except ImportError as e:
        print(f"✗ OpenCV: NOT INSTALLED - {e}")
        return False
    
    # NumPy
    try:
        import numpy
        print(f"✓ NumPy: {numpy.__version__}")
    except ImportError as e:
        print(f"✗ NumPy: NOT INSTALLED - {e}")
        return False
    
    # SciPy
    try:
        import scipy
        print(f"✓ SciPy: {scipy.__version__}")
    except ImportError as e:
        print(f"✗ SciPy: NOT INSTALLED - {e}")
        return False
    
    # Kornia
    try:
        import kornia
        print(f"✓ Kornia: {kornia.__version__}")
    except ImportError as e:
        print(f"✗ Kornia: NOT INSTALLED - {e}")
        return False
    
    # GPU Test
    if torch.cuda.is_available():
        try:
            t = torch.randn(100, 100).cuda()
            result = t @ t.T
            print(f"✓ GPU Test: PASSED")
        except Exception as e:
            print(f"✗ GPU Test: FAILED - {e}")
            return False
    
    print("\n" + "=" * 70)
    print("✓ ALL CHECKS PASSED - Installation successful!")
    print("=" * 70)
    return True

if __name__ == "__main__":
    success = verify_installation()
    sys.exit(0 if success else 1)
```

Run verification:
```bash
python verify_installation.py
```

---

## Model Weights

### Automatic Download

Model weights are downloaded automatically on first run via `torch.hub`:

1. **DINOv2** (from facebookresearch/dinov2)
   - Model: `dinov2_vitb14` (default)
   - Size: ~350MB
   - Location: `~/.cache/torch/hub/`

2. **LoFTR** (included in Kornia)
   - Model: `outdoor` (default)
   - Size: ~150MB
   - Location: `~/.cache/torch/hub/checkpoints/`

### Manual Download (if needed)

If automatic download fails, manually download:

```python
import torch

# Download DINOv2
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')

# Download LoFTR (via Kornia)
from kornia.feature import LoFTR
loftr = LoFTR(pretrained='outdoor')
```

---

## Troubleshooting

### Common Issues

#### 1. CUDA Not Available
**Symptoms**: `torch.cuda.is_available()` returns `False`

**Solutions**:
- Update NVIDIA GPU driver
- Reinstall CUDA Toolkit
- Reinstall PyTorch with correct CUDA version
- Check GPU compatibility (Compute Capability 3.5+)

#### 2. Out of Memory
**Symptoms**: `CUDA out of memory` error

**Solutions**:
- Reduce `grid_size` (e.g., 7 → 5)
- Reduce `resize_max` (e.g., 840 → 640)
- Reduce `tile_cache_items` (e.g., 128 → 64)
- Close other GPU applications

#### 3. Import Errors
**Symptoms**: `ModuleNotFoundError` or `ImportError`

**Solutions**:
- Activate virtual environment
- Reinstall package: `pip install <package>`
- Check Python version compatibility

#### 4. Slow Performance
**Symptoms**: Processing takes very long

**Solutions**:
- Verify GPU is being used (check `torch.cuda.is_available()`)
- Install CUDA-enabled PyTorch
- Update GPU drivers
- Check GPU utilization with `nvidia-smi`

### Getting Help

If issues persist:
1. Check logs in console output
2. Verify all verification steps pass
3. Check GPU memory usage: `nvidia-smi`
4. Review error messages carefully

---

## Next Steps

After successful installation:

1. **Configure paths** in `clean_copy_refactor_loftr.py`:
   - Set `video_path` to your video file
   - Set `tiles_dir` to your satellite tiles directory
   - Set `initial_tile_x` and `initial_tile_y` coordinates

2. **Run the script**:
   ```bash
   python clean_copy_refactor_loftr.py
   ```

3. **Monitor progress**:
   - Watch console output for frame processing
   - Check GPU usage with `nvidia-smi` (Linux) or Task Manager (Windows)

4. **View results**:
   - Trajectory map: `results_*/trajectory_map.png`
   - Trajectory video: `results_*/trajectory_video.mp4`
   - Match visualizations: `results_*/match_*.png`
   - Google Maps: `results_*/trajectory.html`

---

## Performance Tips

### For Best Performance:
- Use GPU with 8GB+ VRAM
- Use SSD for tile storage
- Set `grid_size=7` for accuracy
- Set `frame_skip=10` for speed
- Enable CUDA optimizations (automatic in code)

### For Limited Resources:
- Reduce `grid_size` to 5
- Increase `frame_skip` to 15-20
- Reduce `resize_max` to 640
- Use smaller tile cache

---

**Installation complete! You're ready to process drone videos.**
