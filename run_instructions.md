# Prostate MRI Analysis GUI - Setup Instructions

## Prerequisites
- Python 3.12.6 (verified working version)
- pip package manager (install via: python -m ensurepip --upgrade)
- NVIDIA GPU with CUDA 12.0+ (recommended)

## Installation
1. Clone the repository
2. Set up virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate  # Windows
   ```
3. Install exact dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```

## Running
```bash
export PYTHONPATH=$(pwd)
python Main/GUI.py
```

## Notes
- First run initializes models (may take 5-10 minutes)
- Verify all data files exist in Main/data/
- For GPU support, install matching CUDA/cuDNN versions