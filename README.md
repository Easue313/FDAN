# FDAN

This is the official source code implementation for the **FDAN** method.

## 🛠 Requirements

This project requires the following environment. Please ensure your setup is configured correctly:

* **Python:** 3.8
* **PyTorch:** 1.11.0
* **CUDA:** 11.3

You can use `conda` to quickly create a virtual environment and install the required PyTorch version:

```bash
# 1. Create and activate a Python 3.8 virtual environment
conda create -n fdan_env python=3.8
conda activate fdan_env

# 2. Install PyTorch 1.11.0 with CUDA 11.3
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
```

## 🚀 Quick Start

### 1. Data Preparation

Before running the model, you need to configure the dataset file names and paths in `Read_data.py`. Open the file, locate the `ReadData` class, and update the `read_data_file` method:

```python
# Read_data.py
class ReadData:
    def __init__(self, args):
        ...
    def read_data_file(self):
        #1. Map specific file names to their corresponding fault labels
        labels = {'NC Data': 0, 'IF Data': 1, 'OF Data': 2}
        # Note: 'NC Data', 'IF Data', and 'OF Data' represent the actual file names for each fault category.
        ...
        #2. Set your local dataset path here
        file_data = hdf5storage.loadmat(os.path.join(r"your/local/path/to/dataset/", key))
        ...
```

### 2. Configuration
All parameters that need to be adjusted are located in the `config_arg.py` file. You can modify the hyperparameters, training settings, and model configurations there before running the code.

### 3. Running the Model

Once the data paths are successfully configured, you can start the project by running `FDAN.py` as the main script:

```bash
python FDAN.py
```

## 📁 Project Structure

FDAN

├── utils

├── config_arg.py 

├── Data1.mat 

├── Data2.mat

├── Data3.mat

├── Data4.mat

├── Read_data.py

├── some_tools.py 

├── model.py 

└── FDAN.py 
