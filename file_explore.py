import numpy as np
import scipy.io as sio

INSPECT_NPY = False
# NPY_FILE = 'data/sig.npy'
NPY_FILE = 'data/dVT.npy'

INSPECT_MAT = True
MAT_FILE = 'data/Vd_mean_double_dam.mat'
VARIABLE = 'Vdmean'
# MAT_FILE = 'data/V0_truss.mat'
# VARIABLE = 'V0_truss'
# MAT_FILE = 'data/meshPINN.mat'
# VARIABLE = 'NodalCoords'

if INSPECT_NPY:
    # 1. Load the file
    data = np.load(NPY_FILE, allow_pickle=True)

    # 2. Check the metadata (Structure)
    print("--- METADATA ---")
    print(f"Data Type (dtype): {data.dtype}")
    print(f"Shape (dimensions): {data.shape}")
    print(f"Total Size (number of elements): {data.size}")
    print(f"Number of Dimensions: {data.ndim}\n")

    # 3. Look at the actual content
    print("--- CONTENT ---")
    print(data)

if INSPECT_MAT:
    # Load everything into a Python dictionary
    mat_data = sio.loadmat(MAT_FILE)

    # MATLAB files automatically include some header metadata (keys starting with __)
    # Let's filter those out to see your actual data keys:
    clean_keys = [k for k in mat_data.keys() if not k.startswith('__')]
    print(f"Actual variables you can access: {clean_keys}\n")

    # Look at a specific variable
    my_variable = mat_data[VARIABLE]
    print("--- VARIABLE CONTENT ---")
    print(my_variable)
    print(f"Shape: {my_variable.shape}")