### To run the project:

1. Open config.py and set:
   
   * IMAGES_BASE_DIR           = r"[working_directory]\images"
   
   * MATLAB_FOLDER, NPY_FOLDER = r"[working_directory]\data"

3. Ensure the data files (dVT.npy, sig.npy, Vd_mean_*\_dam.mat, meshPINN.mat) are in the "___\data" folder.

4. Open conv_model.py and set PLOT_SAVE to True or False, indicating whether you want to produce an image.

5. Open optimize_study.py and set desired TRIAL EXECUTION parameters.

6. Run optimize_study.py.
