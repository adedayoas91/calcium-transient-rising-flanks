README


Pandas DataFrame

to load:
import pickle
pickle_filename = 'YOUR_DATA_PATH/df_name.pkl'  # change accordingly
with open(pickle_filename, 'rb') as pickle_in:
     df_name = pickle.load(pickle_in)


Motorneuron data:
Fish 3 Trial 1 and Fish 5 Trial 2 for Figure 3.
Fish 5 Trial 2 for figure 4.

Columns:
- Fish: fish index
- Trial: trial index
- fluo: fluorescence traces [n_cells x n_timesteps]
- fluo_type: 'dff' or 'f_smooth', respectively before and after smoothing procedure
- n_cells: number of cells in the plane (only those kept for analysis, "bad" cells removed)
- mid: middle cell, to split left vs right neurons (left until index mid-1, right from index mid and on)
- cell_centers: x and y position of the cell center [n_cells x 2]
- multivariate: boolean to indicate bivariate (False) or multivariate (True) GC
- GC: Granger causality matrix results [n_cells x n_cells]
- GC_sig: Granger causality matrix results, significant with original threshold (where Fstat > threshold_F) [n_cells x n_cells]
- GC_sig_new_thresh: Granger causality matrix results, significant with new threshold (where Fstat > new_threshold_F) [n_cells x n_cells]
- Fstat: F-statistics matrix [n_cells x n_cells]
- threshold_F: original threshold for the F-statistics significance
- new_threshold_F: new threshold for the F-statistics after the whole pipeline is applied
