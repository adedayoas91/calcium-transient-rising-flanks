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


Hindbrain data
Fish 6 Trial 07

Columns:
- fluo: fluorescence traces [n_cells x n_timesteps]
- cell_centers: x and y position of the cell center [n_cells x 2]
- background: plane background for plotting [249 x 512]
- n_cells: number of cells in the plane
- tail_angle: array of angle of the tail [75000,] - 75000 timesteps: higher frequency than calcium imaging recording
- tail_angle_regressor: tail angle convolved to calcium decay function [75000,] 
- is_swim: boolean array whether each cell in correlated to swim activity (True if pearson correlation between cell's fluorescence trace and tail_angle_regressor > 0.6) [n_cells,]
- swim_neurons: indices of swim-correlated neurons [n_swim_cells,]
- medial_neurons: indices of swim-correlated neurons [n_medial_cells,]
- SNR: signal-to-noise ratio for each cell [n_cells,]

- BV_GC_medial: original bivariate (BV) Granger causality results matrix [n_medial_cells,n_medial_cells]
- BV_Fstat_medial: original BV F-statistics matrix [n_medial_cells,n_medial_cells]
- BV_threshold_F_ori: original threshold for the BV F-statistics significance
- BV_threshold_F_new_mat_medial: new threshold customized for each pair of neurons (BV) [n_medial_cells,n_medial_cells]
- BV_Fstat_normalized_medial: new BV F-statistics matrix normalized by customized threshold [n_medial_cells,n_medial_cells]
- BV_GC_normalized_medial: new BV GC results matrix normalized by customized threshold [n_medial_cells,n_medial_cells]

- MV_GC_medial: original multivariate (MV) Granger causality results matrix [n_medial_cells,n_medial_cells]
- MV_Fstat_medial: original MV F-statistics matrix [n_medial_cells,n_medial_cells]
- MV_threshold_F_ori_medial: original threshold for the MV F-statistics significance
- MV_threshold_F_new_mat_medial: new MV F-statistics matrix normalized by customized threshold [n_medial_cells,n_medial_cells]
- MV_Fstat_normalized_medial: new MV F-statistics matrix normalized by customized threshold [n_medial_cells,n_medial_cells]
- MV_GC_normalized_medial: new MV GC results matrix normalized by customized threshold [n_medial_cells,n_medial_cells]