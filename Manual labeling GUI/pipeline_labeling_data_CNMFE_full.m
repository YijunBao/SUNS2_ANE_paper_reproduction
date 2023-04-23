% Semi-automatically label neurons from the full videos of CNMF-E dataset.
% We labeld the full videos (https://doi.org/10.5061/dryad.kr17k) stored in fullfile('E:','data_CNMFE').
% We cropped the full videos to quarters later after this pipeline. 
%% Initial manual drawing
% Use an initial drawing GUI to draw masks by looking at 
% either the frame-by-frame video or the summary images.
Manual_Labeling_data_CNMFE_full
% Merge repeated drawings belonging to the same neuron.
merge_masks_after_manual_data_CNMFE_full

%% Semi-automatically add neurons
% Supplement neurons through residual activities using a hierarchical clustering 
% algorithm assisted with a manual confirmation GUI.
clustering_mm_manual_data_CNMFE_full
use_GUI_find_missing_data_CNMFE_full
% Merge masks added in different spatial patches belonging to the same neuron.
merge_masks_after_add_data_CNMFE_full

%% Refine the shapes of previously found neurons
% Combine manually labelled and semi-automatically added masks
combine_manual_add_data_CNMFE_full
% Calculate the SNR videos and traces
calculate_traces_bgtraces_SNR_data_CNMFE_full
% Computationally refine the shapes of the masks generated in the previous two steps by 
% averaging their representative frames to better fit the fluorescence profiles of the neurons 
% with the assistance of the third GUI
mask_correction_GUI_data_CNMFE_full
mask_correction_GUI_final_data_CNMFE_full

