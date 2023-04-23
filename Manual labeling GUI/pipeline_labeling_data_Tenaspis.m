% Semi-automatically label neurons from the Tenaspis dataset.
%% Initial manual drawing
% Use an initial drawing GUI to draw masks by looking at 
% either the frame-by-frame video or the summary images.
Manual_Labeling_data_TENASPIS
% Merge repeated drawings belonging to the same neuron.
merge_masks_after_manual_data_TENASPIS

%% Semi-automatically add neurons
% Supplement neurons through residual activities using a hierarchical clustering 
% algorithm assisted with a manual confirmation GUI.
clustering_mm_manual_data_TENASPIS
use_GUI_find_missing_data_TENASPIS
% Merge masks added in different spatial patches belonging to the same neuron.
merge_masks_after_add_data_TENASPIS

%% Refine the shapes of previously found neurons
% Combine manually labelled and semi-automatically added masks
combine_manual_add_data_TENASPIS
% Calculate the SNR videos and traces
calculate_traces_bgtraces_SNR_data_TENASPIS
% Computationally refine the shapes of the masks generated in the previous two steps by 
% averaging their representative frames to better fit the fluorescence profiles of the neurons 
% with the assistance of the third GUI
mask_correction_GUI_data_TENASPIS
mask_correction_GUI_final_data_TENASPIS
