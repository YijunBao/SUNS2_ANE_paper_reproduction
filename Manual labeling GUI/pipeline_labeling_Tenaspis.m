% Semi-automatically label neurons from the Tenaspis dataset.

% Use an initial drawing GUI to draw masks by looking at 
% either the frame-by-frame video or the summary images.
ManualLabeling_TENASPIS

% Supplement neurons through residual activities using a hierarchical clustering 
% algorithm assisted with a manual confirmation GUI.
use_GUI_find_missing_data_TENASPIS
merge_masks_TENASPIS

% Computationally refine the shapes of the masks generated in the previous two steps by 
% averaging their representative frames to better fit the fluorescence profiles of the neurons 
% with the assistance of the third GUI
mask_correction_GUI_data_TENASPIS
mask_correction_GUI_final_data_TENASPIS
