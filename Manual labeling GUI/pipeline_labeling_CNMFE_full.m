% Semi-automatically label neurons from the full videos of CNMF-E dataset.
% We labeld the full videos (https://doi.org/10.5061/dryad.kr17k), and cropped the full videos to quarters. 

% Use an initial drawing GUI to draw masks by looking at 
% either the frame-by-frame video or the summary images.
ManualLabeling_CNMFE_full

% Supplement neurons through residual activities using a hierarchical clustering 
% algorithm assisted with a manual confirmation GUI.
pre_CNN_data_CNMFE_full_manual_blockwise_weighted_sum_mm
use_GUI_find_missing_data_CNMFE_full
merge_masks_CNMFE_full

% Computationally refine the shapes of the masks generated in the previous two steps by 
% averaging their representative frames to better fit the fluorescence profiles of the neurons 
% with the assistance of the third GUI
mask_correction_GUI_data_CNMFE_full
mask_correction_GUI_final_data_CNMFE_full
