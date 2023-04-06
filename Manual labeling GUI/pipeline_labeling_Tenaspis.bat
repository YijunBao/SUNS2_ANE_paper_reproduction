@REM Semi-automatically label neurons from the Tenaspis dataset.

@REM Use an initial drawing GUI to draw masks by looking at 
@REM either the frame-by-frame video or the summary images.
matlab -nojvm -nodesktop -r ManualLabeling_TENASPIS
@REM Supplement neurons through residual activities using a hierarchical clustering 
@REM algorithm assisted with a manual confirmation GUI.
matlab -nojvm -nodesktop -r use_GUI_find_missing_data_TENASPIS
matlab -nojvm -nodesktop -r merge_masks_TENASPIS
@REM Computationally refine the shapes of the masks generated in the previous two steps by 
@REM averaging their representative frames to better fit the fluorescence profiles of the neurons 
@REM with the assistance of the third GUI
matlab -nojvm -nodesktop -r mask_correction_GUI_data_TENASPIS
matlab -nojvm -nodesktop -r mask_correction_GUI_final_data_TENASPIS
