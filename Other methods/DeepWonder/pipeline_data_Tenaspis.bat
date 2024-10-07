@REM Tenaspis dataset

@REM Convert h5 video to tiff format, and scale it so that the average neuron areas 
@REM in the unit of pixels equal the average neuron area of 
@REM the DeepWonder segmentation of the widefield video in that paper (368 pixels)
python convert_h5_tiff_scale_data_Tenaspis.py

@REM Run DeepWonder
python script_data_Tenaspis_scale.py 256

@REM Postprocess the DeepWonder result
matlab -nojvm -nodesktop -r "patch_size = 256; DeepWonder_postprocessing_data_Tenaspis_scale; exit"
