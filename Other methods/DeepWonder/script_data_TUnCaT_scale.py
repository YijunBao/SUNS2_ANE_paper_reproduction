import os
import sys

if True:
	folder = 'E:\\1photon-small\\added_refined_masks\\DeepWonder_scale_full'
	list_Exp_ID = [ 'c25_59_228','c27_12_326','c28_83_210',\
                	'c25_163_267','c27_114_176','c28_161_149',\
               		'c25_123_348','c27_122_121','c28_163_244']

	for Exp_ID in list_Exp_ID:
		os.system('python main_RSM_copy.py \
		--RMBG_model_folder arr_202110011541 \
		--RMBG_model_name E_30_Iter_4009 \
		--SEG_model_folder TS3DUnetFFD_20211129-1355 \
		--SEG_model_name seg_30 \
		--test_datasize 20000 \
		--datasets_path "' + folder + '" \
		--datasets_folder ' + Exp_ID + ' \
		--sub_img_s 1500 \
		--sub_img_w ' + sys.argv[1] + ' \
		--GPU 0')
		# --sub_img_w 480 \
		# --sub_img_h 480 \
		# --sub_gap_w 448 \
		# --sub_gap_h 448 \

# os.system('python main_RSM_copy.py \
# --RMBG_model_folder arr_202110011541 \
# --RMBG_model_name E_30_Iter_4009 \
# --SEG_model_folder TS3DUnetFFD_20211129-1355 \
# --SEG_model_name seg_30 \
# --test_datasize 20000 \
# --datasets_path "E:/data_DeepWonder" \
# --datasets_folder widefield \
# --GPU 0')

# os.system('python main_RSM.py \
# --RMBG_model_folder arr_202110011541 \
# --RMBG_model_name E_30_Iter_4009 \
# --SEG_model_folder TS3DUnetFFD_20211129-1355 \
# --SEG_model_name seg_30 \
# --test_datasize 20000 \
# --datasets_path datasets \
# --datasets_folder Simulated_wide_field_data1 \
# --GPU 0')
