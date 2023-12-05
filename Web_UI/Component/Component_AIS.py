import os as OS
import inspect as Inspect
from collections import namedtuple as NamedTuple
import os.path as Pathing
import sys as Environment

import cv2 as Vision
# import trimesh as Meshing
import gradio as Framework
import mmcv as Vision_MM
# from diffusers import StableDiffusionPipeline as Pipeline
import numpy as Number
import torch as Engine

import anime_3dkenburns.kenburns_effect as Anime3D_KenBurns
from anime_3dkenburns.kenburns_effect import KenBurnsPipeline as Pipeline
import Web_UI.Common.Utility_Sampler as Sampler
import Web_UI.Common.Utility_Style as Style
# import Utility_Model as Model
#
# from Web_UI.Common.Utility_Model import Model as Model
from ..Common.Utility_Model import Model as Model


#region Script

# # Debug
# print ( F"{OS.getcwd()=}" )

#endregion Script



# Text → Text Tab
# Field
PlaceHolder_Label_Camera=F"#### Horizon:　(Original:　), Vertical:　(Original:　)"
Pipeline=Pipeline(F"configs/3dkenburns.yaml")
Limit_Minimum=400       # It is better that does not less than 60% of original Width | Height
Limit_Maxmum=4000
Path_List_Sample=F"examples"
# Pipeline=Pipeline()
#
Configuration_AIS = Pipeline.cfg
Device = Pipeline.device

# Global Carrier
Image_Carrier = None
Image_Tensor_Carrier = None
Instance_Carrier = None
Image_Intermediate_Carrier = None
Configuration_Argument_Carrier=None

IsIntermediateResultSaving = False
IsAllActivatedControlDisplayed=False
Ratio_Crop_Camera_Start_Deault=0.6
Ratio_Crop_Camera_End_Deault=0.6
Ratio_Crop_Minimum=0
Ratio_Crop_AutoZoomed=0.97
Mask_α_Default=0.7
Progress_Default=F"Display the Progress"

IsVerbosing_Global=False
Height_Global=None
Width_Global=None
IsStartingCameraSetting_Global=True
Location_Point=NamedTuple(F"Point",F"Horizon Vertical")
Camera_Start_Global=None
Camera_End_Global=None
Ratio_Global=None
Configuration_Global=None
List_Frame_Global=None
Ratio_Resize_Global=None
Page_Global = None


# Common control for multiple tabs
Create_CheckPoint_ComboBox=lambda:Framework.Dropdown\
(
	choices = Model.List_CheckPointTile (),
	label = F"Stable Diffusion Check-Point",
	# elem_id = F"ComboBox_CheckPoint",
)
# Combo Box无法自行完成对File System中的Content的刷新、自行检查，需要手动处理
Create_RelaunchModelList_Button=lambda:Framework.Button\
(
	# ！考虑改为Icon的Button
	F"Relaunch",
	# show_label = False,
	# lines = 2,	  # As known as Height
	# placeholder = "Negative prompt | input here (Ctrl | Alt + Enter or Button \"Generate\" to work )",
)
Create_CheckPointLoadingStatus_Label=lambda:Framework.Label\
(
	# choices = Model.List_CheckPointTile (),
	label = F"Check-Point Loading Status",
	value =F"None Model has been loaded."	 # 相当于default
	# show_label = False,
	# visible = False,
	# elem_id = F"ComboBox_CheckPoint",
)
Create_Main_Textbox=lambda:Framework.Textbox\
(
	label = "Prompt | Input",
	show_label = False,
	lines = 3,	  # As known as Height
	placeholder = "Prompt | input here (Ctrl | Alt + Enter or Button \"Generate\" to work )",

	# 设置上限+当前字符量额统计、限制

	# Debug
	# value = F"Bill Gates rides on a horse."
)
Create_Negative_TextBox=lambda:Framework.Textbox\
(
	label = "Negative prompt | input",
	show_label = False,
	lines = 2,	  # As known as Height
	placeholder = "Negative prompt | input here (Ctrl | Alt + Enter or Button \"Generate\" to work )",	  # !Need to be classified
)
Create_Interrupt_Button=lambda:Framework.Button\
(
	F"Interrogate CLIP",
	# show_label = False,
	# lines = 2,	  # As known as Height
	# placeholder = "Negative prompt | input here (Ctrl | Alt + Enter or Button \"Generate\" to work )",
)
Create_Skip_Button=lambda:Framework.Button\
(
	F"Interrogate CLIP",
	# show_label = False,
	# lines = 2,	  # As known as Height
	# placeholder = "Negative prompt | input here (Ctrl | Alt + Enter or Button \"Generate\" to work )",
)
Create_Generate_Button=lambda:Framework.Button\
(
	F"Generate",
	# show_label = False,
	# lines = 2,	  # As known as Height
	# placeholder = "Negative prompt | input here (Ctrl | Alt + Enter or Button \"Generate\" to work )",
	variant = "primary",
)
# Pending
Create_Clear_Button=lambda:Framework.Button\
(
	F"Clear Prompted | Input",
	# show_label = False,
	# lines = 2,	  # As known as Height
	# placeholder = "Negative prompt | input here (Ctrl | Alt + Enter or Button \"Generate\" to work )",
	# variant = "primary",
)
Create_Style_ComboBox=lambda:Framework.Dropdown\
(
	choices = Style.List_Style () ,
	value = [],	 # 相当于default
	multiselect = True,
	label = F"Style List",
)



# Control for Image → Image Tab
Create_Normal_Input_ImageBox=lambda:Framework.Image \
(
	label = "Image → Image",
)
Create_Sketch_Input_ImageBox=lambda:Framework.Image \
(
	label = "Sketch",
)
Create_Inpaint_Input_ImageBox=lambda:Framework.Image \
(
	label = "Inpaint",
)
Create_InpaintSketch_Input_ImageBox=lambda:Framework.Image \
(
	label = "Inpaint Sketch",
)
Create_InpaintUpload_Input_ImageBox=lambda:Framework.Image \
(
	label = "Inpaint Upload",
)
Create_Batch_Input_ImageBox=lambda:Framework.Image \
(
	label = "Batch",
)
Create_InterrogateCLIP_Button=lambda:Framework.Button\
(
	F"Interrogate CLIP",
	# show_label = False,
	# lines = 2,	  # As known as Height
	# placeholder = "Negative prompt | input here (Ctrl | Alt + Enter or Button \"Generate\" to work )",
)
Create_Interrogate_DeepBooru_Button=lambda:Framework.Button\
(
	F"Interrogate  DeepBooru",
	# show_label = False,
	# lines = 2,	  # As known as Height
	# placeholder = "Negative prompt | input here (Ctrl | Alt + Enter or Button \"Generate\" to work )",
)

# Pending to check
Create_ResizeMode_RadioGroup=lambda:Framework.Radio\
(
	choices =
	[
		F"Just resize" ,
		F"Crop and resize" ,
		F"Resize and fill" ,
		F"Just resize (Latent up-scale)" ,
	],
	# value = [],	 # 相当于default
	# multiselect = True,
	label = F"Resize mode",
)
Create_MaskBlur_Slider=lambda:Framework.Slider\
(
	# choices = Sampler.List_Sampler () ,
	value = 4,	 # 相当于default
	minimum = 0,
	maximum = 64,
	step = 1,
	# multiselect = True,
	label = F"Mask blur",
)
Create_MaskTransparency=lambda:Framework.Slider\
(
	# choices = Sampler.List_Sampler () ,
	value = 0,	 # 相当于default
	minimum = 0,
	maximum = 100,
	step = 1,
	# multiselect = True,
	label = F"Mask transparency",
)
Create_MaskMode_RadioGroup=lambda:Framework.Radio\
(
	choices =
	[
		F"Inpaint masked",
		F"Inpaint \033[1mnot\033[0m masked" ,  # !Need to be optimised
	],
	# value = [],	 # 相当于default
	# multiselect = True,
	label = F"Mask mode",
)
Create_MaskedContent_RadioGroup=lambda:Framework.Radio\
(
	choices =
	[
		F"Fill",
		F"Original",
		F"Latent noise",
		F"Latent nothing",
	],
	# value = [],	 # 相当于default
	# multiselect = True,
	label = F"Masked content",
)
Create_InpaintArea_RadioGroup=lambda:Framework.Radio\
(
	choices =
	[
		F"Whole picture",
		F"Only Masked",
	],
	# value = [],	 # 相当于default
	# multiselect = True,
	label = F"Inpaint area",
)
Create_OnlyMaskedPadding_Pixel_Slider=lambda:Framework.Slider\
(
	# choices = Style.List_Style () ,
	value = 32,	 # 相当于default
	minimum = 0,
	maximum = 256,
	step = 4,
	# multiselect = True,
	label = F"Only masked padding (Pixel)",
)
Create_Sampler_ComboBox=lambda:Framework.Dropdown\
(
	choices = Sampler.List_Sampler () ,
	value = [],	 # 相当于default
	# multiselect = True,
	label = F"Sampler | Sampling Method",
)
Create_SamplingStep_Slider=lambda:Framework.Slider\
(
	# choices = Style.List_Style () ,
	value = 20,	 # 相当于default
	minimum = 1,
	maximum = 150,
	step = 1,
	# multiselect = True,
	label = F"Sampling Step",
)
Create_CheckBox_Group=lambda :Framework.CheckboxGroup \
(
	choices =
	[
		F"Restore Face" ,
		F"Tiling",
		F"High Resolution Fix",
	],
	value = F"",
)
Create_RestoreFace_CheckBox=lambda:Framework.Checkbox\
(
	# choices = Style.List_Style () ,
	# value = 20,	 # 相当于default
	# multiselect = True,
	label = F"Restore Face",
)
Create_Tiling_CheckBox=lambda:Framework.Checkbox\
(
	# choices = Style.List_Style () ,
	# value = 20,	 # 相当于default
	# multiselect = True,
	label = F"Tiling",
)
Create_HighResolutionFix_CheckBox=lambda:Framework.Checkbox\
(
	# choices = Style.List_Style () ,
	# value = 20,	 # 相当于default
	# multiselect = True,
	label = F"High Resolution Fix",
)
Create_Step_CheckBox=lambda Name_In,Value_Default_In,Tip_In=None,IsVisible_In=True:Framework.Checkbox\
(
	# choices = Style.List_Style () ,
	# value = 20,	 # 相当于default
	# multiselect = True,
	label = Name_In,
	value = Value_Default_In,
	info = Tip_In,
	visible = IsVisible_In,
)
Create_Width_Slider=lambda:Framework.Slider\
(
	# choices = Style.List_Style () ,
	value = 512,	 # 相当于default
	minimum = 64,
	maximum = 2048,
	step = 8,
	# multiselect = True,
	label = F"Width",
)
Create_Height_Slider=lambda:Framework.Slider\
(
	# choices = Style.List_Style () ,
	value = 512,	 # 相当于default
	minimum = 64,
	maximum = 2048,
	step = 8,
	# multiselect = True,
	label = F"Height",
)
# Or called Switch
Create_Swap_Button=lambda:Framework.Button\
(
	# choices = Style.List_Style () ,
	# value = 512,	 # 相当于default
	# multiselect = True,
	F"Swap",
)
Create_BatchCount_Slider=lambda:Framework.Slider\
(
	# choices = Style.List_Style () ,
	value = 1,	 # 相当于default
	minimum = 1,
	# maximum = 2048,
	step = 1,
	# multiselect = True,
	label = F"Batch Count",
)
Create_BatchSize_Slider=lambda:Framework.Slider\
(
	# choices = Style.List_Style () ,
	value = 1,	 # 相当于default
	minimum = 1,
	maximum = 8,
	step = 1,
	# multiselect = True,
	label = F"Batch Size",
)
Create_CFG_Scale_Slider=lambda:Framework.Slider\
(
	# choices = Style.List_Style () ,
	value = 7.0,	 # 相当于default
	minimum = 1.0,
	maximum = 30.0,
	step = 0.5,
	# multiselect = True,
	label = F"CFG Scale",
)
Create_DenoisingStrength_Slider=lambda:Framework.Slider\
(
	# choices = Style.List_Style () ,
	value = 0.75,	 # 相当于default
	minimum = 0,
	maximum = 1,
	step = 0.01,
	# multiselect = True,
	label = F"Denoising strength",
)
Create_Seed_TextBox=lambda:Framework.Textbox\
(
	# choices = Style.List_Style () ,
	value = F"-1",	 # 相当于default
	# multiselect = True,
	label = F"Seed",
)
Create_AdditionalScript_ComboBox=lambda:Framework.Dropdown\
(
	# choices = Style.List_Style () ,
	value = F"None",	 # 相当于default
	# multiselect = True,
	label = F"Additional Script",
)
Create_Output_BatchedImageBox=lambda \
	Name_In=F"Output",\
	Type_In=F"pil",\
	Number_Column=None,\
	Object_Fit_In=F"contain", \
	IsPreviewMode_In=False,\
	IsLabelDisplayed_In=False,\
	Height_In=F"auto",\
:\
	Framework.Gallery\
	(
		# choices = Style.List_Style () ,
		# value = F"None",	 # 相当于default
		# multiselect = True,
		label = Name_In,
		show_label = IsLabelDisplayed_In,	 # 的确不好看，更像是个Holder Tool Tip
		# type = Type_In,
		# select=1,
		# _css= "overflow-x: scroll; overflow-y: hidden;",
		columns = Number_Column,
		object_fit = Object_Fit_In ,
		preview = IsPreviewMode_In,
		height =Height_In,
	)
Create_Output_ImageBox_Single=lambda:Framework.Image\
(
	# choices = Style.List_Style () ,
	# value = F"None",	 # 相当于default
	# multiselect = True,
	label = F"Output",
	show_label = False,	 # 的确不好看，更像是个Holder Tool Tip
	type = "pil",
)
Create_OpenInExplorer_Button=lambda:Framework.Button\
(
	# choices = Style.List_Style () ,
	# value = F"None",	 # 相当于default
	# multiselect = True,
	F"Open in ...",
)
Create_OriginalSaveLink_Button=lambda:Framework.Button\
(
	# choices = Style.List_Style () ,
	# value = F"None",	 # 相当于default
	# multiselect = True,
	F"Save Link",
)
Create_ZipSaveLink_Button=lambda:Framework.Button\
(
	# choices = Style.List_Style () ,
	# value = F"None",	 # 相当于default
	# multiselect = True,
	label = F"Zip Link",
)
Create_ApplySetting_Button=lambda:Framework.Button\
(
	# choices = Style.List_Style () ,
	# value = F"None",	 # 相当于default
	# multiselect = True,
	F"Apply Setting",
)
Create_Restart_Button=lambda:Framework.Button\
(
	# choices = Style.List_Style () ,
	# value = F"None",	 # 相当于default
	# multiselect = True,
	F"Restart",
)

#region Function | Method

#region Event Handler

def DoOnButtonSubmit_AIS_Instance_Segmentation_Clicking (Image_In:Number.ndarray) :
	"""
	The 1st Process

	Args:
		Image_In ():

	Returns:
	"""

	# Pre-Processing
	# Pre-Setting
	Instance_Processing = None
	Image_Processing = None

	if isinstance ( Image_In , str ) :
		Image_Processing = Vision_MM.imread ( Image_In )
	else:
		Image_Processing=Image_In

	with Engine.no_grad () :
		if Instance_Processing is None :
			Instance_Processing , Image_Return = Pipeline.run_instance_segmentation \
			(
				Image_Processing ,
				scale_down_to_maxsize = False ,
			)
		else :
			pass

		Anime3D_KenBurns.torch_gc ()

	global Image_Carrier
	Image_Carrier = Image_Processing

	global Instance_Carrier
	Instance_Carrier = Instance_Processing

	return Image_Return

def DoOnButtonSubmit_AIS_Infer_Disparity_Depth_Estimation_Clicking () :
	"""
	The 2nd Process

	Returns:
	"""

	# Pre-Processing
	# Pre-Declaring
	global Image_Carrier
	global Instance_Carrier

	with Engine.no_grad () :
		Image_Processing = Anime3D_KenBurns.scaledown_maxsize\
		(
			Image_Carrier ,
			Configuration_AIS.max_size,
		)

		Instance_Carrier.resize \
		(
			Image_Processing.shape [ 0 ] ,
			Image_Processing.shape [ 1 ] ,
		)

		Configuration_AIS.int_height , Configuration_AIS.int_width = Image_Processing.shape [ : 2 ]

		Image_Tensor=Engine.FloatTensor\
		(
			Number.ascontiguousarray\
			(
				Image_Processing\
					.transpose(2,0,1)\
					[None,:,:,:]\
					.astype(Number.float32)\
				*(1.0/255.0),
			),
		)\
			.to(Device)

		Configuration_Argument:Anime3D_KenBurns.KenBurnsConfig=Configuration_AIS.copy()

		if Instance_Carrier is None :
			# ！此处的Instance_Carrier指向是否为Global需要测试
			Instance_Carrier , Image_Processing = Pipeline.run_instance_segmentation\
			(
				Image_Processing ,
				scale_down_to_maxsize = False,
			)

			Anime3D_KenBurns.torch_gc ()
		else:
			pass

		# Coarse Depth
		if Image_Tensor is None :
			Image_Tensor = Engine.FloatTensor\
			(
				Number.ascontiguousarray\
				(
					Image_Processing.transpose ( 2 , 0 , 1 )\
						[ None , : , : , : ]\
						.astype ( Number.float32 )\
					* (1.0 / 255.0)
				)
			)\
				.to ( Device )
		else:
			pass

		Disparity_Return = Pipeline._depth_est ( Image_Tensor , Image_Processing )

		Anime3D_KenBurns.torch_gc ()

		# Debug
		print(F"Disparity generated successfully")

	# # global Image_Intermediate_Carrier
	# # Image_Intermediate_Carrier=Disparity_Return
	#
	global Image_Tensor_Carrier
	Image_Tensor_Carrier=Disparity_Return
	#
	# global Image_Carrier
	Image_Carrier=Image_Processing
	#
	global Disparity_Carrier
	Disparity_Carrier=Disparity_Return
	#
	# global Instance_Carrier
	# Instance_Carrier=Instance_Processing
	#
	global Configuration_Argument_Carrier
	Configuration_Argument_Carrier=Configuration_Argument

	return Disparity_Return

# ？和前1函数合并，共用Depth Coarse的名，保持和后面的一致
def DoOnButtonSubmit_AIS_Infer_Disparity_Depth_Coarse_Clicking () :
	"""
	The 3rd Process

	Returns:
	"""

	with Engine.no_grad () :
		Image_With_Depth_Coarse_Return = Anime3D_KenBurns.colorize_depth\
		(
			Disparity_Carrier\
				.cpu ()\
				.numpy () ,
			inverse = True ,
			rgb2bgr = True ,
			cmap = 'magma_r',
		)

		if Configuration_Argument_Carrier is not None :
			Configuration_Argument_Carrier.stage_depth_coarse = Image_With_Depth_Coarse_Return
		else:
			pass

		if IsIntermediateResultSaving :
			# To Do
			pass
		else:
			pass

		Anime3D_KenBurns.torch_gc ()

	# global Image_Intermediate_Carrier
	# Image_Intermediate_Carrier=Image_With_Depth_Coarse_Return

	return Image_With_Depth_Coarse_Return

def DoOnButtonSubmit_AIS_Infer_Disparity_Depth_Adjusted_Depth_Clicking () :
	"""
	The 4th Process

	Returns:
	"""

	# Pre-Processing
	# Pre-Declaring
	global Disparity_Carrier

	with Engine.no_grad () :
		# Adjust Depth
		if Configuration_AIS.detector == F"maskrcnn" :
			Disparity_Return = Anime3D_KenBurns.depth_adjustment_maskrcnn\
			(
				Instance_Carrier ,
				Disparity_Carrier ,
				Image_Tensor_Carrier,
			)
		else :
			Disparity_Return = Anime3D_KenBurns.depth_adjustment_animesseg\
			(
				Instance_Carrier ,
				Disparity_Carrier ,
				Image_Tensor_Carrier,
			)

		Image_With_Depth_Adjusted_Return = Anime3D_KenBurns.colorize_depth\
		(
			Disparity_Return\
				.cpu ()\
				.numpy () ,
			inverse = True ,
			rgb2bgr = True ,
			cmap = 'magma_r',
		)

		if Configuration_Argument_Carrier is not None :
			Configuration_Argument_Carrier.stage_depth_adjusted = Image_With_Depth_Adjusted_Return
		else:
			pass

		if IsIntermediateResultSaving :
			# To Do
			pass
		else:
			pass

		Anime3D_KenBurns.torch_gc ()

	# global Disparity_Carrier
	Disparity_Carrier=Disparity_Return

	return\
	[
		Image_With_Depth_Adjusted_Return,
		Disparity_Return,
	]

def DoOnButtonSubmit_AIS_Adjustment_Anime_Segmentation_TabClicking () :
	"""
	The 5th Process

	Returns:
	"""

	# Pre-Processing
	# Pre-Declaring
	global Disparity_Carrier

	with Engine.no_grad () :
		# Final Depth
		if Configuration_AIS.default_depth_refine :
			Disparity_Return = Pipeline.refine_depth\
			(
				Image_Tensor_Carrier ,
				Disparity_Carrier,
			)
		elif Configuration_AIS.refine_crf :
			Disparity_Return = Pipeline.refine_depth_crf\
			(
				Image_Carrier ,
				Disparity_Carrier ,
				Instance_Carrier,
			)
		else:
			pass

		Image_With_Depth_Final_Return = Anime3D_KenBurns.colorize_depth\
		(
			Disparity_Return\
				.cpu ()\
				.numpy () ,
			inverse = True ,
			rgb2bgr = True ,
			cmap = 'magma_r',
		)

		if Configuration_Argument_Carrier is not None :
			Configuration_Argument_Carrier.stage_depth_final = Image_With_Depth_Final_Return
		else:
			pass

		if IsIntermediateResultSaving :
			# To Do
			pass
		else:
			pass

		Anime3D_KenBurns.torch_gc ()

	# global Disparity_Carrier
	Disparity_Carrier=Disparity_Return

	return\
	[
		Image_With_Depth_Final_Return,
		Disparity_Return,
	]


def DoOnButton_Run_With_Steps_Stepping_TabClicking\
(
	Image_In:Number.ndarray,
	IsBoundingBox_Required_In,
	IsInstanceMask_Required_In,
	IsInstanceContour_Required_In,
	IsTag_List_Required_In,
	Mask_α_In,
) :

	global Configuration_Global

	IsVerbosing_In=IsVerbosing_Global

	# Debug
	print(F"{IsVerbosing_In=} in {Inspect.currentframe().f_code.co_name}()")

	List_Return=[]
	#
	Carrier_Configuration = Pipeline.generate_kenburns_config\
	(
		img = Image_In ,
		verbose = IsVerbosing_In ,
		# savep = None,	 # Gradio的File Control提供了Download，此Argument意义不大
	)

	# Debug
	print(F"Scale Down in {Inspect.currentframe().f_code.co_name} after Configure Ken Burns")
	print(F"{Image_In.shape=}")
	print(F"{Carrier_Configuration.int_width=},{Carrier_Configuration.int_height=}")
	print(F"Ratio: {Carrier_Configuration.int_width/Carrier_Configuration.int_height}")
	print(F"Ratio Global | Original as contrast: {Ratio_Global=}")

	Image_Return_After_2nd_Process = Carrier_Configuration.instances.draw_instances\
	(
		img=Image_In ,
		draw_bbox = IsBoundingBox_Required_In,
		draw_ins_mask = IsInstanceMask_Required_In,
		draw_ins_contour = IsInstanceContour_Required_In,
		# ！需要保持None，Empty不行
		# draw_indices =\
		# [
		# 	# List of indice
		# ],
		draw_tags = IsTag_List_Required_In ,
		mask_alpha=Mask_α_In,
	)

	# Debug
	print(F"Scale Down in {Inspect.currentframe().f_code.co_name} after Draw Instance")
	print(F"{Image_In.shape=}")
	print(F"{Carrier_Configuration.int_width=},{Carrier_Configuration.int_height=}")
	print(F"Ratio: {Carrier_Configuration.int_width/Carrier_Configuration.int_height}")
	print(F"Ratio Global | Original as contrast: {Ratio_Global=}")

	List_Return.append(Image_Return_After_2nd_Process )

	# ！此处为了抵消AIS代码中预先对Image进行的BGR→RGB，进行了1次转换
	Image_Return_Aftewr_3rd_Process=Vision.cvtColor(Carrier_Configuration.stage_depth_coarse, Vision.COLOR_BGR2RGB)
	#
	List_Return.append(Image_Return_Aftewr_3rd_Process)

	# ！此处为了抵消AIS代码中预先对Image进行的BGR→RGB，进行了1次转换
	Image_Return_Aftewr_4th_Process=Vision.cvtColor(Carrier_Configuration.stage_depth_adjusted, Vision.COLOR_BGR2RGB)
	#
	List_Return.append(Image_Return_Aftewr_4th_Process)

	# ！此处为了抵消AIS代码中预先对Image进行的BGR→RGB，进行了1次转换
	Image_Return_Aftewr_5th_Process=Vision.cvtColor(Carrier_Configuration.stage_depth_final, Vision.COLOR_BGR2RGB)
	#
	List_Return.append(Image_Return_Aftewr_5th_Process)


	# Generate High Light Crops for Auto-zoomed Method
	List_Return.append(HighLight_Crop_AutoZoom(Image_In,Carrier_Configuration))
	# List_Return.append(Image_In)


	# Update Control status
	List_Return.append(Framework.update(interactive = True))

	#
	#

	# Update Progress
	List_Return.append(F"Instances generated successfully.")

	# Post-Processing
	Configuration_Global=Carrier_Configuration

	return List_Return


def Adjust_Camera_Location ( Location_Camera_Original_In:Location_Point , Difference_Width_Original_In =0,Ratio_Scaling_In=1) :

	# ！需要使用更简洁的方式设定非null的default
	# Location_Camera_Original_Return:Location_Point=None
	# Location_Central_Original=Location_Point(Approximate_To_Integer( Width_Global * Ratio_Resize_Global/2),Approximate_To_Integer( Height_Global * Ratio_Resize_Global/2 ) )
	# Difference_Height_Original=Approximate_To_Integer(Difference_Width_Original_In/Ratio_Global)
	# Horizon_Carrier=None
	Horizon_Carrier=Approximate_To_Integer(Location_Camera_Original_In.Horizon*Ratio_Scaling_In)
	# Vertical_Carrier=None
	Vertical_Carrier=Approximate_To_Integer(Location_Camera_Original_In.Vertical*Ratio_Scaling_In)

	# if(Location_Camera_Original_In.Horizon<Location_Central_Original.Horizon):
	# 	Horizon_Carrier=Location_Camera_Original_In.Horizon+Difference_Width_Original_In
	# else:
	# 	Horizon_Carrier=Location_Camera_Original_In.Horizon-Difference_Width_Original_In
	#
	# if(Location_Camera_Original_In.Vertical<Location_Central_Original.Vertical):
	# 	Vertical_Carrier=Location_Camera_Original_In.Vertical+Difference_Height_Original
	# else:
	# 	Vertical_Carrier=Location_Camera_Original_In.Vertical-Difference_Height_Original

	Location_Camera_Original_Return=Location_Point(Horizon_Carrier,Vertical_Carrier)

	return Location_Camera_Original_Return


def Adjust_Crop\
(
	Location_Camera_In:Location_Point,
	Crop_In:Location_Point,     # ！Crop应另设Size的Named Tuple，以示区分
	Ratio_Resize_Image_In=1,
	Limit_Ratio_Resize_Crop_In=1,
) :

	Crop_Return:Location_Point=None
	Location_Camera_Return:Location_Point=None
	Crop_Processing:Location_Point=None
	Size_Original=Location_Point ( Approximate_To_Integer ( Width_Global * Ratio_Resize_Image_In ) , Approximate_To_Integer ( Height_Global * Ratio_Resize_Image_In ) )

	# Processing: Adjust Crop Width with boundary
	# # # Debug
	# # print(F"Before generating Crop based on Width in {Inspect.currentframe().f_code.co_name}()")
	# # print(F"{Crop_In=}" )
	# # print(F"Ratio: {Crop_In.Horizon / Crop_In.Vertical}" )
	# # print(F"Ratio Global | Original as contrast: {Ratio_Global=}")
	#
	# Crop_Width_Based_On_Width=Adjust_Crop_Core(Location_In=Location_Camera_In.Horizon,Length_In=Crop_In.Horizon, IsWidth_In = True,Ratio_Resize_In = Ratio_Resize_Image_In )
	# Crop_Height_Based_On_Width=Approximate_To_Integer(Crop_Width_Based_On_Width/Ratio_Global)
	#
	# # # Debug
	# # print(F"After generating Crop based on Width in {Inspect.currentframe().f_code.co_name}()")
	# # print(F"{Crop_Width_Based_On_Width=},{Crop_Height_Based_On_Width=}")
	# # print(F"Ratio: {Crop_Width_Based_On_Width/Crop_Height_Based_On_Width}")
	# # print(F"Ratio Global | Original as contrast: {Ratio_Global=}")
	#
	# Crop_Height_Based_On_Height=Adjust_Crop_Core(Location_In=Location_Camera_In.Vertical,Length_In=Crop_In.Vertical, IsWidth_In = False,Ratio_Resize_In = Ratio_Resize_Image_In )
	# Crop_Width_Based_On_Height=Approximate_To_Integer(Crop_Height_Based_On_Height*Ratio_Global)
	#
	# # # Debug
	# # print(F"After generating Crop based on Height in {Inspect.currentframe().f_code.co_name}()")
	# # print(F"{Crop_Width_Based_On_Height=},{Crop_Height_Based_On_Height=}")
	# # print(F"Ratio: {Crop_Width_Based_On_Height/Crop_Height_Based_On_Height}")
	# # print(F"Ratio Global | Original as contrast: {Ratio_Global=}")
	#
	# if(Crop_Width_Based_On_Height<=Crop_Width_Based_On_Width\
	# 	and Crop_Height_Based_On_Height<=Crop_Height_Based_On_Width):        # Crop的高度更接近边缘，以高度为准获取Crop
	# 	Crop_Processing=Location_Point(Crop_Width_Based_On_Height,Crop_Height_Based_On_Height )
	# elif(Crop_Width_Based_On_Width<=Crop_Width_Based_On_Height\
	# 	and Crop_Height_Based_On_Width<=Crop_Height_Based_On_Height):        # Crop的宽度更接近边缘，以宽度为准获取Crop
	# 	Crop_Processing=Location_Point(Crop_Width_Based_On_Width,Crop_Height_Based_On_Width )
	# else:       #  意外值，需要完全消除的分支
	# 	print(F"Stuck in unknown Branch:\n")
	# 	print(F"{Crop_Width_Based_On_Width=}")
	# 	print(F"{Crop_Width_Based_On_Height=}")
	# 	print(F"{Crop_Height_Based_On_Width=}")
	# 	print(F"{Crop_Height_Based_On_Height=}")
	#
	# 	pass

	# # Debug
	# print(F"After generating Crop-Processing in {Inspect.currentframe().f_code.co_name}()")
	# print(F"{Crop_Processing.Horizon=},{Crop_Processing.Vertical=}")
	# print(F"Ratio: {Crop_Processing.Horizon/Crop_Processing.Vertical}")
	# print(F"Ratio Global | Original as contrast: {Ratio_Global=}")

	# Processing: Adjust Camera Location with Crop Ratio Limit
	Limit_Lower=Approximate_To_Integer( Width_Global * Ratio_Resize_Image_In * Limit_Ratio_Resize_Crop_In )

	# Adjust Crop basically, ∈ (Minimum Size, Image Size)
	if Crop_In.Horizon<Limit_Lower:
		Crop_Processing=Location_Point(Limit_Lower,Approximate_To_Integer(Limit_Lower/Ratio_Global))
	elif Crop_In.Horizon>Size_Original.Horizon:
		Crop_Processing = Size_Original
	else:
		Crop_Processing=Crop_In

	# # Debug
	# print(F"{Inspect.currentframe().f_code.co_name}()")
	# print(F"{Limit_Lower=}" )

	# if(Crop_Processing.Horizon<Limit_Lower):
		# # Debug
		# print(F"After generating Crop based on Height in {Inspect.currentframe().f_code.co_name}()")
		# print(F"{Crop_Width_Based_On_Height=},{Crop_Height_Based_On_Height=}")
		# print(F"Ratio: {Crop_Width_Based_On_Height/Crop_Height_Based_On_Height}")
		# print(F"Ratio Global | Original as contrast: {Ratio_Global=}")

		# # Difference_Width=Limit_Lower-Crop_Processing.Horizon
		# Ratio_Scaling=Limit_Lower / Crop_Processing.Horizon     # 始终＞1

		# # Debug
		# print(F"{Ratio_Scaling=}" )

		# Crop_Processing=Location_Point(Limit_Lower,Approximate_To_Integer(Limit_Lower/Ratio_Global))

		# # Debug
		# print(F"After generating Crop-Processing in {Inspect.currentframe().f_code.co_name}()")
		# print(F"{Crop_Processing.Horizon=},{Crop_Processing.Vertical=}")
		# print(F"Ratio: {Crop_Processing.Horizon/Crop_Processing.Vertical}")
		# print(F"Ratio Global | Original as contrast: {Ratio_Global=}")

		# # Location_Camera_Return= Adjust_Camera_Location ( Location_Camera_In , Difference_Width )
		# Location_Camera_Return=  Adjust_Camera_Location ( Location_Camera_In,Ratio_Scaling_In = Ratio_Scaling )
	Location_Camera_Return=Centralise_Camera(Location_Camera_In,Crop_Processing,Size_Original)

		# # Debug
		# print(F"After generating Camera Location in {Inspect.currentframe().f_code.co_name}()")
		# print(F"{Location_Camera_Return.Horizon=},{Location_Camera_Return.Vertical=}")
		# print(F"Ratio: {Crop_Processing.Horizon/Crop_Processing.Vertical}")
		# print(F"Ratio Global | Original as contrast: {Ratio_Global=}")
	# else:
	# 	Location_Camera_Return=Location_Camera_In

	Crop_Return=Crop_Processing

	return\
	[
		Location_Camera_Return,
		Crop_Return,
	]

def Adjust_Crop_Core ( Location_In,Length_In , IsWidth_In, Ratio_Resize_In=1 ) :

	Limit_Lower=0
	Limit_Higher=None
	Difference_Lower=Location_In-Limit_Lower
	Difference_Higher=None
	Crop_Return=None

	if(IsWidth_In):
		Limit_Higher=Approximate_To_Integer(Width_Global*Ratio_Resize_In)
	else:
		Limit_Higher=Approximate_To_Integer(Height_Global*Ratio_Resize_In)

	Difference_Higher = Limit_Higher - Location_In
	Limit_Crop=min ( Difference_Lower , Difference_Higher ) * 2
	Crop_Return =min(Limit_Crop,Length_In)

	return Crop_Return


IsFullImageRequired=lambda Width_In,Ratio_Resize_In=1 :\
	True\
		if Width_In==Approximate_To_Integer(Width_Global*Ratio_Resize_In)\
		else False
		# ！需要重新规划备份代码的位置
		# if Width_In==Width_Global\
		# 	or Width_In==0\


def DoOnButton_Generate_Camera_View_Stepping_TabClicking\
(
	IsInpainting_In,
	Ratio_Crop_Camera_Start_In,
	# Width_Camera_Start_In,
	# Height_Camera_Start_In,
	Ratio_Crop_Camera_End_In,
	# Width_Camera_End_In,
	# Height_Camera_End_In,
	Number_Frame_In,
	IsUsingAutoZoom_In,
	Depthor_In,
	IsDepthRequired_In,
) :
	global Configuration_Global
	global List_Frame_Global

	IsVerbosing_In=IsVerbosing_Global

	# Debug
	print(F"{IsVerbosing_In=} in {Inspect.currentframe().f_code.co_name}()")

	List_PositiveFilm_Return=[]
	List_Mask_Return=[]
	List_Frame_Return=None
	Configuration_Carrier=None

	if (IsUsingAutoZoom_In):
		# Debug
		print(F"Scale Down in {Inspect.currentframe().f_code.co_name} before Auto Zoom")
		# print(F"{Image_In.shape=}")
		print(F"{Configuration_Global.int_width=},{Configuration_Global.int_height=}")
		print(F"Ratio: {Configuration_Global.int_width/Configuration_Global.int_height}")
		print(F"Ratio Global | Original as contrast: {Ratio_Global=}")

		with Engine.no_grad () :
			Crop_Camera_Start=Location_Point(Configuration_Global.int_width / 2.0,Configuration_Global.int_height / 2.0)
			Crop_Camera_End=Location_Point(Configuration_Global.int_width / 2.0,Configuration_Global.int_height / 2.0)
			Camera_Start =\
			{
				'fltCenterU' : Crop_Camera_Start.Horizon ,
				'fltCenterV' : Crop_Camera_Start.Vertical ,
				'intCropWidth' : Approximate_To_Integer(Ratio_Crop_AutoZoomed*Configuration_Global.int_width),
				'intCropHeight' : Approximate_To_Integer(Ratio_Crop_AutoZoomed*Configuration_Global.int_height),
			}

			Camera_End = Anime3D_KenBurns.process_autozoom\
			(
				{
					'fltShift' : 100.0 ,
					'fltZoom' : 1.25 ,
					'objFrom' : Camera_Start,
				} ,
				Configuration_Global,
			)

			# Debug
			print(F"Scale Down in {Inspect.currentframe().f_code.co_name} before Process Ken Burns")
			# print(F"{Image_In.shape=}")
			print(F"{Configuration_Global.int_width=},{Configuration_Global.int_height=}")
			# print(F"{Camera_Start_Global.Horizon=},{Camera_Start_Global.Vertical=}")
			# print(F"{Camera_End_Global.Horizon=},{Camera_End_Global.Vertical=}")
			print(F"{Camera_Start=}")
			print(F"{Camera_End=}")
			print(F"{Crop_Camera_Start=}")
			print(F"{Crop_Camera_End=}")
			print(F"Ratio: {Configuration_Global.int_width/Configuration_Global.int_height}")
			print(F"Ratio Global | Original as contrast: {Ratio_Global=}")

			List_Frame_Return,Configuration_Carrier = Pipeline.process_kenburns\
			(
				objSettings =\
				{
					'fltSteps' : Number.linspace\
					(
						start = 0.0 ,
						stop = 1.0 ,
						num = 75,
					)\
						.tolist () ,
					'objFrom' : Camera_Start ,
					'objTo' : Camera_End ,
					'boolInpaint' : IsInpainting_In,
				} ,
				objCommon = Configuration_Global ,
				inpaint = IsInpainting_In ,
				verbose = IsVerbosing_In,
				# Depthor_In = Depthor_In,
				# IsDepthRequired_In = IsDepthRequired_In,
			)

		# Debug
		print(F"Scale Down in {Inspect.currentframe().f_code.co_name} after Auto Zoom")
		# print(F"{Image_In.shape=}")
		print(F"{Configuration_Global.int_width=},{Configuration_Global.int_height=}")
		print(F"Ratio: {Configuration_Global.int_width/Configuration_Global.int_height}")
		print(F"Ratio Global | Original as contrast: {Ratio_Global=}")
	else:
		Crop_Camera_Start_Input = Location_Point ( Approximate_To_Integer ( Configuration_Global.int_width * Ratio_Crop_Camera_Start_In ) , Approximate_To_Integer ( Configuration_Global.int_height * Ratio_Crop_Camera_Start_In ) )
		Crop_Camera_End_Input = Location_Point ( Approximate_To_Integer ( Configuration_Global.int_width * Ratio_Crop_Camera_End_In ) , Approximate_To_Integer ( Configuration_Global.int_height * Ratio_Crop_Camera_End_In ) )

		# Pre-Processing
		Crop_Camera_Start=Crop_Camera_Start_Input
		# Crop_Camera_Start=Crop_Camera_Start_Input\
		# 	if IsFullImageRequired(Width_Camera_Start_In)\
		# 	else Adjust_Crop(Location_In = Camera_Start_Global,Crop_In = Crop_Camera_Start_Input)
		Crop_Camera_End=Crop_Camera_End_Input
		# Crop_Camera_End=Crop_Camera_End_Input\
		# 	if IsFullImageRequired(Width_Camera_End_In)\
		# 	else Adjust_Crop(Location_In = Camera_Start_Global,Crop_In=Crop_Camera_End_Input)

		with Engine.no_grad () :
			Camera_Start =\
			{
				'fltCenterU' : Camera_Start_Global.Horizon ,
				'fltCenterV' : Camera_Start_Global.Vertical ,
				'intCropWidth' : Crop_Camera_Start.Horizon,
				'intCropHeight' : Crop_Camera_Start.Vertical,
				# 'intCropHeight' : Height_Camera_Start_In,
			}

			Camera_End =\
			{
				'fltCenterU' : Camera_End_Global.Horizon ,
				'fltCenterV' : Camera_End_Global.Vertical ,
				'intCropWidth' : Crop_Camera_End.Horizon,
				'intCropHeight' : Crop_Camera_End.Vertical,
				# 'intCropHeight' : Height_Camera_End_In,
			}

			# Debug
			print(F"Scale Down in {Inspect.currentframe().f_code.co_name} before Process Ken Burns")
			# print(F"{Image_In.shape=}")
			print(F"{Configuration_Global.int_width=},{Configuration_Global.int_height=}")
			print(F"{Camera_Start_Global.Horizon=},{Camera_Start_Global.Vertical=}")
			print(F"{Camera_End_Global.Horizon=},{Camera_End_Global.Vertical=}")
			print(F"{Crop_Camera_Start.Horizon=},{Crop_Camera_Start.Vertical=}")
			print(F"{Crop_Camera_End.Horizon=},{Crop_Camera_End.Vertical=}")
			print(F"Ratio: {Configuration_Global.int_width/Configuration_Global.int_height}")
			print(F"Ratio Global | Original as contrast: {Ratio_Global=}")

			List_Frame_Return,Configuration_Carrier = Pipeline.process_kenburns\
			(
				objSettings =\
				{
					'fltSteps' : Number.linspace\
					(
						start = 0.0 ,
						stop = 1.0 ,
						num = Number_Frame_In,
					)\
						.tolist () ,
					'objFrom' : Camera_Start ,
					'objTo' : Camera_End ,
					'boolInpaint' : IsInpainting_In,
				} ,
				objCommon = Configuration_Global ,
				inpaint = IsInpainting_In ,
				verbose = IsVerbosing_In,
				# Depthor_In = Depthor_In,
				# IsDepthRequired_In = IsDepthRequired_In,
			)

			# Debug
			print(F"Scale Down in {Inspect.currentframe().f_code.co_name} after Process Ken Burns")
			# print(F"{Image_In.shape=}")
			print(F"{Configuration_Global.int_width=},{Configuration_Global.int_height=}")
			print(F"Ratio: {Configuration_Global.int_width/Configuration_Global.int_height}")
			print(F"Ratio Global | Original as contrast: {Ratio_Global=}")

	# for Carrier_Image_PositiveFilm in Configuration_Carrier.stage_inpainted_imgs :
	# 	List_PositiveFilm_Return.append(Carrier_Image_PositiveFilm )
	if(Configuration_Carrier.stage_inpainted_imgs):
		# Debug
		# print(F"{Configuration_Carrier.stage_inpainted_imgs=}")

		List_PositiveFilm_Return.append(Configuration_Carrier.stage_inpainted_imgs[-2] )		# 倒数2nd张
		List_PositiveFilm_Return.append(Configuration_Carrier.stage_inpainted_imgs[-1] )		# 倒数1st张
	else:
		pass

	# for Carrier_Image_Mask in Configuration_Carrier.stage_inpainted_masks :
	# 	List_Mask_Return.append(Carrier_Image_Mask)
	if(Configuration_Carrier.stage_inpainted_masks):
		# Debug
		# print(F"{Configuration_Carrier.stage_inpainted_masks=}")

		List_Mask_Return.append(Configuration_Carrier.stage_inpainted_masks[-2] )		# 倒数2nd张
		List_Mask_Return.append(Configuration_Carrier.stage_inpainted_masks[-1] )		# 倒数1st张

	# Post-Processing
	List_Frame_Global=List_Frame_Return
	Configuration_Global=Configuration_Carrier

	return \
	[
		List_PositiveFilm_Return[0],
		List_PositiveFilm_Return[1],
		List_Mask_Return[0],
		List_Mask_Return[1],
		List_Frame_Return,
		#
		#
		Framework.update(interactive = True),
		# Reset Video Box
		None,
		#
		#
		# Update Progress
		F"Cameras generated successfully.",
	]

def DoOnButton_Encode_Video_Stepping_TabClicking() :

	IsVerbosing_In=IsVerbosing_Global

	# Debug
	print(F"{IsVerbosing_In=} in {Inspect.currentframe().f_code.co_name}()")

	Path_Video_Return=F"Result-Downloading.mp4"

	# ！此处为了抵消AIS代码中Frame List→Video时对单Frame进行的BGR→RGB，进行了1次转换
	List_Frame_BGR=\
	[
		Carrier_Frame[ :, :, : :-1 ]\
			for Carrier_Frame\
			in List_Frame_Global
	]

	Anime3D_KenBurns.npyframes2video\
	(
		List_Frame_BGR,
		Path_Video_Return,
	)

	# ！VideoBox返回时至少2个值，否则始终报错
	return\
	[
		Path_Video_Return,
		Path_Video_Return,
	]

def DoOnImageBox_Input_Original_Stepping_TabChanged\
(
	Image_In:Number.ndarray,
	Ratio_Crop_Camera_Start_In,
	Ratio_Crop_Camera_End_In,
) :
	"""
	Clear all results previous while new processing starting.

	Args:
		Image_In ():

	Returns:
	"""

	global Height_Global
	global Width_Global
	global Ratio_Global
	global Ratio_Resize_Global

	Height_Original=None
	Width_Original=None
	Ratio_Resize:float=1

	# ！需要提取为新的函数、Constant
	List_Return = \
	[
		None ,
		None ,
		None ,
		None ,
		#
		True,
		True,
		True,
		False,
		Mask_α_Default,
		#
		None ,
		None ,
		None ,
		None ,
		#
		False,
		None ,
		None ,
		#
		PlaceHolder_Label_Camera ,
		PlaceHolder_Label_Camera ,
		#
		75,
		True,
		#
		#
		True,
		#
		#
	]

	if(Image_In is None):
		List_Return.append(Ratio_Crop_Camera_Start_Deault)
		List_Return.append ( None )
		List_Return.append ( None )
		#
		List_Return.append(Ratio_Crop_Camera_End_Deault)
		List_Return.append ( None )
		List_Return.append ( None )
		#
		List_Return.append ( Image_In )
		List_Return.append ( Image_In )
		#
		#
		#
		# List_Return.append ( None )
		List_Return.append ( Progress_Default )
		List_Return.append ( Progress_Default )
	else:
		Height_Original,Width_Original=Image_In.shape[:2]

		IsHorizonRectangle=True\
			if Width_Original>Height_Original\
			else False
		Ratio_Global=Width_Original/Height_Original		# Save Ratio of Image Input as Single

		# Processing: 计算Down Scale后的Ratio、Pixel坐标
		if(IsHorizonRectangle):
			Ratio_Resize=Width_Original/Configuration_AIS.max_size      # ！考虑更换获取来源，更统一一些，而不是当前的Legacy的Pipeline的残留来源

			Width_Global=Configuration_AIS.max_size     # 直接获取；或Original通过Ratio计算
			Height_Global=Approximate_To_Integer(Height_Original/Ratio_Resize)      # 直接Original通过Ratio计算；或给定值通过长宽Ratio计算
		else:
			Ratio_Resize=Height_Original/Configuration_AIS.max_size      # ！考虑更换获取来源，更统一一些，而不是当前的Legacy的Pipeline的残留来源

			Width_Global=Approximate_To_Integer(Width_Original/Ratio_Resize)      # 直接Original通过Ratio计算；或给定值通过长宽Ratio计算
			Height_Global=Configuration_AIS.max_size     # 直接获取；或Original通过Ratio计算

		# Debug
		print ( F"After generating Down Scaled Image in {Inspect.currentframe ().f_code.co_name}()" )
		print ( F"{Width_Global=}, {Height_Global=}" )
		print ( F"{Width_Original=}, {Height_Original=}" )
		print ( F"Ratio: {Width_Global / Height_Global}" )
		print ( F"Ratio Global | Original as contrast: {Ratio_Global=}" )

		Width_Preferred=Approximate_To_Integer( Width_Global * Ratio_Crop_Camera_Start_In )
		Height_Preferred=Approximate_To_Integer( Height_Global * Ratio_Crop_Camera_End_In )

		List_Return.append(Ratio_Crop_Camera_Start_Deault)
		List_Return.append(Width_Preferred)
		List_Return.append(Height_Preferred)
		#
		List_Return.append(Ratio_Crop_Camera_End_Deault)
		List_Return.append(Width_Preferred)
		List_Return.append(Height_Preferred)
		#
		List_Return.append ( Image_In )
		List_Return.append ( Image_In )
		#
		#
		#
		List_Return.append ( Progress_Default )
		List_Return.append ( Progress_Default )

	# Post-Processing
	Ratio_Resize_Global=Ratio_Resize

	# List_Return.append(Framework.update(width=Width_Global,height=Height_Global))

	return List_Return

# ！需要拆分Wrapper、合并内容
def DoOnImageBox_Stepping_TabSelecting\
(
	Image_BackUp_In,
	Image_In,
	# Width_Crop_Camera_Start_In,
	# Width_Crop_Camera_End_In,
	#
	Ratio_Crop_Camera_Start_In,
	Ratio_Crop_Camera_End_In,
	#
	#
	Argument_System:Framework.SelectData,
) :

	global IsStartingCameraSetting_Global
	global Camera_Start_Global
	global Camera_End_Global

	List_Return=None
	Image_Return=None

	if(IsStartingCameraSetting_Global):
		# Pre-Processing
		# 清除前次的痕迹
		Image_In=Number.copy(Image_BackUp_In)

		Location_Camera_Start_Original=Location_Point(Argument_System.index [ 0 ],Argument_System.index[1 ] )

		# Debug
		print(F"Before Processing Start Camera Location in {Inspect.currentframe().f_code.co_name}()")
		print(F"{Location_Camera_Start_Original.Horizon=},{Location_Camera_Start_Original.Vertical=}")
		print(F"Ratio: {Location_Camera_Start_Original.Horizon/Location_Camera_Start_Original.Vertical}")
		print(F"Ratio Global | Original as contrast: {Ratio_Global=}")
		#
		print(F"Resized Image Size should be :")
		print(F"Horizon : {Location_Camera_Start_Original.Horizon/Ratio_Resize_Global}, Vertical : {Location_Camera_Start_Original.Vertical/Ratio_Resize_Global}")

		# Width_Crop_Camera_Start_Original=Approximate_To_Integer(Width_Crop_Camera_Start_In*Ratio_Resize_Global)

		Image_Return,Location_Camera_Start= HighLight_Crop_Using_Ratio ( Image_In , Location_Camera_Start_Original , Ratio_Crop_Camera_Start_In, Color_In = (255 , 165,  0) )     # ！需要Constant化：Orange

		List_Return=\
		[
			Image_Return,
			# Width_Crop_Camera_Start_Return,
			# Width_Crop_Camera_End_In,

			# 为了保证数值的精度，Original的坐标使用原始数据
			# ！需要合并、提取函数+Constant
			F"#### Horizon: {Location_Camera_Start.Horizon} (Original: {Location_Camera_Start_Original.Horizon }), Vertical: {Location_Camera_Start.Vertical} (Original: {Location_Camera_Start_Original.Vertical })",
			PlaceHolder_Label_Camera,
		]

		IsStartingCameraSetting_Global=False
		Camera_Start_Global=Location_Camera_Start
	else:
		Location_Camera_End_Original=Location_Point(Argument_System.index [ 0 ],Argument_System.index[1])

		# Debug
		print(F"Before Processing End Camera Location in {Inspect.currentframe().f_code.co_name}()")
		print(F"{Location_Camera_End_Original.Horizon=},{Location_Camera_End_Original.Vertical=}")
		print(F"Ratio: {Location_Camera_End_Original.Horizon/Location_Camera_End_Original.Vertical}")
		print(F"Ratio Global | Original as contrast: {Ratio_Global=}")
		#
		print(F"Resized Image Size should be :")
		print(F"Horizon : {Location_Camera_End_Original.Horizon/Ratio_Resize_Global}, Vertical : {Location_Camera_End_Original.Vertical/Ratio_Resize_Global}")
		# Width_Crop_Camera_End_Original=Approximate_To_Integer(Width_Crop_Camera_End_In*Ratio_Resize_Global)

		Image_Return, Location_Camera_End= HighLight_Crop_Using_Ratio ( Image_In , Location_Camera_End_Original , Ratio_Crop_Camera_End_In, Color_In = (255 , 0 , 255) )     # ！需要Constant化：Magenta

		List_Return=\
		[
			Image_Return ,
			# Width_Crop_Camera_Start_In,
			# Width_Crop_Camera_End_Return,

			F"#### Horizon: {Camera_Start_Global.Horizon} (Original: {Approximate_To_Integer(Camera_Start_Global.Horizon*Ratio_Resize_Global)}), Vertical: {Camera_Start_Global.Vertical} (Original: {Approximate_To_Integer(Camera_Start_Global.Vertical*Ratio_Resize_Global)})",
			# 为了保证数值的精度，Original的坐标使用原始数据
			F"#### Horizon: {Location_Camera_End.Horizon} (Original: {Approximate_To_Integer(Location_Camera_End_Original.Horizon)}), Vertical: {Location_Camera_End.Vertical} (Original: {Approximate_To_Integer(Location_Camera_End_Original.Vertical)})",
		]

		IsStartingCameraSetting_Global=True
		Camera_End_Global=Location_Camera_End

	return List_Return

# ！考虑合并Event Handler到1个
def DoOnSlider_Camera_Start_Width_Stepping_TabChanged (Width_In) :

	return Adjust_Ratio(Width_In = Width_In)

def DoOnSlider_Camera_Start_Height_Stepping_TabChanged (Height_In) :

	return Adjust_Ratio(Height_In=Height_In)

def DoOnSlider_Camera_End_Width_Stepping_TabChanged (Width_In) :

	return Adjust_Ratio(Width_In=Width_In)

def DoOnSlider_Camera_End_Height_Stepping_TabChanged (Height_In) :

	return Adjust_Ratio(Height_In=Height_In)

#region Core

def Adjust_Ratio\
(
	Width_In=None,
	Height_In=None,
) :

	Value_Return=None

	if(Width_In is not None):
		Value_Return=Approximate_To_Integer(Width_In/Ratio_Global)
	else:
		Value_Return=Approximate_To_Integer(Height_In*Ratio_Global)

	return Value_Return

#endregion Core

#region Utility

def Approximate_To_Integer ( Input_In ) :

	# return Mathemastics.floor(Input_In)       # 比较合理的估算方式：保障坐标不会超出范围
	# return int(Input_In)       # 是截尾法，是比较合理的估算方式：保障坐标不会超出范围，同时对负数值的处理符合预期（趋向〇，而非-∞）
	return int(round(Input_In))     # AIS源码|调用的Python源码中主要的坐标的处理逻辑

def HighLight_Crop_Using_Ratio\
(
	Image_In,
	Location_Camera_Original_In:Location_Point,
	Ratio_Crop_In,
	Color_In,
):

	Height_Original,Width_Original=Image_In.shape[:2]
	# Crop_Original_Input = Location_Point ( Width_Original_In , Approximate_To_Integer ( Width_Original_In / Ratio_Global ) )
	Crop_Original_Input = Location_Point ( Approximate_To_Integer(Width_Original*Ratio_Crop_In) , Approximate_To_Integer ( Height_Original*Ratio_Crop_In ) )
	Size_Original=Location_Point ( Width_Original , Height_Original )

	return HighLight_Crop_Core(Image_In,Location_Camera_Original_In,Crop_Original_Input,Color_In,Size_Original)

def HighLight_Crop_Using_Crop\
(
	Image_In,
	Location_Camera_Original_In:Location_Point,
	Crop_Original_In:Location_Point,
	Color_In,
):

	Height_Original,Width_Original=Image_In.shape[:2]
	Size_Original=Location_Point ( Width_Original , Height_Original )

	return HighLight_Crop_Core(Image_In,Location_Camera_Original_In,Crop_Original_In,Color_In,Size_Original)

def HighLight_Crop_Core\
(
	Image_In,
	Location_Camera_Original_In:Location_Point,
	Crop_Original_In:Location_Point,
	Color_In,
	Size_Original_In:Location_Point,
):

	# Debug
	print(F"{Inspect.currentframe().f_code.co_name}()")
	print(F"{Color_In=}")

	Image_Return=Number.copy(Image_In)
	Location_Camera_Return:Location_Point=None
	Width_Return=None
	Location_Camera_Original:Location_Point=None
	Crop_Original=None

	# Pre-Processing
	# Crop_Camera = Crop_Camera \
	# 	if IsFullImageRequired ( Width_In,Ratio_Resize_Global ) \
	# 	else Adjust_Crop ( Location_In = Location_Camera_In , Crop_In = Crop_Camera ,Ratio_Resize_In = Ratio_Resize_Global)
	# Location_Camera_Original,Crop_Original = Adjust_Crop ( Location_Camera_In = Location_Camera_Original_In , Crop_In = Crop_Original_Input , Ratio_Resize_Image_In = Ratio_Resize_Global , Limit_Ratio_Resize_Crop_In = Ratio_Crop_In )
	Crop_Original=Crop_Original_In
	Location_Camera_Original=Centralise_Camera(Location_Camera_Original_In,Crop_Original_In,Size_Original_In)

	# Processing: Add Rectangle to mark Selection
	# ？是否需要进行Open CV的颜色顺序的转换
	Location_Start_Original,Location_End_Original=ConvertLocation_From_Center_To_Corner(Location_Camera_Original,Crop_Original )
	Vision.rectangle\
	(
		img = Image_Return ,
		pt1 = Location_Start_Original ,       # Start Location
		pt2 = Location_End_Original ,     # End Location
		color = Color_In ,
		# thickness = 2,     # ！需要Constant化
		# thickness = 5,     # ！需要Constant化
		thickness = 10,     # ！需要Constant化
	)

	# ？Python语法是否对Structure的算术运算有简化
	# ！需要提取函数
	Location_Camera_Return=Location_Point ( Approximate_To_Integer ( Location_Camera_Original.Horizon / Ratio_Resize_Global ), Approximate_To_Integer(Location_Camera_Original.Vertical / Ratio_Resize_Global ) )
	Width_Return=Approximate_To_Integer( Crop_Original.Horizon / Ratio_Resize_Global )

	return\
	[
		Image_Return,
		Location_Camera_Return,
		# Width_Return,
	]

def ConvertLocation_From_Center_To_Corner\
(
	Camera_In:Location_Point,
	Crop_In:Location_Point,
) :

	# 计算起止点坐标
	Location_Start_Return = Location_Point\
	(
		Camera_In.Horizon - Approximate_To_Integer ( Crop_In.Horizon / 2 ) ,
		Camera_In.Vertical - Approximate_To_Integer ( Crop_In.Vertical / 2 ) ,
	)
	Location_End_Return = Location_Point\
	(
		Camera_In.Horizon + Approximate_To_Integer ( Crop_In.Horizon / 2 ) ,
		Camera_In.Vertical + Approximate_To_Integer ( Crop_In.Vertical / 2 ) ,
	)

	# 返回结果
	return\
	[
		Location_Start_Return,
		Location_End_Return,
	]

def Centralise_Camera(Location_Camera_In:Location_Point,Crop_In:Location_Point,Size_In:Location_Point):

	Location_Camera_Return:Location_Point=None
	Location_Start:Location_Point
	Location_End:Location_Point

	Location_Start,Location_End=ConvertLocation_From_Center_To_Corner(Location_Camera_In,Crop_In)

	ΔHorizon=0
	ΔVertical=0

	if Location_Start.Horizon<0:
		ΔHorizon=-Location_Start.Horizon
	elif Location_End.Horizon>Size_In.Horizon:
		ΔHorizon= Size_In.Horizon - Location_End.Horizon
	else:
		pass

	if Location_Start.Vertical<0:
		ΔVertical=-Location_Start.Vertical
	elif Location_End.Vertical>Size_In.Vertical:
		ΔVertical= Size_In.Vertical - Location_End.Vertical
	else:
		pass

	Location_Camera_Return=Location_Point(Location_Camera_In.Horizon+ΔHorizon,Location_Camera_In.Vertical+ΔVertical)

	return Location_Camera_Return

#endregion Utility

#endregion Event Handler

#region Customised

#region Anime Instance Segmentation


def HighLight_Crop_AutoZoom (Image_In,Configuration_In) :

	# with Engine.no_grad():
	Image_Return=None
	# Image_Return=Image_In

	if \
		(Image_In is not None)\
		and (Configuration_In is not None):
		Height_Original , Width_Original = Image_In.shape [ :2 ]
		Location_Camera_Start_Original = Location_Point ( Approximate_To_Integer ( Width_Original / 2 ) , Approximate_To_Integer ( Height_Original / 2 ) )
		Camera_Start = \
			{
				'fltCenterU' : Location_Camera_Start_Original.Horizon ,
				'fltCenterV' : Location_Camera_Start_Original.Vertical ,
				'intCropWidth' : Approximate_To_Integer ( Ratio_Crop_AutoZoomed * Configuration_In.int_width ) ,
				'intCropHeight' : Approximate_To_Integer ( Ratio_Crop_AutoZoomed * Configuration_In.int_height ) ,
			}
		# Camera_End=Anime3D_KenBurns.process_autozoom\
		# (
		# 	{
		# 		'fltShift' : 100.0 ,
		# 		'fltZoom' : 1.25 ,
		# 		'objFrom' : Camera_Start,
		# 	} ,
		# 	Configuration_In,
		# )
		# Location_Camera_End_Original=Location_Point(Camera_End[F"fltCenterU"],Camera_End[F"fltCenterV"])
		Location_Camera_End_Original = Location_Point ( Location_Camera_Start_Original.Horizon + 20 , Location_Camera_Start_Original.Vertical + 20 )
		# Crop_Camera_End_Original=Location_Point(Camera_End[F"intCropWidth"],Camera_End[F"intCropHeight"])

		# Start Camera
		Image_Processing , _ = HighLight_Crop_Using_Ratio ( Image_In , Location_Camera_Start_Original , Ratio_Crop_AutoZoomed , Color_In = (255 , 165 , 0) )  # ！需要Constant化：Orange
		# End Camera
		# Image_Return , _  = HighLight_Crop_Using_Crop ( Image_Processing , Location_Camera_End_Original , Crop_Original_In = Crop_Camera_End_Original , Color_In = (255 , 0 , 255) )  # ！需要Constant化：Magenta
		Image_Return , _ = HighLight_Crop_Using_Ratio ( Image_Processing , Location_Camera_End_Original , Ratio_Crop_In = Ratio_Crop_AutoZoomed * 0.8 , Color_In = (255 , 0 , 255) )  # ！需要Constant化：Magenta
	elif Configuration_In is None:
		Image_Return=Image_In
	else:
		pass

	return Image_Return


def DoOnCheckBox_Auto_Zoom_Stepping_TabChanged (IsAutoZoomed_In,Image_In) :

	Image_Return=None

	if(IsAutoZoomed_In):
		Image_Return=HighLight_Crop_AutoZoom(Image_In,Configuration_Global)
	else:
		Image_Return=Image_In

	return Image_Return

def Create_Stepping_Demonstration_Tab () :

	Title=F"3D Ken Burns"

	with Framework.Tab ( Title ) as Page:

		# Pre-Processing
		Page_Global = Page
		# Framework.Markdown(Title)
		# Framework.Markdown()
		# Framework.Markdown()
		Framework.Markdown ( F"# Step 0. Choose Image")

		with Framework.Row():
			with Framework.Column():
				Title_Image_Input_Original=F"Input Image"
				Framework.Markdown ( F"#### {Title_Image_Input_Original}" )
				# ！需要需要禁用更改Image功能，仅作为大屏展示|操作
				ImageBox_Input_Original_Stepping_Tab = Framework.Image \
				(
					label = Title_Image_Input_Original ,
					show_label = False,
					#
					# type=F"pil",
					type=F"numpy",
					#
					interactive = True,
					# editable = False,
					# height = 600,
				)

		# with Framework.Row\
		# (
		# 	# wrap=False,
		# 	# min_width = 180,
		# ):
			# with Framework.Column\
			# (
			# 	min_width = 160,
			# ):
				# CheckBox_Convert_To_BGR_Stepping_Tab = Create_Step_CheckBox\
				# (
				# 	Name_In = F"RGB → BGR",
				# 	Value_Default_In = True,
				# )

			# with Framework.Column\
			# (
			# 	min_width = 180,
			# ):
			# 	ImageBox_After_1st_Process_Stepping_Tab=Framework.Image\
			# 	(
			# 		label = F"Image-After 1st Process",
			# 		# type=F"pil",
			# 		type=F"numpy",
			# 	)
			# 	Button_Submit_After_1st_Process_Stepping_Tab=Framework.Button\
			# 	(
			# 		F"Submit",
			# 	)
			# 	CheckBox_After_1st_Process_Stepping_Tab=Create_Step_CheckBox\
			# 	(
			# 		Name_In = F"1st",
			# 		Value_Default_In = True,
			# 	)

		# with Framework.Row():
			with Framework.Column():
				List_Example_Stepping_Tab=Framework.Examples\
				(
					examples=\
					[
						OS.path.join ( Path_List_Sample , Carrier_Path_Image ) for Carrier_Path_Image in OS.listdir ( Path_List_Sample )
					],
					inputs = \
					[
						ImageBox_Input_Original_Stepping_Tab,
					],
					examples_per_page = 40,
				)

		Framework.Markdown ( F"---" )

		Framework.Markdown ( F"# Step 1. Generate Instance Masks")

		with Framework.Row():
			with Framework.Column \
			(
				min_width = 400 ,
			) :
				Title_After_Drawing_Instances=F"Instances"
				Framework.Markdown(F"#### {Title_After_Drawing_Instances}")
				ImageBox_After_2nd_Process_Stepping_Tab = Framework.Image \
				(
					label = Title_After_Drawing_Instances ,
					show_label = False,
					# type=F"pil",
					type = F"numpy" ,
					interactive = False,
					# editable = False,
					height = 600,
				)
				# Button_Submit_After_2nd_Process_Stepping_Tab=Framework.Button\
				# (
				# 	F"Submit",
				# )
				# CheckBox_After_2nd_Process_Stepping_Tab = Create_Step_CheckBox\
				# (
				# 	Name_In = F"2nd",
				# 	Value_Default_In = True,
				# )

			with Framework.Column \
			(
				min_width = 180 ,
			) :
				Framework.Markdown(F"#### Progress")
				Label_Progress_Instance_Generating=Framework.Label\
				(
					value =Progress_Default,
					# label = F"Progress",
					label = F"",
					# show_label = False,
				)

				Framework.Markdown(F"---")

				Framework.Markdown(F"#### Options")
				CheckBox_BoundingBox_Stepping_Tab = Create_Step_CheckBox\
				(
					Name_In = F"Bounding Box",
					Value_Default_In = True,
				)
				CheckBox_InstanceMask_Stepping_Tab = Create_Step_CheckBox\
				(
					Name_In = F"Instance Mask",
					Value_Default_In = True,
				)
				CheckBox_InstanceContour_Stepping_Tab = Create_Step_CheckBox\
				(
					Name_In = F"Instance Contour",
					Value_Default_In = True,
					IsVisible_In = IsAllActivatedControlDisplayed,
				)
				# 生成Text Tag
				CheckBox_Tag_List_Stepping_Tab = Create_Step_CheckBox\
				(
					Name_In = F"Tag",
					Value_Default_In = False,
					IsVisible_In = IsAllActivatedControlDisplayed,
				)
				Slider_Mask_α_Stepping_Tab=Framework.Slider\
				(
					value = Mask_α_Default,		# 相当于default
					minimum = 0,
					maximum = 1,
					step = 0.01,
					# multiselect = True,
					label = F"Transparency of Mask",
					# info = F"The higher the value set, the brighter the brightness is.",
				)

		# with Framework.Row():
			# with Framework.Column():
				CheckBox_IsVerbosing_Stepping_Tab = Create_Step_CheckBox\
				(
					Name_In = F"Verbose",
					Value_Default_In = False,
					Tip_In = F"If checked, Programme will generate intermediate | temporary Image Results in its Root Directory.",
					IsVisible_In = IsAllActivatedControlDisplayed,
				)
				Button_Run_With_Steps_Stepping_Tab=Framework.Button\
				(
					F"Run",
				)

		with Framework.Row():
			# ！需要命名为F"Intermediate Masks"
			# with Framework.Group():
			with Framework.Column \
			(
				min_width = 160 ,
			) :
				Title_After_Generating_Coarse_Depth=F"Coarse Depth"
				Framework.Markdown(F"#### {Title_After_Generating_Coarse_Depth}")
				ImageBox_After_3rd_Process_Stepping_Tab = Framework.Image \
				(
					label = Title_After_Generating_Coarse_Depth ,
					show_label = False,
					# type=F"pil",
					type = F"numpy" ,
					interactive = False,
					# editable = False,
				)
			# Button_Submit_After_3rd_Process_Stepping_Tab=Framework.Button\
			# (
			# 	F"Submit",
			# )
			# CheckBox_After_3rd_Process_Stepping_Tab = Create_Step_CheckBox\
			# (
			# 	Name_In = F"3rd",
			# 	Value_Default_In = True,
			# )

			with Framework.Column \
			(
				min_width = 160 ,
			) :
				Title_After_Generating_Adjusted_Depth=F"Adjusted Depth"
				Framework.Markdown(F"#### {Title_After_Generating_Adjusted_Depth}")
				ImageBox_After_4th_Process_Stepping_Tab = Framework.Image \
				(
					label = Title_After_Generating_Adjusted_Depth ,
					show_label = False,
					# type=F"pil",
					type = F"numpy" ,
					interactive = False,
					# editable = False,
				)
			# Button_Submit_After_4th_Process_Stepping_Tab=Framework.Button\
			# (
			# 	F"Submit",
			# )
			# CheckBox_After_4th_Process_Stepping_Tab = Create_Step_CheckBox\
			# (
			# 	Name_In = F"4th",
			# 	Value_Default_In = True,
			# )

			with Framework.Column \
			(
				min_width = 160 ,
			) :
				Title_After_Generating_Final_Depth=F"Final Depth"
				Framework.Markdown(F"#### {Title_After_Generating_Final_Depth}")
				ImageBox_After_5th_Process_Stepping_Tab = Framework.Image \
				(
					label = Title_After_Generating_Final_Depth ,
					show_label = False,
					# type=F"pil",
					type = F"numpy" ,
					interactive = False,
					# editable = False,
				)
				# Button_Submit_After_5th_Process_Stepping_Tab=Framework.Button\
				# (
				# 	F"Submit",
				# )
				# CheckBox_After_5th_Process_Stepping_Tab = Create_Step_CheckBox\
				# (
				# 	Name_In = F"5th",
				# 	Value_Default_In = True,
				# )



		Framework.Markdown ( F"---" )

		# Frame List Generating Part
		Framework.Markdown ( F"# Step 2. Set Camera View at Start and End" )

		Framework.Markdown(F"## The <span style='color:rgb(255,165,0)'>ORANGE</span> box shows camera view at start. The <span style='color:rgb(255,0,255)'>MAGENTA</span> box shows camera view at end.")
		# Framework.Markdown(F"## Crop of Start Camera in <span style='color:rgb(255,165,0)'>Orange</span>, Crop of End Camera in <span style='color:rgb(255,0,255)'>Magenta</span>.")

		with Framework.Row():
			with Framework.Column () :
				Title_ImageBox_Output_Original=F"Input Image with Camera Views at Start and End"
				Framework.Markdown ( F"#### {Title_ImageBox_Output_Original}" )
				# ！需要需要禁用更改Image功能，仅作为大屏展示|操作
				ImageBox_Output_Original_Stepping_Tab=Framework.Image\
				(
					label = Title_ImageBox_Output_Original,
					show_label = False,
					# type=F"pil",
					type=F"numpy",
					#
					# interactive = False,
					# Due to the bug under solving, we have to work in interactive mode: https://github.com/gradio-app/gradio/issues/5945
					interactive = True,
					#
					# editable = False,
					# height = 600,
				)

		# with Framework.Row():
			with Framework.Column () :
				Framework.Markdown(F"#### Progress:")
				Label_Progress_Camera_Generating=Framework.Label\
				(
					value =Progress_Default,
					# label = F"Progress:",
					label = F"",
					# show_label = False,
				)

				Framework.Markdown(F"---")

				Framework.Markdown ( F"### Method 1: Auto Zoom (Pre-set Camera Views)" )
				CheckBox_Auto_Zoom_Stepping_Tab = Create_Step_CheckBox\
				(
					Name_In = F"Auto Zoom",
					Value_Default_In = True,
				)

				Framework.Markdown ( F"---" )

				Framework.Markdown ( F"### Method 2: Set Camera Views Manually" )
				# Label_Camera_Start_Stepping_Tab=Framework.Label\
				# (
				# 	value =PlaceHolder_Label_Camera,
				# 	label = F"Start Camera Location",
				# 	# elem_classes = F"Label-Display-Position",
				# )
				Label_Camera_Start_Stepping_Tab=Framework.Markdown\
				(
					PlaceHolder_Label_Camera,
					visible = IsAllActivatedControlDisplayed,
				)
				Title_Ratio_Crop_Start=F"Camera View at Start"
				Framework.Markdown ( F"{Title_Ratio_Crop_Start}" )
				Slider_Ratio_Crop_Camera_Start = Framework.Slider \
				(
					value = Ratio_Crop_Camera_Start_Deault ,  # 相当于default
					minimum = Ratio_Crop_Minimum ,      # ！考虑改为Constant
					maximum = 1 ,      # ！考虑改为Constant
					step = 0.01 ,
					# multiselect = True,
					# label = Title_Ratio_Crop_Start ,
					label = F"Width of Camera View at Start: Width of Input Image" ,
					# show_label = False,
					show_label = True,
				)

				Framework.Markdown\
				(
					F"---",
					visible = IsAllActivatedControlDisplayed,
				)

		# with Framework.Row():
		# 	with Framework.Column () :
				Slider_Camera_Start_Width = Framework.Slider \
				(
					value = Width_Global ,  # 相当于default
					minimum = Limit_Minimum ,      # ！考虑改为Constant
					maximum = Limit_Maxmum ,      # ！考虑改为Constant
					step = 1 ,
					# multiselect = True,
					label = F"Start Camera-Width (After Down Scaling)" ,
					visible = IsAllActivatedControlDisplayed,
				)
				Slider_Camera_Start_Height = Framework.Slider \
				(
					value = Height_Global ,  # 相当于default
					minimum = Limit_Minimum ,      # ！考虑改为Constant
					maximum = Limit_Maxmum ,      # ！考虑改为Constant
					step = 1 ,
					# multiselect = True,
					label = F"Start Camera-Height (After Down Scaling)" ,
					visible = False,
				)

			# with Framework.Column () :
				# Label_Camera_End_Stepping_Tab=Framework.Label\
				# (
				# 	value = PlaceHolder_Label_Camera,
				# 	label = F"End Camera Location",
				# 	# elem_classes = F"Label-Display-Position",
				# )
				Label_Camera_End_Stepping_Tab=Framework.Markdown\
				(
					PlaceHolder_Label_Camera,
					visible = IsAllActivatedControlDisplayed,
				)
				Title_Ratio_Crop_End=F"Camera View at End"
				Framework.Markdown ( F"{Title_Ratio_Crop_End}" )
				Slider_Ratio_Crop_Camera_End = Framework.Slider \
				(
					value = Ratio_Crop_Camera_End_Deault ,  # 相当于default
					minimum = Ratio_Crop_Minimum ,      # ！考虑改为Constant
					maximum = 1 ,      # ！考虑改为Constant
					step = 0.01 ,
					# multiselect = True,
					# label = Title_Ratio_Crop_End ,
					label = F"Width of Camera View at End: Width of Input Image" ,
					# show_label = False,
					show_label = True,
				)
				Slider_Camera_End_Width = Framework.Slider \
				(
					value = Width_Global ,  # 相当于default
					minimum = Limit_Minimum ,
					maximum = Limit_Maxmum ,
					step = 1 ,
					# multiselect = True,
					label = F"End Camera-Width (After Down Scaling)" ,
					visible = IsAllActivatedControlDisplayed,
				)
				Slider_Camera_End_Height = Framework.Slider \
				(
					value = Height_Global ,  # 相当于default
					minimum = Limit_Minimum ,
					maximum = Limit_Maxmum ,
					step = 1 ,
					# multiselect = True,
					label = F"End Camera-Height (After Down Scaling)" ,
					visible = False,
				)

				Framework.Markdown ( F"---" )

				Button_Generate_Camera_View_Stepping_Tab=Framework.Button\
				(
					F"Generate Camera View",
					interactive = False,
				)

		# Framework.Markdown(F"#### The closer the Cameras are, the steadier the effect is.")

		with Framework.Row\
		(
			visible = IsAllActivatedControlDisplayed,
		):
			Slider_Number_Frame_Stepping_Tab = Framework.Slider \
			(
				value = 75 ,  # 相当于default
				minimum = 0 ,
				maximum = 100 ,	 # ！Pending
				step = 1 ,
				# multiselect = True,
				label = F"Frame Number" ,
			)
			CheckBox_IsInpainting_Stepping_Tab = Create_Step_CheckBox\
			(
				Name_In = F"Inpainting",
				Value_Default_In = True,
			)
			Slider_Depthor_Stepping_Tab = Framework.Slider \
			(
				value = 3 ,  # 相当于default
				minimum = 0 ,
				maximum = 10 ,	 # ！Pending
				step = 1 ,
				# multiselect = True,
				label = F"Depth Factor" ,
				visible = False,    # 目前认为不需要进行手动设置
			)



		Framework.Markdown(F"#### Inpainted Backgrounds")

		with Framework.Row():
			with Framework.Column():
				Title_Camera_PositiveFilm_Start=F"Start"
				Framework.Markdown(F"{Title_Camera_PositiveFilm_Start}")
				ImageBox_Output_PositiveFilm_Start_Stepping_Tab=Framework.Image\
				(
					label = Title_Camera_PositiveFilm_Start ,
					show_label = False,
					# type=F"pil",
					type=F"numpy",
					interactive = False,
					# editable = False,
					height = 600 ,
				)

			with Framework.Column () :
				Title_Camera_PositiveFilm_End = F"End"
				Framework.Markdown ( F"{Title_Camera_PositiveFilm_End}" )
				ImageBox_Output_PositiveFilm_End_Stepping_Tab=Framework.Image\
				(
					label = Title_Camera_PositiveFilm_End ,
					show_label = False,
					# type=F"pil",
					type=F"numpy",
					interactive = False,
					# editable = False,
					height = 600 ,
				)

		Framework.Markdown(F"#### Masks for Background Inpainting")

		with Framework.Row():
			with Framework.Column () :
				Title_Camera_Mask_Start = F"Start"
				Framework.Markdown ( F"{Title_Camera_Mask_Start}" )
				ImageBox_Output_Mask_Start_Stepping_Tab=Framework.Image\
				(
					label = Title_Camera_Mask_Start ,
					show_label = False,
					# type=F"pil",
					type=F"numpy",
					interactive = False,
					# editable = False,
					height = 600 ,
				)

			with Framework.Column () :
				Title_Camera_Mask_End = F"End"
				Framework.Markdown ( F"{Title_Camera_Mask_End}" )
				ImageBox_Output_Mask_End_Stepping_Tab=Framework.Image\
				(
					label = Title_Camera_Mask_End ,
					show_label = False,
					# type=F"pil",
					type=F"numpy",
					interactive = False,
					# editable = False,
					height = 600 ,
				)

		Framework.Markdown ( F"---" )

		# Video Encoding Part
		with Framework.Row():
			with Framework.Column\
			(
				min_width = 200,
			):
				Framework.Markdown ( F"# Step 3. 3D Ken Burns Result" )

				CheckBox_Depth_Field_Stepping_Tab = Create_Step_CheckBox\
				(
					Name_In = F"Depth Field",
					Value_Default_In = False,
					IsVisible_In = False,
				)

			with Framework.Column\
			(
				min_width = 200,
			):
				Button_Encode_Video_Stepping_Tab=Framework.Button\
				(
					F"Encode Video",
					 interactive = False,
					# disable = True,
				)
			# Video Box已自带，不再需要，但是Video无法独立使用
			with Framework.Column\
			(
				min_width = 200,
				visible = IsAllActivatedControlDisplayed,
			):
				File_Output_Downloaing_Stepping_Tab=Framework.File \
				(
					# size=(300,100),
				)

		with Framework.Row():
			# ！需要缩小占用，最好以Button+浮层形式展示；或最佳的是类似Clip工具的横向Frame轴形式拉动（横向Scroll，纵向以单张Frame Height自适应）
			with Framework.Column \
			(
				min_width = 200 ,
				visible = IsAllActivatedControlDisplayed,
			) :
				Framework.Markdown\
				(
					F"#### After 1 Image clicked, use Left (**←**) | Right (**→**) to experience the animation effect.",
					# visible = IsAllActivatedControlDisplayed,
				)
				BatchedImageBox_Output_ListFrame_Stepping_Tab = Create_Output_BatchedImageBox \
				(
					Name_In = F"Frame List" ,
					Type_In = F"numpy" ,
					# IsPreviewMode_In = True,	  # ！慎用，会导致UI Thread死循环，UI直接卡死
					Height_In = 400 ,
				)

			with Framework.Column(min_width = 300,visible = IsAllActivatedControlDisplayed):
				Title_Image_Input_Original_Contrast=F"Original Input Image as Contrast"
				Framework.Markdown ( F"#### {Title_Image_Input_Original_Contrast}" )
				# ！需要需要禁用更改Image功能，仅作为大屏展示|操作
				ImageBox_Output_Original_Contrast = Framework.Image \
				(
					label = Title_Image_Input_Original_Contrast ,
					show_label = False,
					# type=F"pil",
					type=F"numpy",
					interactive = False,
					# editable = False,
					height = 600 ,
				)

			with Framework.Column\
			(
				min_width = 100,
				# visible = False,
		    ):
				Framework.Checkbox(visible = False)

			with Framework.Column\
			(
				# width=1000,
			) :
				Framework.Markdown(F"### A **Five** Seconds Video in variant Frame Ratio.")
				# with Framework.Row():
				# Choose 1 Video related Control, then combined with File Control in same Column
				VideoBox_Output_Stepping_Tab = Framework.PlayableVideo \
				(
					show_label = False,
					# style={ "height" : "600px" },
					# height=600,
					# width=800,
					# width=F"60%",
					height = 600,
				)

			with Framework.Column\
			(
				min_width = 100,
				# visible = False,
			):
				Framework.Checkbox(visible = False)

		# !Add basic authority | permission, authors, references



		# Bind Event handler
		# Button_Submit_After_1st_Process_Stepping_Tab.click\
		# (
		# 	fn = DoOnButtonSubmit_AIS_Instance_Segmentation_Clicking,
		# 	inputs =\
		# 	[
		# 		ImageBox_Input_Original_Stepping_Tab,
		# 	],
		# 	outputs =\
		# 	[
		# 		ImageBox_After_1st_Process_Stepping_Tab,
		# 	],
		# )
		# Button_Submit_After_2nd_Process_Stepping_Tab.click\
		# (
		# 	fn = DoOnButtonSubmit_AIS_Infer_Disparity_Depth_Estimation_Clicking,
		# 	inputs =\
		# 	[
		# 		# ImageBox_After_1st_Process_Stepping_Tab,
		# 	],
		# 	outputs =\
		# 	[
		# 		ImageBox_After_2nd_Process_Stepping_Tab,
		# 	],
		# )
		# Button_Submit_After_3rd_Process_Stepping_Tab.click\
		# (
		# 	fn = DoOnButtonSubmit_AIS_Infer_Disparity_Depth_Coarse_Clicking,
		# 	inputs =\
		# 	[
		# 		# ImageBox_After_2nd_Process_Stepping_Tab,
		# 	],
		# 	outputs =\
		# 	[
		# 		ImageBox_After_3rd_Process_Stepping_Tab,
		# 	],
		# )
		# Button_Submit_After_4th_Process_Stepping_Tab.click\
		# (
		# 	fn = DoOnButtonSubmit_AIS_Infer_Disparity_Depth_Adjusted_Depth_Clicking,
		# 	inputs =\
		# 	[
		# 		# ImageBox_After_3rd_Process_Stepping_Tab,
		# 	],
		# 	outputs =\
		# 	[
		# 		ImageBox_Disparity_After_4th_Process_Stepping_Tab,
		# 		ImageBox_After_4th_Process_Stepping_Tab,
		# 	],
		# )
		# Button_Submit_After_5th_Process_Stepping_Tab.click\
		# (
		# 	fn = DoOnButtonSubmit_AIS_Adjustment_Anime_Segmentation_TabClicking,
		# 	inputs =\
		# 	[
		# 		# ImageBox_After_4th_Process_Stepping_Tab,
		# 	],
		# 	outputs =\
		# 	[
		# 		ImageBox_Disparity_After_5th_Process_Stepping_Tab,
		# 		ImageBox_After_5th_Process_Stepping_Tab,
		# 	],
		# )
		# ImageBox_Input_Original_Stepping_Tab.change\
		# (
		# 	# fn=DoOnImageBox_Input_Original_Stepping_TabChanged,
		# 	fn=lambda Image_In_Out:[Image_In_Out,Image_In_Out,Image_In_Out,Image_In_Out,],
		# 	inputs =\
		# 	[
		# 		ImageBox_Input_Original_Stepping_Tab,
		# 	],
		# 	outputs =\
		# 	[
		# 		# ImageBox_After_1st_Process_Stepping_Tab,
		# 		ImageBox_After_2nd_Process_Stepping_Tab,
		# 		ImageBox_After_3rd_Process_Stepping_Tab,
		# 		ImageBox_After_4th_Process_Stepping_Tab,
		# 		ImageBox_After_5th_Process_Stepping_Tab,
		# 	],
		# )
		#

		# ImageBox_Input_Original_Stepping_Tab.select\
		# (
		# 	fn=DoOnImageBox_Stepping_TabSelecting,
		# 	inputs = \
		# 	[
		# 		Slider_Camera_Start_Width,
		# 		Slider_Camera_End_Width,
		# 	] ,
		# 	outputs = \
		# 	[
		# 		Label_Camera_Start_Stepping_Tab,
		# 		Label_Camera_End_Stepping_Tab,
		# 	] ,
		# )
		ImageBox_Input_Original_Stepping_Tab.change\
		(
			fn=DoOnImageBox_Input_Original_Stepping_TabChanged,
			inputs = \
			[
				ImageBox_Input_Original_Stepping_Tab,
				Slider_Ratio_Crop_Camera_Start,
				Slider_Ratio_Crop_Camera_End,
			] ,
			outputs = \
			[
				ImageBox_After_2nd_Process_Stepping_Tab,
				ImageBox_After_3rd_Process_Stepping_Tab,
				ImageBox_After_4th_Process_Stepping_Tab,
				ImageBox_After_5th_Process_Stepping_Tab,
				#
				CheckBox_BoundingBox_Stepping_Tab,
				CheckBox_InstanceMask_Stepping_Tab,
				CheckBox_InstanceContour_Stepping_Tab,
				CheckBox_Tag_List_Stepping_Tab,
				Slider_Mask_α_Stepping_Tab,
				#
				ImageBox_Output_PositiveFilm_Start_Stepping_Tab,
				ImageBox_Output_PositiveFilm_End_Stepping_Tab,
				ImageBox_Output_Mask_Start_Stepping_Tab,
				ImageBox_Output_Mask_End_Stepping_Tab,
				#
				CheckBox_Depth_Field_Stepping_Tab,
				BatchedImageBox_Output_ListFrame_Stepping_Tab,
				VideoBox_Output_Stepping_Tab,
				#
				Label_Camera_Start_Stepping_Tab,
				Label_Camera_End_Stepping_Tab,
				#
				Slider_Number_Frame_Stepping_Tab,
				CheckBox_IsInpainting_Stepping_Tab,
				#
				#
				CheckBox_Auto_Zoom_Stepping_Tab,
				#
				#
				Slider_Ratio_Crop_Camera_Start,
				Slider_Camera_Start_Width,
				Slider_Camera_Start_Height,
				#
				Slider_Ratio_Crop_Camera_End,
				Slider_Camera_End_Width,
				Slider_Camera_End_Height,
				#
				ImageBox_Output_Original_Stepping_Tab,
				ImageBox_Output_Original_Contrast,
				#
				#
				#
				# VideoBox_Output_Stepping_Tab,
				Label_Progress_Instance_Generating,
				Label_Progress_Camera_Generating,
			] ,
		)
		# ImageBox_Output_Original_Stepping_Tab.change\
		# (
		# 	fn=DoOnImageBox_Output_Original_Stepping_TabChanged,
		# 	inputs = \
		# 	[
		# 		ImageBox_Output_Original_Stepping_Tab,
		# 	] ,
		# 	outputs = \
		# 	[
		# 		ImageBox_Output_Original_Stepping_Tab,
		# 	] ,
		# )
		ImageBox_Output_Original_Stepping_Tab.select\
		(
			fn=DoOnImageBox_Stepping_TabSelecting,
			inputs = \
			[
				ImageBox_Input_Original_Stepping_Tab,
				ImageBox_Output_Original_Stepping_Tab,
				# Slider_Camera_Start_Width,
				# Slider_Camera_End_Width,
				#
				Slider_Ratio_Crop_Camera_Start,
				Slider_Ratio_Crop_Camera_End,
			] ,
			outputs = \
			[
				ImageBox_Output_Original_Stepping_Tab,
				# Slider_Camera_Start_Width,
				# Slider_Camera_End_Width,

				Label_Camera_Start_Stepping_Tab,
				Label_Camera_End_Stepping_Tab,
			] ,
		)
		CheckBox_Auto_Zoom_Stepping_Tab.change\
		(
			fn=DoOnCheckBox_Auto_Zoom_Stepping_TabChanged,
			inputs = \
			[
				CheckBox_Auto_Zoom_Stepping_Tab,
				ImageBox_Input_Original_Stepping_Tab,
			] ,
			outputs = \
			[
				ImageBox_Output_Original_Stepping_Tab,
			] ,
		)
		# Slider_Camera_Start_Width.change\
		# (
		# 	fn = DoOnSlider_Camera_Start_Width_Stepping_TabChanged,
		# 	inputs =\
		# 	[
		# 		Slider_Camera_Start_Width,
		# 	],
		# 	outputs =\
		# 	[
		# 		# Slider_Camera_Start_Width,
		# 	],
		# )
		# Slider_Camera_Start_Height.change\
		# (
		# 	fn = DoOnSlider_Camera_Start_Height_Stepping_TabChanged,
		# 	inputs =\
		# 	[
		# 		Slider_Camera_Start_Height,
		# 	],
		# 	outputs =\
		# 	[
		# 		# Slider_Camera_Start_Height,
		# 	],
		# )
		# Slider_Camera_End_Width.change\
		# (
		# 	fn = DoOnSlider_Camera_End_Width_Stepping_TabChanged,
		# 	inputs =\
		# 	[
		# 		Slider_Camera_End_Width,
		# 	],
		# 	outputs =\
		# 	[
		# 		# Slider_Camera_End_Width,
		# 	],
		# )
		# Slider_Camera_End_Height.change\
		# (
		# 	fn = DoOnSlider_Camera_End_Height_Stepping_TabChanged,
		# 	inputs =\
		# 	[
		# 		Slider_Camera_End_Height,
		# 	],
		# 	outputs =\
		# 	[
		# 		# Slider_Camera_End_Height,
		# 	],
		# )
		Button_Run_With_Steps_Stepping_Tab.click\
		(
			fn = DoOnButton_Run_With_Steps_Stepping_TabClicking,
			inputs =\
			[
				ImageBox_Input_Original_Stepping_Tab,
				CheckBox_BoundingBox_Stepping_Tab,
				CheckBox_InstanceMask_Stepping_Tab,
				CheckBox_InstanceContour_Stepping_Tab,
				CheckBox_Tag_List_Stepping_Tab,
				Slider_Mask_α_Stepping_Tab,
			],
			outputs =\
			[
				# ImageBox_After_1st_Process_Stepping_Tab,
				#
				ImageBox_After_2nd_Process_Stepping_Tab,
				#
				ImageBox_After_3rd_Process_Stepping_Tab,
				#
				ImageBox_After_4th_Process_Stepping_Tab,
				#
				ImageBox_After_5th_Process_Stepping_Tab,
				#
				#
				ImageBox_Output_Original_Stepping_Tab,
				#
				#
				Button_Generate_Camera_View_Stepping_Tab,
				#
				#
				Label_Progress_Instance_Generating,
			],
		)
		Button_Generate_Camera_View_Stepping_Tab.click\
		(
			fn = DoOnButton_Generate_Camera_View_Stepping_TabClicking,
			inputs =\
			[
				CheckBox_IsInpainting_Stepping_Tab,
				#
				Slider_Ratio_Crop_Camera_Start,
				# Slider_Camera_Start_Width,
				# Slider_Camera_Start_Height,
				#
				Slider_Ratio_Crop_Camera_End,
				# Slider_Camera_End_Width,
				# Slider_Camera_End_Height,
				#
				Slider_Number_Frame_Stepping_Tab,
				CheckBox_Auto_Zoom_Stepping_Tab,
				#
				Slider_Depthor_Stepping_Tab,
				CheckBox_Depth_Field_Stepping_Tab,
			],
			outputs =\
			[
				ImageBox_Output_PositiveFilm_Start_Stepping_Tab,
				ImageBox_Output_PositiveFilm_End_Stepping_Tab,
				ImageBox_Output_Mask_Start_Stepping_Tab,
				ImageBox_Output_Mask_End_Stepping_Tab,
				BatchedImageBox_Output_ListFrame_Stepping_Tab,
				#
				#
				Button_Encode_Video_Stepping_Tab,
				VideoBox_Output_Stepping_Tab,
				#
				#
				Label_Progress_Camera_Generating,
			],
		)
		Button_Encode_Video_Stepping_Tab.click\
		(
			fn = DoOnButton_Encode_Video_Stepping_TabClicking,
			# inputs =\
			# [
			# 	# To Do
			# ],
			outputs =\
			[
				VideoBox_Output_Stepping_Tab,
				File_Output_Downloaing_Stepping_Tab,
			],
			# outputs = VideoBox_Output_Stepping_Tab,
		)

# ！Gradio不能直接呈现URL，只能呈现HTML，但是Flask是基于URL，仅HTML则只有静态Framework
def Create_Original_Demonstration_Tab () :

	with Framework.Tab ( F"Original Demonstration-Ken Burns" ) :
		Framework.HTML(F"<a href=\"http://localhost:8080\">Original Demonstration</a>")

#endregion Anime Instance Segmentation

#endregion Customised

def Create_Information_Tab () :

	with Framework.Tab ( F"Information" ):  # Including Image information and Model information
		with Framework.Row():
			ImageBox_Input_TabInformation=Create_Batch_Input_ImageBox()
			ImageBox_Output=Create_Output_BatchedImageBox()		# !Need to be changed to Status Control|Component


def DoOnCheckBox_IsVerbosing_Setting_TabChanged (IsVerbosing_In) :

	global IsVerbosing_Global
	IsVerbosing_Global=IsVerbosing_In

	# Debug
	print(F"{IsVerbosing_In=} in {Inspect.currentframe().f_code.co_name}()")
	print(F"{IsVerbosing_Global=} in {Inspect.currentframe().f_code.co_name}()")


def Create_Setting_Tab () :

	with Framework.Tab ( F"Setting" ):
		# with Framework.Row():
		# 	Button_Apply_Setting = Create_ApplySetting_Button ()
		# 	Button_Restart = Create_Restart_Button ()

		with Framework.Row\
		(
			Title=F"Option",
			# wrap=False,
			# min_width = 180,
		):
			with Framework.Column():
				CheckBox_IsVerbosing_Stepping_Tab = Create_Step_CheckBox\
				(
					Name_In = F"Verbose",
					Value_Default_In = False,
					Tip_In = F"If checked, Programme will generate intermediate | temporary Image Results in its Root Directory.",
				)



		#
		CheckBox_IsVerbosing_Stepping_Tab.change\
		(
			fn = DoOnCheckBox_IsVerbosing_Setting_TabChanged,
			inputs =\
			[
				CheckBox_IsVerbosing_Stepping_Tab,
			],
			outputs =\
			[
				# To Do
			],
		)


#endregion Function | Method