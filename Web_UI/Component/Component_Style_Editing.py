# #region Script Pre-processing
#
# # Test by Francis
# import os as OS
#
# print ( F"{OS.getcwd()=}" )
# import os.path as Pathing
# import sys as Environment
#
# Directory_Current = Pathing.dirname ( Pathing.abspath ( __file__ ) )
# Directory_Project = Pathing.dirname(Pathing.dirname  ( Directory_Current ))
# print ( F"{Directory_Project=}" )
# Environment.path.append ( Directory_Project )
#
# #endregion Script Pre-processing



#region Import

#region System

import base64 as Base64
import inspect as Inspect
import io as IO
import json as JSON
import os as OS
import os.path as Path
from collections import namedtuple as NamedTuple

# endregion System


# region 3rd Reference

# from PIL.Image import Resampling as Resampling
import cv2 as Vision
import gradio as Framework
import numpy as Number
import PIL.Image as Imaging

# endregion 3rd Reference


# region Local Reference

import animeinsseg as Anime_Instance_Segmentation
import animeinsseg.inpainting as Inpainting
# import repaint_person as Style_Editing
import utils.io_utils as Utility
from anime_3dkenburns.kenburns_effect import KenBurnsPipeline as Pipeline

import Web_UI.Common.Constant as Constant

#endregion Local Reference

#endregion Import



#region Definition

#region Field

# ？叫Sample更合适
Directory_Image_Example=F"examples"

List_Method_Inpainting_Fill =\
[
	"fill",
	"original",
	"latent_noise",
	"latent_nothing",
]
Location_Point=NamedTuple(F"Point",F"Horizon Vertical")

# Debug Control Flag
IsAllActivatedControlDisplayed=False

# Default Variable
Sampler_Default = F"DPM++ 2M Karras"
Method_Inpainting_Fill_Default = F"original"
Detector_Default=F"models/AnimeInstanceSegmentation/rtmdetl_e60.ckpt"
Prompt_Foreground_Positive_Default = F"ultra-detailed, ultra high resolution"
Prompt_Positive_Default= F"masterpiece, best quality"
Prompt_Negative_Default= F"lowres, ((bad anatomy)), ((bad hands)), text, missing finger, extra digits, fewer digits, blurry, ((mutated hands and fingers)), (poorly drawn face), ((mutation)), ((deformed face)), (ugly), ((bad proportions)), ((extra limbs)), extra face, (double head), (extra head), ((extra feet)), monster, logo, cropped, worst quality, low quality, normal quality, jpeg, humpbacked, long body, long neck, ((jpeg artifacts))"
Prompt_Background_Negative_Default= F"((person)), character, 1girl, 1boy"
Mask_α_Default=0.6
Model_Default=F"control_v11p_sd15s2_lineart_anime [3825e83e]"
Directory_Output_Default=F"repaint_output"
Progress_Default=F"Display the Progress"

# Constant
Limit_Instance=10
Granularity_Pixel=32

# ！需要使用YAML存储Configuration
# Local Debug
# Address_Default_Model_Remote = F"http://localhost:7860/sdapi/v1/img2img"
# Release in Server
# Call from Local Net
# Address_Default_Model_Remote = F"http://localhost:8888/sdapi/v1/img2img"
# Call from Remote Net
Address_Default_Model_Remote = F"http://localhost:7860/sdapi/v1/img2img"

Pipeline=Pipeline(F"configs/3dkenburns.yaml")



# Global
List_Option_Shared_Global = None
Instance_Global = None
List_Prompt_Foreground_Positive_Global = None
List_Mask_Global = None
Image_Processing_Global = None
# Prompt_Positive_Input_Global = None
# Prompt_Negative_Input_Global = None
Size_Approximated_Global:Location_Point = None
Detector_Global = None

#endregion Field

#region Method

#region Interface

# ！Pending
# ！考虑将Instance Row Programmaticalise
def Create_Instance_Row (Number_In) :

	with Framework.Row () :
		with Framework.Column \
		(
			min_width = 400 ,
		) :
			Title_Instance = F"Instance {Number_In}"
			Framework.Markdown ( F"#### {Title_Instance}" )
			ImageBox_Instances_Output = Framework.Image \
			(
				label = Title_Instance ,
				show_label = False ,
				# type=F"pil",
				type = F"numpy" ,
				interactive = False ,
				editable = False ,
				height = F"auto",
				width = F"auto",
			)

		with Framework.Column () :
			Title_Prompt_Positive = F"Positive Prompts {Number_In}"
			Framework.Markdown ( F"#### {Title_Prompt_Positive}" )
			TextBox_Prompt_Positive_Input = Framework.Textbox \
			(
				label = Title_Prompt_Positive ,
				show_label = False ,
				lines = 3 ,  # As known as Height
				value = F"" ,
			)


def Create_Style_Editing_Demonstration_Tab():

	Title=F"Style Editing"

	with Framework.Tab(Title):
		# Framework.Markdown(F"# In-paint Instances of People | Persons (≈ 50 ~ 80 Seconds per Person).")

		# Positive Text Input Row

		# Step 0. Choosing Image
		Framework.Markdown(F"# Step 0. Choose Image")

		with Framework.Row():
			# Input Image + Output Image Row
			with Framework.Column():
				Framework.Markdown ( F"#### Input Image" )
				# Framework.Markdown(F"Path to input Image.")
				ImageBox_Original_Input=Framework.Image \
				(
					label = F"Input Image-Original" ,
					# type=F"pil",
					# type = F"numpy" ,
					type = F"filepath" ,
					show_label = False,
					interactive = True,
					# editable = False,
					# minwidth=300,
					# shape = (None,400),
					# height = 600,
				)

			with Framework.Column():
				List_Example=Framework.Examples\
				(
					examples=\
					[
						OS.path.join ( Directory_Image_Example , Carrier_Path_Image ) for Carrier_Path_Image in OS.listdir ( Directory_Image_Example )
					],
					inputs = \
					[
						ImageBox_Original_Input,
					],
					examples_per_page = 40,
				)

		# # Button Row
		# with Framework.Row():
		# 	with Framework.Column():
				# Framework.Markdown(F"Re-painting Person **One by One**.")
				Button_Run_1=Framework.Button\
				(
					F"Run Style Editing 1",
					visible = False,
				)

		Framework.Markdown(F"---")

		# Step 1. Generate Instance Masks
		Framework.Markdown(F"# Step 1. Generate Instance Masks")

		with Framework.Row():
			with Framework.Column \
			(
				min_width = 400 ,
			) :
				Title_After_Drawing_Instances=F"Instance Masks with Tags"
				Framework.Markdown(F"#### {Title_After_Drawing_Instances}")
				ImageBox_Instances_Output = Framework.Image \
				(
					label = Title_After_Drawing_Instances ,
					show_label = False,
					# type=F"pil",
					type = F"numpy" ,
					interactive = False,
					# editable = False,
					height = 600,
				)

			with Framework.Column():
				Framework.Markdown(F"#### Progress")
				Label_Progress_Instance_Generating=Framework.Label\
				(
					value =Progress_Default,
					# label = F"Progress",
					label = F"",
					# show_label = False,
				)

				Framework.Markdown(F"---")

				Framework.Markdown(F"## Background / General")

				Title_Prompt_Positive=F"Positive Prompts"
				Framework.Markdown(F"#### {Title_Prompt_Positive}")
				TextBox_Prompt_Positive_General_Input=Framework.Textbox\
				(
					label = Title_Prompt_Positive,
					show_label = False,
					lines = 3,	  # As known as Height
					value = F"",
				)

				Title_Prompt_Negative=F"Negative Prompts"
				Framework.Markdown(F"#### {Title_Prompt_Negative}")
				TextBox_Prompt_Negative_General_Input=Framework.Textbox\
				(
					label = Title_Prompt_Negative,
					show_label = False,
					lines = 5,	  # As known as Height
					value = F"",
				)

			# with Framework.Column():
				BatchedCheckBox_Option_General_Input=Framework.CheckboxGroup \
				(
					value =\
					[
						F"White Background" ,
						F"To Grey",
					],
					show_label = False,
					visible = False,
				)

			# with Framework.Row():
			# 	with Framework.Column\
			# 	(
			# 		min_width = 100,
			# 	):
				CheckBox_Instance_PositiveFilm_Input=Create_CheckBox \
				(
					Name_In = F"Instance in Positive Film" ,
					Value_Default_In = False ,
					# IsVisible_In = False,
				)

				# with Framework.Column\
				# (
				# 	min_width = 100,
				# ):
				CheckBox_To_Grey_Input=Create_CheckBox \
				(
					Name_In = F"To Grey" ,
					Value_Default_In = False ,
				)

		# # Button Row
		# with Framework.Row():
		# 	with Framework.Column():
				# Framework.Markdown(F"Re-painting Person **One by One**.")
				Button_Run_2=Framework.Button\
				(
					F"Generate Instance Masks & Tags",
				)

		Framework.Markdown(F"---")

		# Step 2. Edit Tags / Prompts in Instance
		Framework.Markdown(F"# Step 2. Edit Tags / Prompts in Instance")

		# Button Row
		with Framework.Row(visible = False):
			with Framework.Column():
				# Framework.Markdown(F"Re-painting Person **One by One**.")
				Button_Run_3=Framework.Button\
				(
					F"Run Style Editing 3",
				)

		# Create_Instance_Row()

		# Row 1
		with Framework.Row (visible = True) as Row_Instance_1:
			with Framework.Column \
			(
				min_width = 400 ,
			) :
				Title_Instance_1 = F"Instance 1"
				Framework.Markdown ( F"#### {Title_Instance_1}" )
				ImageBox_Instances_1_Output = Framework.Image \
				(
					label = Title_Instance_1 ,
					show_label = False ,
					# type=F"pil",
					type = F"numpy" ,
					interactive = False ,
					# editable = False ,
					height = 600 ,
				)

			with Framework.Column () :
				Title_Prompt_Foreground_Positive_1 = F"Positive Prompts of Instance 1"
				Framework.Markdown ( F"#### {Title_Prompt_Foreground_Positive_1}" )
				TextBox_Prompt_Foreground_Positive_1_Input = Framework.Textbox \
				(
					label = Title_Prompt_Foreground_Positive_1 ,
					show_label = False ,
					lines = 3 ,  # As known as Height
					value = F"" ,
					# height = 600 ,
				)

		# Row 2
		with Framework.Row (visible = False)  as Row_Instance_2:
			with Framework.Column \
			(
				min_width = 400 ,
			) :
				Title_Instance_2 = F"Instance 2"
				Framework.Markdown ( F"#### {Title_Instance_2}" )
				ImageBox_Instances_2_Output = Framework.Image \
				(
					label = Title_Instance_2 ,
					show_label = False ,
					# type=F"pil",
					type = F"numpy" ,
					interactive = False ,
					# editable = False ,
					height = 600 ,
				)

			with Framework.Column () :
				Title_Prompt_Foreground_Positive_2 = F"Positive Prompts of Instance 2"
				Framework.Markdown ( F"#### {Title_Prompt_Foreground_Positive_2}" )
				TextBox_Prompt_Foreground_Positive_2_Input = Framework.Textbox \
				(
					label = Title_Prompt_Foreground_Positive_2 ,
					show_label = False ,
					lines = 3 ,  # As known as Height
					value = F"" ,
					# height = 600 ,
				)

		# Row 3
		with Framework.Row (visible = False)  as Row_Instance_3:
			with Framework.Column \
			(
				min_width = 400 ,
			) :
				Title_Instance_3 = F"Instance 3"
				Framework.Markdown ( F"#### {Title_Instance_3}" )
				ImageBox_Instances_3_Output = Framework.Image \
				(
					label = Title_Instance_3 ,
					show_label = False ,
					# type=F"pil",
					type = F"numpy" ,
					interactive = False ,
					# editable = False ,
					height = 600 ,
				)

			with Framework.Column () :
				Title_Prompt_Foreground_Positive_3 = F"Positive Prompts of Instance 3"
				Framework.Markdown ( F"#### {Title_Prompt_Foreground_Positive_3}" )
				TextBox_Prompt_Foreground_Positive_3_Input = Framework.Textbox \
				(
					label = Title_Prompt_Foreground_Positive_3 ,
					show_label = False ,
					lines = 3 ,  # As known as Height
					value = F"" ,
					# height = 600 ,
				)

		# Row 4
		with Framework.Row (visible = False)  as Row_Instance_4:
			with Framework.Column \
			(
				min_width = 400 ,
			) :
				Title_Instance_4 = F"Instance 4"
				Framework.Markdown ( F"#### {Title_Instance_4}" )
				ImageBox_Instances_4_Output = Framework.Image \
				(
					label = Title_Instance_4 ,
					show_label = False ,
					# type=F"pil",
					type = F"numpy" ,
					interactive = False ,
					# editable = False ,
					height = 600 ,
				)

			with Framework.Column () :
				Title_Prompt_Foreground_Positive_4 = F"Positive Prompts of Instance 4"
				Framework.Markdown ( F"#### {Title_Prompt_Foreground_Positive_4}" )
				TextBox_Prompt_Foreground_Positive_4_Input = Framework.Textbox \
				(
					label = Title_Prompt_Foreground_Positive_4 ,
					show_label = False ,
					lines = 3 ,  # As known as Height
					value = F"" ,
					# height = 600 ,
				)

		# Row 5
		with Framework.Row (visible = False)  as Row_Instance_5:
			with Framework.Column \
			(
				min_width = 400 ,
			) :
				Title_Instance_5 = F"Instance 5"
				Framework.Markdown ( F"#### {Title_Instance_5}" )
				ImageBox_Instances_5_Output = Framework.Image \
				(
					label = Title_Instance_5 ,
					show_label = False ,
					# type=F"pil",
					type = F"numpy" ,
					interactive = False ,
					# editable = False ,
					height = 600 ,
				)

			with Framework.Column () :
				Title_Prompt_Foreground_Positive_5 = F"Positive Prompts of Instance 5"
				Framework.Markdown ( F"#### {Title_Prompt_Foreground_Positive_5}" )
				TextBox_Prompt_Foreground_Positive_5_Input = Framework.Textbox \
				(
					label = Title_Prompt_Foreground_Positive_5 ,
					show_label = False ,
					lines = 3 ,  # As known as Height
					value = F"" ,
					# height = 600 ,
				)

		# Row 6
		with Framework.Row (visible = False)  as Row_Instance_6:
			with Framework.Column \
			(
				min_width = 400 ,
			) :
				Title_Instance_6 = F"Instance 6"
				Framework.Markdown ( F"#### {Title_Instance_6}" )
				ImageBox_Instances_6_Output = Framework.Image \
				(
					label = Title_Instance_6 ,
					show_label = False ,
					# type=F"pil",
					type = F"numpy" ,
					interactive = False ,
					# editable = False ,
					height = 600 ,
				)

			with Framework.Column () :
				Title_Prompt_Foreground_Positive_6 = F"Positive Prompts of Instance 6"
				Framework.Markdown ( F"#### {Title_Prompt_Foreground_Positive_6}" )
				TextBox_Prompt_Foreground_Positive_6_Input = Framework.Textbox \
				(
					label = Title_Prompt_Foreground_Positive_6 ,
					show_label = False ,
					lines = 3 ,  # As known as Height
					value = F"" ,
					# height = 600 ,
				)

		# Row 7
		with Framework.Row (visible = False)  as Row_Instance_7:
			with Framework.Column \
			(
				min_width = 400 ,
			) :
				Title_Instance_7 = F"Instance 7"
				Framework.Markdown ( F"#### {Title_Instance_7}" )
				ImageBox_Instances_7_Output = Framework.Image \
				(
					label = Title_Instance_7 ,
					show_label = False ,
					# type=F"pil",
					type = F"numpy" ,
					interactive = False ,
					# editable = False ,
					height = 600 ,
				)

			with Framework.Column () :
				Title_Prompt_Foreground_Positive_7 = F"Positive Prompts of Instance 7"
				Framework.Markdown ( F"#### {Title_Prompt_Foreground_Positive_7}" )
				TextBox_Prompt_Foreground_Positive_7_Input = Framework.Textbox \
				(
					label = Title_Prompt_Foreground_Positive_7 ,
					show_label = False ,
					lines = 3 ,  # As known as Height
					value = F"" ,
					# height = 600 ,
				)

		# Row 8
		with Framework.Row (visible = False)  as Row_Instance_8:
			with Framework.Column \
			(
				min_width = 400 ,
			) :
				Title_Instance_8 = F"Instance 8"
				Framework.Markdown ( F"#### {Title_Instance_8}" )
				ImageBox_Instances_8_Output = Framework.Image \
				(
					label = Title_Instance_8 ,
					show_label = False ,
					# type=F"pil",
					type = F"numpy" ,
					interactive = False ,
					# editable = False ,
					height = 600 ,
				)

			with Framework.Column () :
				Title_Prompt_Foreground_Positive_8 = F"Positive Prompts of Instance 8"
				Framework.Markdown ( F"#### {Title_Prompt_Foreground_Positive_8}" )
				TextBox_Prompt_Foreground_Positive_8_Input = Framework.Textbox \
				(
					label = Title_Prompt_Foreground_Positive_8 ,
					show_label = False ,
					lines = 3 ,  # As known as Height
					value = F"" ,
					# height = 600 ,
				)

		# Row 9
		with Framework.Row (visible = False)  as Row_Instance_9:
			with Framework.Column \
			(
				min_width = 400 ,
			) :
				Title_Instance_9 = F"Instance 9"
				Framework.Markdown ( F"#### {Title_Instance_9}" )
				ImageBox_Instances_9_Output = Framework.Image \
				(
					label = Title_Instance_9 ,
					show_label = False ,
					# type=F"pil",
					type = F"numpy" ,
					interactive = False ,
					# editable = False ,
				)

			with Framework.Column () :
				Title_Prompt_Foreground_Positive_9 = F"Positive Prompts of Instance 9"
				Framework.Markdown ( F"#### {Title_Prompt_Foreground_Positive_9}" )
				TextBox_Prompt_Foreground_Positive_9_Input = Framework.Textbox \
				(
					label = Title_Prompt_Foreground_Positive_9 ,
					show_label = False ,
					lines = 3 ,  # As known as Height
					value = F"" ,
					# height = 600 ,
				)

		# Row 10
		with Framework.Row (visible = False)  as Row_Instance_10:
			with Framework.Column \
			(
				min_width = 400 ,
			) :
				Title_Instance_10 = F"Instance 10"
				Framework.Markdown ( F"#### {Title_Instance_10}" )
				ImageBox_Instances_10_Output = Framework.Image \
				(
					label = Title_Instance_10 ,
					show_label = False ,
					# type=F"pil",
					type = F"numpy" ,
					interactive = False ,
					# editable = False ,
				)

			with Framework.Column () :
				Title_Prompt_Foreground_Positive_10 = F"Positive Prompts of Instance 10"
				Framework.Markdown ( F"#### {Title_Prompt_Foreground_Positive_10}" )
				TextBox_Prompt_Foreground_Positive_10_Input = Framework.Textbox \
				(
					label = Title_Prompt_Foreground_Positive_10 ,
					show_label = False ,
					lines = 3 ,  # As known as Height
					value = F"" ,
				)

		Framework.Markdown(F"---")

		# Step 3. Edit Prompts in General
		Framework.Markdown(F"# Step 3. Edit Prompts in General")
		with Framework.Row () :
			with Framework.Column():
				Framework.Markdown(F"## General")

				Title_Prompt_Positive_Input=F"Positive Prompts"
				Framework.Markdown(F"#### {Title_Prompt_Positive_Input}" )
				TextBox_Prompt_Positive_Input=Framework.Textbox\
				(
					label = Title_Prompt_Positive_Input,
					show_label = False,
					lines = 3,	  # As known as Height
					# placeholder = F"Positive Prompt here",
					# placeholder = Prompt_Positive_Default,
					# value = F"",
					value = Prompt_Positive_Default,
					# info = F"Positive Prompt to use, including follows as default: masterpiece, best quality.",

					# 设置上限+当前字符量额统计、限制

					# Debug
					# value = F"Bill Gates rides on a horse."
				)

		# Negative Text Input Row
		# with Framework.Row():
		# 	with Framework.Column():
				Title_Prompt_Negative_Input=F"Negative Prompts"
				Framework.Markdown(F"#### {Title_Prompt_Negative_Input}" )
				TextBox_Prompt_Negative_Input=Framework.Textbox\
				(
					label = Title_Prompt_Negative_Input,
					show_label = False,
					lines = 4,	  # As known as Height
					# placeholder = F"Negative Prompt here",
					# placeholder = Prompt_Negative_Default,
					# value = F"",
					value = Prompt_Negative_Default,

					# 设置上限+当前字符量额统计、限制

					# Debug
					# value = F"Bill Gates rides on a horse."
				)

			with Framework.Column () :
				Framework.Markdown(F"## Background")

				Title_Prompt_Background_Negative_Input=F"Negative Prompts"
				Framework.Markdown(F"#### {Title_Prompt_Background_Negative_Input}" )
				TextBox_Prompt_Background_Negative_Input=Framework.Textbox\
				(
					label = Title_Prompt_Background_Negative_Input,
					show_label = False,
					lines = 3,	  # As known as Height
					value = Prompt_Background_Negative_Default,
					# info = F"Background Negative Prompt | Input.",

					# 设置上限+当前字符量额统计、限制

					# Debug
					# value = F"Bill Gates rides on a horse."
				)

		Framework.Markdown(F"---")

		# Step 4. Result
		Framework.Markdown(F"# Step 4. Result")


				# Framework.Markdown(F"---")

		# Background Negative Text Input Row
		with Framework.Row () :
			with Framework.Column():
				Framework.Markdown(F"#### Progress")

		# Button Row
		with Framework.Row():
			with Framework.Column():
				Label_Progress_Result_Generating=Framework.Label\
				(
					value =Progress_Default,
					# label = F"Progress",
					label = F"",
					# show_label = False,
				)

			with Framework.Column():
				# Framework.Markdown(F"Re-painting Person **One by One**.")
				Framework.Markdown(F" ")
				Button_Run_4=Framework.Button\
				(
					F"{Constant.Character_NewLine}Run Style Editing{Constant.Character_NewLine}",
				)

				# Framework.Markdown(F"---")

		Framework.Markdown ( F"---" )

		# Slider-Size Row
		with Framework.Row():
			with Framework.Column():
				Slider_Width_Output_Input = Framework.Slider \
				(
					value = 768 ,  # 相当于default
					minimum = Granularity_Pixel ,  # ！考虑改为Constant
					maximum = 1_0000 ,  # ！考虑改为Constant
					step = Granularity_Pixel ,
					# multiselect = True,
					label = F"Result Image Width" ,
					# info = F"Width of Output Image.",
				)

			with Framework.Column():
				Slider_Height_Output_Input=Framework.Slider\
				(
					value = 768 ,  # 相当于default
					minimum = Granularity_Pixel ,      # ！考虑改为Constant
					maximum = 1_0000 ,      # ！考虑改为Constant
					step = Granularity_Pixel ,
					# multiselect = True,
					label = F"Result Image Height" ,
					# info = F"Height of Output Image.",
				)

		Framework.Markdown(F"---")

		# Example Input Image List Row
		with Framework.Row():
			with Framework.Column():
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
					height = 600,
				)

			with Framework.Column():
				Title_Output=F"Output Image"
				Framework.Markdown(F"#### {Title_Output}")
				ImageBox_Output=Framework.Image \
				(
					label = Title_Output ,
					# type=F"pil",
					# type = F"numpy" ,
					type = F"filepath" ,
					show_label = False,
					interactive = False,
					# editable = False,
					# minwidth=300,
					# shape = (None,400),
					height = 600,
				)

		Framework.Markdown(F"---")

		# Step 5. Options
		Framework.Markdown(F"# Step 5. Options")

		# Processor Address Row
		with Framework.Row():
			with Framework.Column\
			(
				visible = IsAllActivatedControlDisplayed,
			):
				# ！缺少限定词，是什么部分的URL|Remote地址
				# ！更换为Combo Box更佳
				Hover_Processor=F"Image → Image URL."
				Title_Processor= F"Processor Address"

				Framework.Markdown\
				(
					F"#### {Title_Processor}",
				)
				ComboBox_Address_Processor_Input=Framework.Dropdown\
				(
					choices = List_Processor () ,
					label = Title_Processor,
					info = Hover_Processor,
					visible = False,
				)
				TextBox_Address_Processor_Input=Framework.Textbox\
				(
					label = Title_Processor,
					show_label = False,
					lines = 1,	  # As known as Height
					value = Address_Default_Model_Remote,     # ！应该使用对User输入|理解更友好的Control呈现，如：Combo Box
					info = Hover_Processor,

					# 设置上限+当前字符量额统计、限制

					# Debug
					# value = F"Bill Gates rides on a horse."
				)

			with Framework.Column():
				# ！更换为Combo Box更佳
				Hover_Detector=F"Detector Check Point."
				Title_Detector= F"Detector Check Point"
				ComboBox_Detector_Input=Framework.Dropdown\
				(
					choices = List_Detector (),
					label = Title_Detector,
					# info = Hover_Detector,
					visible = False,
				)
				TextBox_Path_CheckPoint_Detector_Input=Framework.Textbox\
				(
					label = Title_Detector,
					show_label = True,
					lines = 1,	  # As known as Height
					value = Detector_Default,
					# info = Hover_Detector,

					# 设置上限+当前字符量额统计、限制

					# Debug
					# value = F"Bill Gates rides on a horse."
					visible = IsAllActivatedControlDisplayed,
				)

		# with Framework.Row():
		# 	with Framework.Column():
				TextBox_Directory_Output_Input=Framework.Textbox\
				(
					label = F"Save Directory",
					show_label = True,
					lines = 1,	  # As known as Height
					value = Directory_Output_Default,     # ！应该使用对User输入|理解更友好的Control呈现，如：Combo Box
					# info = F"Image → Image URL.",

					# 设置上限+当前字符量额统计、限制

					# Debug
					# value = F"Bill Gates rides on a horse."
					visible = IsAllActivatedControlDisplayed,
				)

		# Check Box + Button Row
		with Framework.Row(visible = IsAllActivatedControlDisplayed):
			with Framework.Column():
				Framework.Markdown(F"Re-painting Person **One by One**.")
				CheckBox_One_By_One_Input=Create_CheckBox \
				(
					Name_In = F"One by One" ,
					Value_Default_In = True ,
				)

			with Framework.Column():
				Framework.Markdown(F"Verbose intermediate Images | Results or not.")
				CheckBox_Intermediate_Save_Input=Create_CheckBox \
				(
					Name_In = F"Save Intermediate Image | Result" ,
					Value_Default_In = True ,
				)

		# Temporary Output Image Row
		with Framework.Row\
		(
			visible = IsAllActivatedControlDisplayed,
		) :
			with Framework.Column () :
				Framework.Markdown\
				(
					F"Final Result | Image.",
					visible = False,
				)
				BatchedImageBox_Positive_Film_Output = Create_Output_BatchedImageBox \
				(
					Name_In = F"Result-Positive Film",
				)

			with Framework.Column () :
				Framework.Markdown\
				(
					F"Final Result | Image.",
					visible = False,
				)
				BatchedImageBox_Mask_Output = Create_Output_BatchedImageBox \
				(
					Name_In = F"Result-Mask",
				)

		# Slider-Size Row
		with Framework.Row():
			# ！位置待定，是Intermediate | Temporary Result
			with Framework.Column(visible = IsAllActivatedControlDisplayed):
				Framework.Markdown\
				(
					F"Result | Image in Temporary | Intermediate.",
				)
				ImageBox_Imediate_Output = Framework.Image \
				(
					label = F"Output Image-Temporary | Intermediate" ,
					# type=F"pil",
					type = F"numpy" ,
					# type = F"filepath" ,
				)

		# Controls Et Cetera Row
		# with Framework.Row():
			with Framework.Column():
				# ？就是Sampling Step
				Slider_Step_Input=Framework.Slider\
				(
					value = 24 ,  # 相当于default
					minimum = 1 ,      # ！考虑改为Constant
					maximum = 150 ,      # ！考虑改为Constant
					step = 1 ,
					# multiselect = True,
					label = F"Step" ,
					info = F"Number of Stable Diffusion Steps.",
				)

			with Framework.Column():
				Slider_Scale_ClassifierFreeGuidance_Input=Framework.Slider\
				(
					value = 9 ,  # 相当于default
					minimum = 1 ,      # ！考虑改为Constant
					maximum = 30 ,      # ！考虑改为Constant
					step = 1 ,
					# multiselect = True,
					label = F"Classifier Free Guidance Scale" ,
					info = F"Scale of Classifier Free Guidance, id est how strongly the Image should conform to prompt.",
				)

			with Framework.Column():
				# ！更换为Combo Box更佳，但根据SD WebUI的入参显示，这是Customised Script的Argument
				Hover_Sampler=F"Name of Samplers to use."
				ComboBox_Sampler_Input=Framework.Dropdown\
				(
					choices = List_Sampler (),
					# value =     # ！需要找到设置默认值的方法
					label = F"Sampler:",
					value =Sampler_Default,        # ！考虑Programme化
					info = Hover_Sampler,
					visible = True,
				)
				TextBox_Sampler_Input=Framework.Text\
				(
					label = F"Sampler:",
					show_label = True,
					lines = 1,	  # As known as Height
					value = Sampler_Default,
					info = Hover_Sampler,
					visible = False,

					# 设置上限+当前字符量额统计、限制

					# Debug
					# value = F"Bill Gates rides on a horse."
				)

			with Framework.Column():
				ComboBox_Method_Filling_Input=Framework.Dropdown\
				(
					choices = List_Filling_Method (),
					value = Method_Inpainting_Fill_Default,        # ！考虑Programme化
					label = F"Filling Method:",
					info = F"The Filling Method to use for In-painting.",
				)

			with Framework.Column\
			(
				visible = IsAllActivatedControlDisplayed ,
			):
				TextBox_Path_Configuration_Input=Framework.Textbox\
				(
					label = F"Configuration",
					show_label = True,
					lines = 1,	  # As known as Height
					value = F"",
					info = F"Re-paint Configuration Path.",

					# 设置上限+当前字符量额统计、限制

					# Debug
					# value = F"Bill Gates rides on a horse."
				)

			with Framework.Column():
				CheckBox_Resolution_Full_Inpainting_Input=Create_CheckBox \
				(
					Name_In = F"In-painting Full Resolution" ,
					Value_Default_In = True ,
				)

			with Framework.Column():
				CheckBox_Infer_Tagger_Input=Create_CheckBox \
				(
					Name_In = F"Infer Tagger" ,
					Value_Default_In = True ,
				)

			with Framework.Column():
				Slider_Strength_Denoising_Input=Framework.Slider\
				(
					value = 0.75 ,  # 相当于default
					minimum = 0 ,      # ！考虑改为Constant
					maximum = 1 ,      # ！考虑改为Constant
					step = 0.01 ,
					# multiselect = True,
					label = F"Denoising Strength" ,
					info = F"How much to disregard Original Image.",
				)

			with Framework.Column():
				Slider_Blur_Mask_Input=Framework.Slider\
				(
					value = 4 ,  # 相当于default
					minimum = 0 ,      # ！考虑改为Constant
					maximum = 64 ,      # ！考虑改为Constant
					step = 1 ,
					# multiselect = True,
					label = F"Mask Blur" ,
					info = F"Blur Radius of Gaussian Filter to apply Mask.",
				)

			with Framework.Column():
				Slider_Resolution_Inpainting_Input=Framework.Slider\
				(
					value = 640 ,  # 相当于default
					minimum = 1 ,      # ！考虑改为Constant
					maximum = 1_0000 ,      # ！考虑改为Constant
					step = 1 ,
					# multiselect = True,
					label = F"In-painting Resolution" ,
					# info = F"In-painting Resolution.",
				)

			with Framework.Column():
				Slider_Padding_Resolution_Full_Inpainting_Input=Framework.Slider\
				(
					value = 32 ,  # 相当于default
					minimum = 1 ,      # ！考虑改为Constant
					maximum = 256 ,      # ！考虑改为Constant
					step = 1 ,
					# multiselect = True,
					label = F"In-painting Full Resolution Padding" ,
					# info = F"In-painting Full Resolution Padding.",
				)



		# Bind Event Handler
		Button_Run_2.click\
		(
			fn=DoOnButton_Run_2Clicking,
			inputs = \
			[
				ImageBox_Original_Input,

				CheckBox_One_By_One_Input,
				# TextBox_Prompt_Positive_Input,
				# TextBox_Prompt_Negative_Input,
				# Slider_Width_Output_Input,
				# Slider_Height_Output_Input,
				Slider_Step_Input,
				Slider_Scale_ClassifierFreeGuidance_Input,
				#
				ComboBox_Sampler_Input,
				# TextBox_Sampler_Input,
				#
				Slider_Strength_Denoising_Input,
				ComboBox_Method_Filling_Input,
				# Slider_Blur_Mask_Input,
				Slider_Resolution_Inpainting_Input,
				TextBox_Directory_Output_Input,
				#
				# ComboBox_Address_Processor_Input,
				TextBox_Address_Processor_Input,
				#
				TextBox_Path_Configuration_Input,
				# TextBox_Prompt_Background_Negative_Input,
				# CheckBox_Resolution_Full_Inpainting_Input,
				# Slider_Padding_Resolution_Full_Inpainting_Input,
				#
				# ComboBox_Detector_Input,
				TextBox_Path_CheckPoint_Detector_Input,
				#
				CheckBox_Intermediate_Save_Input,
				# CheckBox_Infer_Tagger_Input,
				CheckBox_To_Grey_Input,

				CheckBox_Instance_PositiveFilm_Input,
			] ,
			outputs = \
			[
				Label_Progress_Instance_Generating,
				#
				# Step 1.
				ImageBox_Instances_Output,
				TextBox_Prompt_Positive_General_Input,
				TextBox_Prompt_Negative_General_Input,
				#
				# Step 2.
				ImageBox_Instances_1_Output,
				ImageBox_Instances_2_Output,
				ImageBox_Instances_3_Output,
				ImageBox_Instances_4_Output,
				ImageBox_Instances_5_Output,
				ImageBox_Instances_6_Output,
				ImageBox_Instances_7_Output,
				ImageBox_Instances_8_Output,
				ImageBox_Instances_9_Output,
				ImageBox_Instances_10_Output,
				#
				TextBox_Prompt_Foreground_Positive_1_Input ,
				TextBox_Prompt_Foreground_Positive_2_Input ,
				TextBox_Prompt_Foreground_Positive_3_Input ,
				TextBox_Prompt_Foreground_Positive_4_Input ,
				TextBox_Prompt_Foreground_Positive_5_Input ,
				TextBox_Prompt_Foreground_Positive_6_Input ,
				TextBox_Prompt_Foreground_Positive_7_Input ,
				TextBox_Prompt_Foreground_Positive_8_Input ,
				TextBox_Prompt_Foreground_Positive_9_Input ,
				TextBox_Prompt_Foreground_Positive_10_Input ,
				#
				#
				Row_Instance_1,
				Row_Instance_2,
				Row_Instance_3,
				Row_Instance_4,
				Row_Instance_5,
				Row_Instance_6,
				Row_Instance_7,
				Row_Instance_8,
				Row_Instance_9,
				Row_Instance_10,
			] ,
		)
		Button_Run_4.click\
		(
			fn=DoOnButton_Run_4Clicking,
			inputs = \
			[
				ComboBox_Method_Filling_Input,
				TextBox_Address_Processor_Input,
				Slider_Blur_Mask_Input,
				CheckBox_Resolution_Full_Inpainting_Input,
				Slider_Padding_Resolution_Full_Inpainting_Input,
				# CheckBox_Instance_PositiveFilm_Input,

				TextBox_Prompt_Positive_Input,
				TextBox_Prompt_Negative_Input,
				TextBox_Prompt_Background_Negative_Input,

				CheckBox_One_By_One_Input,
				CheckBox_Intermediate_Save_Input,



				# [
				TextBox_Prompt_Foreground_Positive_1_Input,
				TextBox_Prompt_Foreground_Positive_2_Input,
				TextBox_Prompt_Foreground_Positive_3_Input,
				TextBox_Prompt_Foreground_Positive_4_Input,
				TextBox_Prompt_Foreground_Positive_5_Input,
				TextBox_Prompt_Foreground_Positive_6_Input,
				TextBox_Prompt_Foreground_Positive_7_Input,
				TextBox_Prompt_Foreground_Positive_8_Input,
				TextBox_Prompt_Foreground_Positive_9_Input,
				TextBox_Prompt_Foreground_Positive_10_Input,
				# ],

				# # ImageBox_Original_Input,
				#
				# Slider_Width_Output_Input,
				# Slider_Height_Output_Input,
				# Slider_Step_Input,
				# Slider_Scale_ClassifierFreeGuidance_Input,
				# #
				# ComboBox_Sampler_Input,
				# # TextBox_Sampler_Input,
				# #
				# Slider_Strength_Denoising_Input,
				# ComboBox_Method_Filling_Input,
				# Slider_Blur_Mask_Input,
				# Slider_Resolution_Inpainting_Input,
				# TextBox_Directory_Output_Input,
				# #
				# # ComboBox_Address_Processor_Input,
				# TextBox_Address_Processor_Input,
				# #
				# TextBox_Path_Configuration_Input,
				# CheckBox_Resolution_Full_Inpainting_Input,
				# Slider_Padding_Resolution_Full_Inpainting_Input,
				# #
				# # ComboBox_Detector_Input,
				# TextBox_Path_CheckPoint_Detector_Input,
				# #
			] ,
			outputs = \
			[
				Label_Progress_Result_Generating,
				#
				# ImageBox_Imediate_Output,
				BatchedImageBox_Positive_Film_Output,
				BatchedImageBox_Mask_Output,
				#
				# Step 3.
				ImageBox_Output,
			] ,
		)
		ImageBox_Original_Input.change\
		(
			fn=DoOnImageBox_Input_OriginalChanged,
			inputs = \
			[
				ImageBox_Original_Input,
			] ,
			outputs = \
			[
				TextBox_Address_Processor_Input,

				Label_Progress_Instance_Generating,
				Label_Progress_Result_Generating,

				CheckBox_One_By_One_Input,
				CheckBox_Intermediate_Save_Input,

				BatchedImageBox_Positive_Film_Output,
				BatchedImageBox_Mask_Output,

				TextBox_Prompt_Positive_Input,
				TextBox_Prompt_Negative_Input,

				Slider_Width_Output_Input,
				Slider_Height_Output_Input,

				ImageBox_Imediate_Output,

				Slider_Step_Input,
				Slider_Scale_ClassifierFreeGuidance_Input,
				ComboBox_Sampler_Input,
				Slider_Strength_Denoising_Input,
				ComboBox_Method_Filling_Input,
				Slider_Blur_Mask_Input,
				Slider_Resolution_Inpainting_Input,
				TextBox_Directory_Output_Input,
				TextBox_Path_Configuration_Input,
				TextBox_Prompt_Background_Negative_Input,
				CheckBox_Resolution_Full_Inpainting_Input,
				Slider_Padding_Resolution_Full_Inpainting_Input,
				TextBox_Path_CheckPoint_Detector_Input,
				CheckBox_To_Grey_Input,
				CheckBox_Infer_Tagger_Input,
				ImageBox_Output,
				ImageBox_Output_Original_Contrast,
				ImageBox_Instances_Output,
				TextBox_Prompt_Positive_General_Input,
				TextBox_Prompt_Negative_General_Input,
				CheckBox_Instance_PositiveFilm_Input,
				#
				#
				ImageBox_Instances_1_Output,
				ImageBox_Instances_2_Output,
				ImageBox_Instances_3_Output,
				ImageBox_Instances_4_Output,
				ImageBox_Instances_5_Output,
				ImageBox_Instances_6_Output,
				ImageBox_Instances_7_Output,
				ImageBox_Instances_8_Output,
				ImageBox_Instances_9_Output,
				ImageBox_Instances_10_Output,
				#
				TextBox_Prompt_Foreground_Positive_1_Input,
				TextBox_Prompt_Foreground_Positive_2_Input,
				TextBox_Prompt_Foreground_Positive_3_Input,
				TextBox_Prompt_Foreground_Positive_4_Input,
				TextBox_Prompt_Foreground_Positive_5_Input,
				TextBox_Prompt_Foreground_Positive_6_Input,
				TextBox_Prompt_Foreground_Positive_7_Input,
				TextBox_Prompt_Foreground_Positive_8_Input,
				TextBox_Prompt_Foreground_Positive_9_Input,
				TextBox_Prompt_Foreground_Positive_10_Input,
				#
				Row_Instance_1,
				Row_Instance_2,
				Row_Instance_3,
				Row_Instance_4,
				Row_Instance_5,
				Row_Instance_6,
				Row_Instance_7,
				Row_Instance_8,
				Row_Instance_9,
				Row_Instance_10,
			] ,
		)

#endregion Interface

#region Evnet Handler

def DoOnButton_Run_2Clicking\
(
	# Image_In:Number.ndarray,
	# Image_In:Imaging,
	Path_Image_In:str,

	IsOneByOneInput_In:bool,
	# Prompt_Positive_In:str,
	# Prompt_Negative_In:str,
	# Width_Output_In:int,
	# Height_Output_In:int,
	Step_In:int,
	Scale_ClassifierFreeGuidance_In:int,
	#
	Name_Sampler_In:str,
	#
	Strength_Denoising_In:float,
	Method_Filling_In:str,
	# Blur_Mask_In:int,
	Resolution_Inpainting_In:int,
	Directory_Output_In:str,
	#
	Address_Processor_In:str,
	#
	Path_Configuration_In:str,
	# Prompt_Background_Negative_In:str,
	# IsInpaintingFullResolution_In:bool,
	# Padding_Resolution_Full_Inpainting_In:int,
	#
	Path_Detector_In:str,
	#
	IsSavingIntermediateImage_Required_In:bool,
	# IsInferringTagger_Required_In:bool,
	IsToGrey_Required_In:bool,

	IsInstanceDisplayed_Required_In:bool,
) :
	"""
	Argument List:
		Image_In→ImageBox_Original_Input,

		IsOneByOneInput_In→CheckBox_One_By_One_Input,
		Prompt_Positive_In→TextBox_Prompt_Positive_Input,
		Prompt_Negative_In→TextBox_Prompt_Negative_Input,
		Width_Output_In→Slider_Width_Output_Input,
		Height_Output_In→Slider_Height_Output_Input,
		Step_In→Slider_Step_Input,
		Scale_ClassifierFreeGuidance_In→Slider_Scale_ClassifierFreeGuidance_Input,
		#
		Name_Sampler_In→TextBox_Sampler_Input,
		#
		Strength_Denoising_In→Slider_Strength_Denoising_Input,
		Method_Filling_In→ComboBox_Method_Filling_Input,
		Blur_Mask_In→Slider_Blur_Mask_Input,
		Resolution_Inpainting_In→Slider_Resolution_Inpainting_Input,
		Directory_Output_In→TextBox_Directory_Output_Input,
		#
		Address_Processor_In→TextBox_Address_Processor_Input,
		#
		Path_Configuration_In→TextBox_Path_Configuration_Input,
		Prompt_Negative_Background_In→TextBox_Prompt_Negative_Background_Input,
		IsInpaintingFullResolution_In→CheckBox_Resolution_Full_Inpainting_Input,
		Padding_Resolution_Full_Inpainting_In→Slider_Padding_Resolution_Full_Inpainting_Input,
		#
		Path_Detector_In→TextBox_Path_CheckPoint_Detector_Input,
		#
		IsSavingIntermediateImage_Required_In→CheckBox_Intermediate_Save_Input,
		IsInferringTagger_Required_In→CheckBox_Infer_Tagger_Input,
		IsToGrey_Required_In→CheckBox_To_Grey_Input,

		IsInstanceDisplayed_Required_In→CheckBox_WhiteBackground_Input,
	"""

	if Path.exists ( Path_Configuration_In ) :
		pass
		# ！Pending：需要确认该Branch设计逻辑|目的
		# args = Configuration_Ω.OmegaConf.create ( vars ( args ) )
		# args.merge_with ( Configuration_Ω.OmegaConf.load ( args.cfg ) )
		# print ( args )
	else:
		pass

	if not Path.exists ( Directory_Output_In ) :
		OS.makedirs ( Directory_Output_In )
	else:
		pass

	Detector = Anime_Instance_Segmentation.AnimeInsSeg ( Path_Detector_In , device = F"cuda" )

	Detector.init_tagger ()


	# Processing: Generate Instance List from Image input
	Image_Intermediate_Array_Return = None
	List_PositiveFilm_Return = [ ]
	List_Mask_Return = [ ]
	List_PositiveFilm_Return = [ ]
	Prompt_Positive_Return = None
	Prompt_Negative_Return = None

	List_Prompt_Foreground_Positive=[]
	List_Prompt_Character_Positive=[]

	# ！改用Text Box直接输入
	# if Prompt_Positive_In :
	# 	Prompt_Positive_Input = F"{Prompt_Positive_Default}, {Prompt_Positive_In}"
	# else:
	Prompt_Positive_Input=Prompt_Positive_Default
	#
	# Prompt_Positive_Input = Prompt_Positive_In
	#
	# if Prompt_Negative_In :
	# 	Prompt_Negative_Input = F"{Prompt_Negative_Default}, {Prompt_Negative_In}"
	# else:
	Prompt_Negative_Input=Prompt_Negative_Default
	#
	# Prompt_Negative_Input = Prompt_Negative_In

	# Debug
	assert Method_Filling_In in List_Method_Inpainting_Fill , F"Fill method must be one of {List_Method_Inpainting_Fill}."

	# Processing: 计算长宽比、Image类型（横向|纵向）
	Limit_Resolution = Resolution_Inpainting_In

	# Debug
	print(F"In {Inspect.currentframe().f_code.co_name}()")
	print(F"{Path_Image_In=}" )

	Image_Processing = Imaging.open ( Path_Image_In )
	# Name_Image = Path.basename ( Path_Image_In )

	Width_Approximated,Height_Approximated=Calculate_Approximate_Size\
	(
		Size_In=Location_Point\
		(
			Image_Processing.width,
			Image_Processing.height,
		),
		Limit_Resolution_In=Limit_Resolution,
	)
	Size_Approximated=Location_Point(Width_Approximated,Height_Approximated)

	# Debug
	print(F"{Image_Processing.width=},{Image_Processing.height=}")
	print(F"Ratio: {Image_Processing.width/Image_Processing.height}")
	print(F"{Size_Approximated.Horizon=},{Size_Approximated.Vertical=}" )
	print(F"Ratio: {Size_Approximated.Horizon / Size_Approximated.Vertical}" )      # ！应该设置Ratio的Property，但这就会从Named Tuple转为Class，复杂了

	if IsToGrey_Required_In :
		Image_Processing=Image_Processing\
			.convert(F"L")\
			.convert(F"RGB")        # Contrast→RGB Coloured，增强对比度
	else:
		pass

	List_Option_Shared =\
	{
		F"width" : Size_Approximated.Horizon ,
		F"height" : Size_Approximated.Vertical ,
		F"steps" : Step_In ,
		F"cfg_scale" : Scale_ClassifierFreeGuidance_In ,
		F"sample_name" : Name_Sampler_In ,
		F"denoising_strength" : Strength_Denoising_In ,
		F"alwayson_scripts" :\
		{
			F"controlnet" :\
			{
				F"args" :\
				[
					{
						F"input_image" : F"" ,
						F"module" : F"lineart_anime" ,
						F"model" : Model_Default ,
						F"weight" : 1 ,
						F"resize_mode" : F"Inner Fit (Scale to Fit)" ,
						F"lowvram" : False ,
						F"processor_res" : Limit_Resolution ,
						F"threshold_a" : 64 ,
						F"threshold_b" : 64 ,
						F"guidance" : 1 ,
						F"guidance_start" : 0 ,
						F"guidance_end" : 1 ,
						F"guessmode" : False,
                        F"pixel_perfect": True,
					},
				],
			},
		} ,
	}

	Image_Processing = Image_Processing.resize\
	(
		size = (Size_Approximated.Horizon , Size_Approximated.Vertical) ,       # ！应该可以直接赋值：Named Tuple→Tuple
		resample = Imaging.Resampling.LANCZOS,
	)

	Instance = Detector.infer\
	(
		Path_Image_In ,
		output_type = F"numpy",
		infer_grey = IsToGrey_Required_In,
		infer_tags = True,
	)

	Instance.remove_duplicated ()

	# Processing: Generate Instances
	Image_Processing_Array=Number.array(Image_Processing )

	Image_Instances_Return = Instance.draw_instances \
	(
		img=Image_Processing_Array ,
		draw_bbox = True,
		draw_ins_mask = True,
		draw_ins_contour = True,
		# ！需要保持None，Empty不行
		# draw_indices =\
		# [
		# 	# List of indice
		# ],
		draw_tags = True ,
		mask_alpha=Mask_α_Default,
	)

	# # Debug
	# print ( f"{List_Tag_Character=}" )
	# print ( f"{List_Tag=}" )

	List_Mask = [ ]

	if not Instance.is_empty :
		for Carrier_Mask_Array in Instance.masks :
			Carrier_Mask_Array = Vision.resize\
			(
				Carrier_Mask_Array.astype ( Number.uint8 ) \
					* 255 ,
				(Size_Approximated.Horizon , Size_Approximated.Vertical) ,
				interpolation = Vision.INTER_AREA,
			)
			Carrier_Mask = Imaging.fromarray ( Carrier_Mask_Array )

			# Prst-Processing
			# Processing: Generate Instance in Positive Film or Mask
			Carrier_Image_Processing_Foreground_Mask_Array = Number.asarray ( Carrier_Mask )
			Casrrier_Image_Foreground = Generate_Foreground ( Image_Processing_Array , Carrier_Image_Processing_Foreground_Mask_Array , IsWhiteBackground_In = True )

			List_Mask.append ( Carrier_Mask )
			List_Mask_Return.append ( Carrier_Mask )
			List_PositiveFilm_Return.append(Casrrier_Image_Foreground )

		# Processing: Generate Positive Foreground Tag List of per Instance
		for Carrier_List_Tag_Foreground_Positive in Instance.tags :
			# List_Tag_Foreground_Positive = F"{Carrier_List_Tag_Foreground_Positive.replace ( F' ' , F',' )}, {Prompt_Positive_Input}"
			List_Tag_Foreground_Positive = F"{Carrier_List_Tag_Foreground_Positive.replace ( F' ' , F',' )}"

			List_Prompt_Foreground_Positive.append(List_Tag_Foreground_Positive)

		# ！Character Tag长期没什么用（有时有值，有时无值）|未使用，Positive Foreground | Character | Person | Instance均在.tags Property
		for Carrier_List_Tag_Character_Positive in Instance.character_tags :
			# List_Tag_Character_Positive = F"{Carrier_List_Tag_Character_Positive.replace ( F' ' , F',' )}, {Prompt_Positive_Input}"
			# List_Tag_Character_Positive = F"{Carrier_List_Tag_Character_Positive.replace ( F' ' , F',' )}"

			List_Prompt_Character_Positive.append(Carrier_List_Tag_Character_Positive)

		# Post-Processing
		# Debug
		print(F"{List_Prompt_Foreground_Positive=}")
		print(F"{List_Prompt_Character_Positive=}")
	else:
		pass



	# Post-Processing
	# Global Field Assigning
	global List_Option_Shared_Global
	List_Option_Shared_Global=List_Option_Shared
	global Instance_Global
	Instance_Global=Instance
	global List_Prompt_Foreground_Positive_Global
	List_Prompt_Foreground_Positive_Global=List_Prompt_Foreground_Positive
	global List_Mask_Global
	List_Mask_Global=List_Mask
	global Image_Processing_Global
	Image_Processing_Global=Image_Processing
	# global Prompt_Positive_Input_Global
	# Prompt_Positive_Input_Global=Prompt_Positive_Input
	# global Prompt_Negative_Input_Global
	# Prompt_Negative_Input_Global=Prompt_Negative_Input
	global Size_Approximated_Global
	Size_Approximated_Global=Size_Approximated
	global Detector_Global
	Detector_Global=Detector

	# Processing: Generate Return List
	List_Return=\
	[
		F"Style Editing executed successfully.",
		#
		# Step 1.
		Image_Instances_Return,
		#
		# ！实际不是Instance Genearting时产生的，只是为了呈现效果而添加的Global Default
		Prompt_Positive_Default,
		Prompt_Negative_Default,
		#
		# Step 2.
	]

	# Step 2.

	# Pre-Processing
	# Debug
	if len(List_Mask_Return) == len(List_Prompt_Foreground_Positive) == len(List_PositiveFilm_Return):
		print(F"Instances generated successfully, number is : {len(List_Mask_Return)}")
	else:
		print(F"Instances generated unsuccessfully, the number as follows:")
		print(F"{len(List_Mask_Return)=}")
		print(F"{len(List_Prompt_Foreground_Positive)=}")
		print(F"{len(List_PositiveFilm_Return)=}")

	# Processing: Generate Step 2 Instance List Result
	if IsInstanceDisplayed_Required_In :
		List_Return.extend(List_PositiveFilm_Return)

		# 补齐List到指定数量
		Number_Instance = len ( List_PositiveFilm_Return )
	else:
		List_Return.extend(List_Mask_Return)

		# 补齐List到指定数量
		Number_Instance=len(List_Mask_Return)

	while Number_Instance<Limit_Instance:
		List_Return.append ( None )

		Number_Instance+=1

	#

	# Processing: Generate Step 2 Prompt of Instance
	List_Return.extend(List_Prompt_Foreground_Positive)

	# 补齐List到指定数量
	Number_Instance=len(List_Prompt_Foreground_Positive)        # ！理论上和Mask List一致，需要使用统一的来源，并Constant化

	while Number_Instance<Limit_Instance:
		List_Return.append ( F"" )

		Number_Instance+=1

	#
	#

	# Processing: Update Step 2 Instance List Control

	# 补齐List到指定数量
	Number_Instance=len(List_Mask_Return)
	Index=0

	while Index<Number_Instance:
		List_Return.append ( Framework.update(visible=True) )

		Index+=1

	while Number_Instance<Limit_Instance:
		List_Return.append ( Framework.update(visible=False) )

		Number_Instance+=1



	return List_Return

def DoOnImageBox_Input_OriginalChanged\
(
	Image_In:Number.ndarray,
) :
	"""
	Clear all results previous while new processing starting.

	Args:
		Image_In ():

	Returns:
	"""

	# ！需要提取为新的函数、Constant
	List_Return = \
	[
		Address_Default_Model_Remote ,

		Progress_Default ,
		Progress_Default ,

		True ,
		True ,

		None ,
		None,

		Prompt_Positive_Default,
		Prompt_Negative_Default,

		768,
		768,

		None,

		24 ,
		9 ,
		Sampler_Default ,
		0.75 ,
		F"original" ,
		4 ,
		640,
		Directory_Output_Default,
		F"",
		Prompt_Background_Negative_Default,
		True,
		32,
		Detector_Default,
		False,
		True,
		None,
	]

	# Image Box as Contrast
	if(Image_In is None):
		List_Return.append(None)
	else:
		List_Return.append(Image_In)

	# Image Box Instance Output
	List_Return.append ( None )
	# Prompt Positive General
	List_Return.append ( F"" )
	# Prompt Negative General
	List_Return.append ( F"" )
	# Check Box Instance Positive Film
	List_Return.append ( True )

	#
	#

	# Reset Instances Result
	List_Return.append ( None )
	List_Return.append ( None )
	List_Return.append ( None )
	List_Return.append ( None )
	List_Return.append ( None )
	List_Return.append ( None )
	List_Return.append ( None )
	List_Return.append ( None )
	List_Return.append ( None )
	List_Return.append ( None )
	#
	List_Return.append ( F"" )
	List_Return.append ( F"" )
	List_Return.append ( F"" )
	List_Return.append ( F"" )
	List_Return.append ( F"" )
	List_Return.append ( F"" )
	List_Return.append ( F"" )
	List_Return.append ( F"" )
	List_Return.append ( F"" )
	List_Return.append ( F"" )
	#
	List_Return.append ( Framework.update(visible=True) )
	List_Return.append ( Framework.update(visible=False) )
	List_Return.append ( Framework.update(visible=False) )
	List_Return.append ( Framework.update(visible=False) )
	List_Return.append ( Framework.update(visible=False) )
	List_Return.append ( Framework.update(visible=False) )
	List_Return.append ( Framework.update(visible=False) )
	List_Return.append ( Framework.update(visible=False) )
	List_Return.append ( Framework.update(visible=False) )
	List_Return.append ( Framework.update(visible=False) )

	# # Debug
	# print ( F"After generating Down Scaled Image in {Inspect.currentframe ().f_code.co_name}()" )
	# print ( F"{Width_Global=}" )
	# print ( F"{Height_Global=}" )
	# print ( F"Ratio: {Width_Global / Height_Global}" )
	# print ( F"Ratio Global | Original as contrast: {Ratio_Global=}" )

	return List_Return

def DoOnButton_Run_4Clicking\
(
	Method_Filling_In,
	Address_Processor_In,
	Blur_Mask_In,
	IsInpaintingFullResolution_In,
	Padding_Resolution_Full_Inpainting_In,
	# IsWhiteBackground_In,

	Prompt_Positive_In,
	Prompt_Negative_In,
	Prompt_Background_Negative_In,

	IsOneByOneInput_In,
	IsSavingIntermediateImage_Required_In,



	# List_Prompt_Positive_In,      # ！考虑简化入参
	Prompt_Positive_1_In:str,
	Prompt_Positive_2_In:str,
	Prompt_Positive_3_In:str,
	Prompt_Positive_4_In:str,
	Prompt_Positive_5_In:str,
	Prompt_Positive_6_In:str,
	Prompt_Positive_7_In:str,
	Prompt_Positive_8_In:str,
	Prompt_Positive_9_In:str,
	Prompt_Positive_10_In:str,

	# Resolution_Inpainting_In,
	# Step_In,
	# Scale_ClassifierFreeGuidance_In,
	# Name_Sampler_In,
	# Strength_Denoising_In,
	# Directory_Output_In,
	# Detector_In,
	# Path_Image_In ,
	# IsToGrey_Required_In,
) :
	"""
	As known as Repaint_Image()
	"""

	List_PositiveFilm_Return = [ ]
	List_Mask_Return = [ ]

	# Global Field Assigning
	global List_Option_Shared_Global
	global Instance_Global
	global List_Mask_Global
	global Image_Processing_Global
	# global Prompt_Positive_Input_Global
	# global Prompt_Negative_Input_Global
	global Size_Approximated_Global
	global Detector_Global
	#
	Object_Instance=Instance_Global
	List_Mask=List_Mask_Global
	Image_Processing=Image_Processing_Global
	Prompt_Positive_Input=Prompt_Positive_In
	Prompt_Negative_Input=Prompt_Negative_In
	List_Option_Shared=List_Option_Shared_Global
	Size_Approximated=Size_Approximated_Global
	Detector=Detector_Global

	# Pre-Processing
	if Object_Instance.is_empty :
		return
	else:
		pass

	# ！后面的引用未考虑Index越界（Item存在，但是值是None）
	List_Prompt_Foreground_Positive=\
	[
		Prompt_Positive_1_In,
		Prompt_Positive_2_In,
		Prompt_Positive_3_In,
		Prompt_Positive_4_In,
		Prompt_Positive_5_In,
		Prompt_Positive_6_In,
		Prompt_Positive_7_In,
		Prompt_Positive_8_In,
		Prompt_Positive_9_In,
		Prompt_Positive_10_In,
	]

	# Debug
	for Index,Carrier_Prompt in enumerate(List_Prompt_Foreground_Positive):
		print(F"{Index}-th (0-Indexed) , {Carrier_Prompt=}")

	# Processing: Edit Style in Bundle
	if not IsOneByOneInput_In :     # 实际不会执行
		# detected tags got mixed during the rendering processing
		# imgbgr = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
		# tags, character_tags = detector.tagger.label_cv2_bgr(imgbgr)
		Prompt_Foreground_Positive = F"{Prompt_Positive_Input}, {Prompt_Foreground_Positive_Default}"
		Prompt_Foreground_Negative = Prompt_Negative_Input

		Image_Processing_Base64 = Utility.img2b64 ( Image_Processing )
		List_Option_Shared [ F"alwayson_scripts" ] [ F"controlnet" ] [ F"args" ] [ 0 ] [ F"input_image" ] = Image_Processing_Base64
		Content_Request = JSON.dumps\
		(
			{
				F"init_images" : [ Image_Processing_Base64 ] ,
				F"prompt" : Prompt_Foreground_Positive ,
				F"negative_prompt" : Prompt_Foreground_Negative ,
				**List_Option_Shared ,
			},
		)

		# Log
		print ( F"Runing Default CLDM..." )
		print ( f"{Prompt_Foreground_Positive=}" )
		print ( f"{Prompt_Foreground_Negative=}" )

		Stream_Responsed = Utility.submit_request ( Address_Processor_In , Content_Request )
		Image_Base64_Intermediate_Output = Stream_Responsed.json () [ F"images" ] [ 0 ]
		#
		# Utility.save_encoded_image\
		# (
		# 	Image_Base64_Intermediate_Output ,
		# 	Path.join\
		# 	(
		# 		Directory_Output_In ,
		# 		F"repaint-default-{Name_Image}.png",
		# 	),
		# )
		#
		# ！Pending
		Image_Intermediate_Array_Return= Convert_Image_From_Base64_To_NumpyArray ( Image_Base64_Intermediate_Output )

		return
	else:
		pass



	Mask_Foreground_Together_Array = Vision.resize\
	(
		Object_Instance\
			.compose_masks ()\
			.astype ( Number.uint8 ) \
			* 255 ,
		(Size_Approximated.Horizon , Size_Approximated.Vertical) ,
		interpolation = Vision.INTER_AREA,
	)

	# Processing: Re-paint Background
	if Mask_Foreground_Together_Array is not None :
		Image_Processing_Array=Number.array ( Image_Processing )
		Image_Background_Array = Inpainting.patch_match.inpaint\
		(
			Image_Processing_Array ,
			Mask_Foreground_Together_Array ,
			patch_size = 3,
		)

		# List_Tag_Background_Positive , List_Tag_Foreground = Detector_In.tagger.label_cv2_bgr ( Vision.cvtColor ( Image_Background_Array , Vision.COLOR_BGR2RGB ) )
		# List_Tag_Background_Positive , _ = Detector.tagger.label_cv2_bgr\
		# (
		# 	Vision.cvtColor\
		# 	(
		# 		Image_Background_Array ,
		# 		Vision.COLOR_BGR2RGB,
		# 	),
		# )
		List_Tag_Background_Positive=Generate_Tag_List(Image_Background_Array)

		# Debug
		# print ( f"{List_Tag_Character=}" )
		print ( f"{List_Tag_Background_Positive=}" )

		Prompt_Background_Positive = F"{F','.join ( List_Tag_Background_Positive )}, {Prompt_Positive_Input}".replace ( F"portrait" , F"" )
		Prompt_Background_Negative = F"{Prompt_Negative_Input}, {Prompt_Background_Negative_In}"

		Image_Background = Imaging.fromarray ( Image_Background_Array )
		Image_Background_Base64 = Utility.img2b64 ( Image_Background )

		List_Option_Shared [ F"alwayson_scripts" ] [ F"controlnet" ] [ F"args" ] [ 0 ] [ F"input_image" ] = Image_Background_Base64
		Content_Request = JSON.dumps\
		(
			{
				F"init_images" : [ Image_Background_Base64 ] ,
				F"prompt" : Prompt_Background_Positive ,
				F"negative_prompt" : Prompt_Background_Negative ,
				**List_Option_Shared ,
			}
		)

		# Log
		print ( F"Runing Background repainting..." )
		print ( F"{Prompt_Background_Positive=}" )
		print ( F"{Prompt_Background_Negative=}" )

		Stream_Responsed = Utility.submit_request ( Address_Processor_In , Content_Request )
		Image_Base64_Intermediate_Output = Stream_Responsed.json () [ F"images" ] [ 0 ]

		if IsSavingIntermediateImage_Required_In :
			# Utility.save_encoded_image\
			# (
			# 	Image_Base64_Intermediate_Output ,
			# 	Path.join\
			# 	(
			# 		Directory_Output_In ,
			# 		F"repaint-bg-{Name_Image}.png",
			# 	),
			# )
			#
			Image_Intermediate_Array_Return= Convert_Image_From_Base64_To_NumpyArray ( Image_Base64_Intermediate_Output )
		else:
			pass

		Image_Processing_Background = Imaging.open ( IO.BytesIO ( Base64.b64decode ( Image_Base64_Intermediate_Output ) ) )
		Image_Processing_Background_Repainted = Imaging.composite\
		(
			Image_Processing ,
			Image_Processing_Background ,
			Imaging.fromarray ( Mask_Foreground_Together_Array ),
		)
		# img_bg_repainted.save(osp.join(save_dir, 'repaint-bg-composed-' + imgname+'.png'))
		Image_Processing = Image_Processing_Background_Repainted
	else:
		pass



	Image_Processing_Base64 = None
	Image_Processing_Inpainted_Base64=None
	Number_Character = len ( List_Mask )

	# Foreground are numeral
	for Index , Carrier_Mask in enumerate ( List_Mask ) :
		if Image_Processing_Inpainted_Base64 is None :
			Image_Processing_Base64 = Utility.img2b64 ( Image_Processing )
		else:
			Image_Processing_Base64=Image_Processing_Inpainted_Base64

		Mask_Base64 = Utility.img2b64 ( Carrier_Mask )
		# List_Tag_Foreground_Positive = Object_Instance.tags [ Index ]
		# List_Tag_Foreground_Positive = List_Prompt_Foreground_Positive [ Index ]

		# # Debug
		# print ( F"The {Index}-th (0-Indexed) {List_Tag_Foreground_Positive=}" )

		# Prompt_Foreground_Positive = F"{List_Tag_Foreground_Positive.replace ( F' ' , F',' )}, {Prompt_Positive_Input}"
		Prompt_Foreground_Positive = F"{List_Prompt_Foreground_Positive [ Index ]}, {Prompt_Positive_Input}"
		Prompt_Foreground_Negative = Prompt_Negative_Input

		List_Option_Shared [ F"alwayson_scripts" ] [ F"controlnet" ] [ F"args" ] [ 0 ] [ F"input_image" ] = Image_Processing_Base64
		Content_Request = JSON.dumps\
		(
			{
				F"init_images" : [ Image_Processing_Base64 ] ,
				F"mask" : Mask_Base64 ,
				F"prompt" : Prompt_Foreground_Positive ,
				F"negative_prompt" : Prompt_Foreground_Negative ,
				F"mask_blur" : Blur_Mask_In ,
				F"inpainting_fill" : List_Method_Inpainting_Fill.index ( Method_Filling_In ) ,
				F"inpaint_full_res" : IsInpaintingFullResolution_In ,
				F"inpaint_full_res_padding" : Padding_Resolution_Full_Inpainting_In ,
				**List_Option_Shared,
			}
		)

		# Log
		print ( F"Runing Foreground Repainting... {Index + 1}/{Number_Character}" )
		print ( F"{Prompt_Foreground_Positive=}" )
		print ( F"{Prompt_Foreground_Negative=}" )

		Stream_Responsed = Utility.submit_request ( Address_Processor_In , Content_Request )
		Image_Processing_Inpainted_Base64 = Stream_Responsed.json () [ F"images" ] [ 0 ]

		# if\
		# 	IsSavingIntermediateImage_Required_In\
		# 	or Index == Number_Mask - 1\
		# :
		# 	Utility.save_encoded_image\
		# 	(
		# 		Image_Base64_Input ,
		# 		Path.join\
		# 		(
		# 			Directory_Output_In ,
		# 			F"repaint-fg{Index}-{Name_Image}.png",
		# 		),
		# 	)
		# else:
		# 	pass
		Image_Processing_Inpainted_Array=Convert_Image_From_Base64_To_NumpyArray(Image_Processing_Inpainted_Base64 )
		Image_Processing_Inpainted_Mask_Array=Number.asarray(Carrier_Mask)

		# Image_Foreground_Inpainted= Generate_Foreground ( Image_Processing_Inpainted_Array , Image_Processing_Inpainted_Mask_Array,IsWhiteBackground_In=IsWhiteBackground_In )
		# List_PositiveFilm_Return.append(Image_Foreground_Inpainted )

		List_PositiveFilm_Return.append(Image_Processing_Inpainted_Array )

		# Carrier_Mask_Array.save\
		# (
		# 	Path.join\
		# 	(
		# 		Directory_Output_In ,
		# 		F"repaint-fg{Index}-mask-{Name_Image}.png",
		# 	),
		# )
		# #
		# List_Mask_Return.append(Carrier_Mask )
		List_Mask_Return.append(Image_Processing_Inpainted_Mask_Array )

	# Post-Processing
	Image_Return=List_PositiveFilm_Return[-1 ]

	List_Return=\
	[
		F"Style Editing executed successfully.",
		#
		# Image_Intermediate_Array_Return,
		List_PositiveFilm_Return,
		List_Mask_Return,
		#
		# Step 3.
		Image_Return,
	]

	return List_Return

#endregion Event Handler

#region Core

# endregion Core

# region Utility

Create_CheckBox = lambda Name_In , Value_Default_In , Tip_In = None,IsVisible_In=True,Width_In=F"auto" : Framework.Checkbox \
(
	# choices = Style.List_Style () ,
	# value = 20,	 # 相当于default
	# multiselect = True,
	label = Name_In ,
	value = Value_Default_In ,
	info = Tip_In ,
	visible = IsVisible_In,
)

Create_Output_BatchedImageBox=lambda \
	Name_In=F"Output",\
	Type_In=F"numpy",\
	Number_Column=None,\
	Object_Fit_In=F"contain", \
	IsPreviewMode_In=False,\
	IsLabelDisplayed_In=True,\
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
	)

List_Sampler=lambda:\
[
	F"Euler a",
	F"Euler",
	F"LMS",
	F"Heun",
	F"DPM2",
	F"DPM2 a",
	F"DPM++ 2S a",
	F"DPM++ 2M",
	F"DPM++ SDE",
	F"DPM fast",
	F"DPM adaptive",
	F"LMS Karras",
	F"DPM2 Karras",
	F"DPM2 a Karras",
	F"DPM++ 2S a Karras",
	F"DPM++ 2M Karras",
	F"DPM++ SDE Karras",
	F"DDIM",
]

List_Filling_Method=lambda:List_Method_Inpainting_Fill

def List_Processor () :
	pass

def List_Detector () :
	pass

# ！此处为了抵消Style Editing代码中后续对Image进行的BGR→RGB，进行了1次转换
Convert_Image_From_Base64_To_NumpyArray = lambda img_b64 : \
	Vision.cvtColor\
	(
		Vision.imdecode\
		(
			Number.frombuffer\
			(
				Base64.b64decode(img_b64),
				dtype=Number.uint8,
			),
			flags=Vision.IMREAD_COLOR,
		) ,
		Vision.COLOR_RGB2BGR,
	)

def Approximate_To_Integer ( Input_In ) :

	# return Mathemastics.floor(Input_In)       # 比较合理的估算方式：保障坐标不会超出范围
	return int(Input_In)       # 是截尾法，是比较合理的估算方式：保障坐标不会超出范围，同时对负数值的处理符合预期（趋向〇，而非-∞）
	# return int(round(Input_In))     # AIS源码|调用的Python源码中主要的坐标的处理逻辑

def Calculate_Approximate_Size ( Size_In:Location_Point,Limit_Resolution_In ) :

	if Size_In.Vertical > Size_In.Horizon :       # 高度更大的Image：纵向图
		Width = Limit_Resolution_In
		#
		Height = Approximate_To_Integer ( (Size_In.Vertical / Size_In.Horizon * Limit_Resolution_In) // Granularity_Pixel * Granularity_Pixel )
	else :       # 宽度更大的Image：横向图
		Width = Approximate_To_Integer ( (Size_In.Horizon / Size_In.Vertical * Limit_Resolution_In) // Granularity_Pixel * Granularity_Pixel )
		#
		Height = Limit_Resolution_In

	return\
	[
		Width,
		Height,
	]

def Generate_Foreground ( Image_Array_In , Image_Mask_Array_In , IsWhiteBackground_In = False ) :

	Image_Array_Return=None

	# Pre-Processing: Generate Inversed Mask
	Image_Mask_Inversed_Array=Vision.bitwise_not(Image_Mask_Array_In )
	Image_White_Array= Number.ones_like(Image_Array_In ) * 255
	Image_Background_Array=Vision.bitwise_or\
	(
		Image_White_Array,
		Image_White_Array,
		mask = Image_Mask_Inversed_Array,
	)

	# Processing: Generate Black Background + Instance Inpainted
	Image_Crop_Array=Vision.bitwise_and\
	(
		Image_Array_In,
		Image_Array_In,
		mask = Image_Mask_Array_In,
	)
	Image_Result_Array=Vision.addWeighted\
	(
		Image_Crop_Array,
		1,
		Image_Background_Array,
		1,
		0,
	)

	# Post-Processing
	if IsWhiteBackground_In :
		Image_Array_Return=Image_Result_Array
	else:
		Image_Array_Return = Image_Crop_Array

	return Image_Array_Return

def Generate_Tag_List(Image_In):

	global Detector_Global

	if isinstance(Image_In, Imaging.Image):
		Image_Processing_Array=Number.asarray(Image_In)
	elif isinstance(Image_In, Number.ndarray):
		Image_Processing_Array=Image_In
	else:
		pass

	List_Tag_Positive ,List_Tag_Character_Positive  = Detector_Global.tagger.label_cv2_bgr \
	(
		Vision.cvtColor \
		(
			Image_Processing_Array ,
			Vision.COLOR_BGR2RGB ,
		) ,
	)

	# Debug
	print(F"{List_Tag_Positive=}")
	print(F"{List_Tag_Character_Positive=}")

	# Prompt_Positive_Return = F"{F','.join ( List_Tag_Positive )}"
	Prompt_Positive_Return = List_Tag_Positive

	return Prompt_Positive_Return

#endregion Utility

#endregion Method

#endregion Definition