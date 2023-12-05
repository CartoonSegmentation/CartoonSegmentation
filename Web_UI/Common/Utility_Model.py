import os.path as Pathing
import os as OS

import torch
from diffusers import StableDiffusionPipeline as Pipeline

import torch as Engine

# ！需要完全地面向对象传值
# ·Stable Diffusion Web UI使用ldm Library读取Model，超出本项目的需求，直接恢复PyTorch直读

class Model:
	# Field
	ListModel = [ ]
	Model_Activated=None
	# ！需要转为Private
	Directory_Diffusion = F"./Reference/Model/Diffusion"


	# Method

	# 拉取相对路径
	@staticmethod
	def List_CheckPointTile () :
		if Model.ListModel is True :
			return Model.Refresh ()
		else :
			return Model.Initialise ()


	# Core
	@staticmethod
	def IsCorrectModelType ( File , Filter = None ) :
		if (Filter is None) :
			return File.endswith ( F".vae." ) == False  # 需要调整
		else :
			return True  # 暂定


	@staticmethod
	def Initialise () :
		# Path_Processing = Directory.join ( OS.getcwd ().replace ( "\\" , "/" ) , Model.Directory_Diffusion )  # ！需要更优雅地跨平台
		Path_Processing = Model.Directory_Diffusion
		List_Return = Model.ListModel = [ File for File in OS.listdir ( Path_Processing ) if Model.IsCorrectModelType ( File ) ]        # Only scan the 1st level folder as item

		return List_Return


	@staticmethod
	def Refresh () :
		pass


	# ？我记得是等号不是冒号+等号
	# ？Diffusion能否直接使用PyTorch操作，不能的话需要转为Diffusers Library的Pipeline进行Instantiate
	# Load_Model:=lambda Model_Input :Model_Activated=Engine.load(Directory_Diffusion)
	@staticmethod
	def Load_Model ( Model_Input , Variant_In = "fp16" , Type_In = torch.float16 ) :
		# Path_Relative=OS.path.join(Directory_Diffusion,Model_Input)
		# Path_Relative=F"{Directory_Diffusion}/{Model_Input}"
		Path_Relative = F"{Model.Directory_Diffusion}/{Model_Input}"
		Model_Return:str=str()

		try:
			Model_Return=Pipeline.from_pretrained(Path_Relative,variant = Variant_In,torch_dtype = Type_In)
			# Model_Return = Pipeline.from_pretrained ( Path_Relative )
		except Exception:       # ！需要扩充|替换
			# ！需要增加try来解决不合格Model的剔除，或转为其他方式调用（如原生PyTorch）
			print(F"Error occurred: {Exception}")

			return Model_Return     # ！需要去除冗余
		else:
			pass

		Model_Return.to ( "cuda" )  # ！需要提取常量|变量

		# Debug
		print ( F"Model {Model_Input} has been loaded successfully." )      # ！需要转为Model的自身描述信息

		return Model_Return