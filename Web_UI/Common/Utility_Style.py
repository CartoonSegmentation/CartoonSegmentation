import os.path as Pathing
import os as OS

# ·！未实现

# Field
ListStyle = [ ]

# Method

# 拉取绝对路径
def List_Style () :
	if ListStyle is not None :
		Refresh ()
	else :
		Initialise ()

# Core
def IsCorrectStyleType ( File , Filter = None ) :
	if (Filter is None) :
		return True  # 暂定
	else :
		return True  # 暂定

def Initialise () :
	Path_Processing = Pathing.join ( OS.getcwd ().replace ( "\\" , "/" ) , F"./Reference/Model/Diffusion" )
	List_Return = [ File for File in OS.listdir ( Path_Processing ) if IsCorrectStyleType ( File ) ]

	return List_Return

def Refresh () :
	pass