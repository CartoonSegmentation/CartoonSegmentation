#region Script Pre-processing

# Test by Francis
import os as OS
#
print ( F"{OS.getcwd()=}" )

# import os.path as Pathing
# import sys as Environment
# #
# Directory_Current = Pathing.dirname ( Pathing.abspath ( __file__ ) )
# Directory_Project = Pathing.dirname(Pathing.dirname  ( Directory_Current ))
# print ( F"{Directory_Project=}" )
# Environment.path.append ( Directory_Project )
# OS.chdir(Directory_Project)
# #
# print ( F"{OS.getcwd()=}" )

#endregion Script Pre-processing



import gradio as Framework


import Web_UI.Component.Component_AIS as Control_AIS
import Web_UI.Component.Component_Style_Editing as Control_StyleEditing


# ·当前的Gradio有如下限制，导致必须将UI相关的定义Inline放置：
# -在Block中直接创建Control的Instance，否则不显示
# -强制Control的Event Binding在Block中
# ·探索是否具有Class点出Member的方式简化各个COntrol的Variable的调用+区分

# Class
class Application :
	def Launch ( self , IsSharing_In = False , Address_In = F"0.0.0.0" , Port_In = 7860 ) :       # ？或更名为Create_UI()，与其他部分的Initialising一起调用的部分叫Launch()
																																						# ！需要COnstant化

		# Debug
		print ( F"Gradio based Web Site has been launched successfully, on https://{Address_In}:{Port_In}" )

		Title=F"Web UI-Anime Instance Segmentation"     #！需要修改一致

		# Draw basic Frame Page
		Framework.Markdown ( Title )      # Title
		Framework.Markdown ( F"" )      # Version
		Framework.Markdown ( F"" )      # Description

		with Framework.Blocks\
		(
			css = F"file=Resource/Style/Style-Genral.css",
			# _js = F"file=Resource/Function/Function-Genral.js",
		) as Page :
			Content_Page = self.Create_Content_Page ()

		# return Page.launch\
		return Page.queue ()\
			.launch\
			(
				share = IsSharing_In,
				server_name =Address_In,
				server_port = Port_In,
			)


	# ！布局为主的部分虽然用函数的形式分离了，但感觉还是应该在Frame的File中

	def Create_Content_Page (self) :
		"""
		Tabs of Anime Instance Segementation, created by 林 健

		Returns:
		"""

		Framework.Markdown(F"<h1>Instance-guided Cartoon Editing</ h1>")

		# 2 Ways: 1st separate Functions into Event Handler; 2nd just read Temporary Result
		# Now, only 2nd Way succeeded
		Tab_Demonstration_Steping = Control_AIS.Create_Stepping_Demonstration_Tab ()

		# Pending
		# Tab_Demonstration_Original = Control_AIS.Create_Original_Demonstration_Tab ()

		# Style Editing
		Tab_Demonstration_StyleEditing = Control_StyleEditing.Create_Style_Editing_Demonstration_Tab ()

		# Tab_Demonstration_Migrated = Create_Demonstration_Tab ()

		# Tab_Information = Create_Information_Tab ()
		# Tab_Setting = Control_AIS.Create_Setting_Tab ()

		# Bind Event handler