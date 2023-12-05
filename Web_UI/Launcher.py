#region Script Pre-processing

# Test by Francis
import os as OS
#
print ( F"{OS.getcwd()=}" )

import os.path as Pathing
import sys as Environment
#
Directory_Current = Pathing.dirname ( Pathing.abspath ( __file__ ) )
Directory_Project = Pathing.dirname  ( Directory_Current )
print ( F"{Directory_Project=}" )
Environment.path.append ( Directory_Project )
OS.chdir(Directory_Project)
#
print ( F"{OS.getcwd()=}" )

#endregion Script Pre-processing



#region Import

#region Reference

# import torch as Engine
# import cv2 as Vision

#endregion Reference

#region Self

# import Web_UI.Component.Frame_Main as Page
import Component.Frame_Main as Page

#endregion Self

#endregion Import



#region Method

def Execute_Application (IsSharing_In = False,Port_In=1234) :

	Application = Page.Application ()

	Application.Launch \
	(
		IsSharing_In = IsSharing_In ,
		Port_In = Port_In ,
	)

#endregion Method

#region Entry Point

if __name__ == F"__main__" :
	# ！未解决Base | Root识别为General Web UI这个Directory的问题
	Execute_Application\
	(
		IsSharing_In = True,
	)
else :

	pass

#endregion Entry Point