#!/usr/bin/env python

import base64
import argparse
import cv2
import flask
import getopt
import gevent
import gevent.pywsgi
import glob
import h5py
import io
import math
import moviepy
import moviepy.editor
import numpy as np
import os
import random
import re
import scipy
import scipy.io
import shutil
import sys
import tempfile
import time
import torch
import torchvision
import urllib
import zipfile
from anime_3dkenburns import KenBurnsPipeline, npyframes2video, KenBurnsConfig

##########################################################

torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance
torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

KCFG = {}
KPIPE: KenBurnsPipeline = None

objPlayback = {
    'strImage': None,
    'npyImage': None,
    'strMode': 'automatic',
    'intTime': 0,
    'fltTime': np.linspace(0.0, 1.0, 75).tolist() + list(reversed(np.linspace(0.0, 1.0, 75).tolist())),
    'strCache': {},
    'objFrom': {
        'fltCenterU': 512.0,
        'fltCenterV': 384.0,
        'intCropWidth': 1024,
        'intCropHeight': 768
    },
    'objTo': {
        'fltCenterU': 512.0,
        'fltCenterV': 384.0,
        'intCropWidth': 1024,
        'intCropHeight': 768
    }
}

objFlask = flask.Flask(import_name=__name__, static_url_path='', static_folder=os.path.abspath('./'))

@objFlask.route(rule='/', methods=[ 'GET' ])
def index():
    return objFlask.send_static_file('naive_interface.html')
# end

@objFlask.route(rule='/load_image', methods=[ 'POST' ])
def load_image():
    global KCFG

    objPlayback['strImage'] = flask.request.form['strFile']
    objPlayback['npyImage'] = np.ascontiguousarray(cv2.imdecode(buf=np.frombuffer(base64.b64decode(flask.request.form['strData'].split(';base64,')[1]), np.uint8), flags=-1)[:, :, 0:3])
    objPlayback['strCache'] = {}

    KCFG = KPIPE.generate_kenburns_config(objPlayback['npyImage'], verbose=args.verbose)
    return ''
# end

# @objFlask.route(rule='/autozoom', methods=[ 'POST' ])
# def autozoom():
#     objPlayback['objFrom'] = {
#         'fltCenterU': 512.0,
#         'fltCenterV': 384.0,
#         'intCropWidth': 1000,
#         'intCropHeight': 750
#     }

#     objPlayback['objTo'] = process_autozoom({
#         'fltShift': 100.0,
#         'fltZoom': 1.25,
#         'objFrom': objPlayback['objFrom']
#     }, KCFG)

#     return flask.jsonify({
#         'objFrom': objPlayback['objFrom'],
#         'objTo': objPlayback['objTo']
#     })
# # end

@objFlask.route(rule='/update_mode', methods=[ 'POST' ])
def update_mode():
    objPlayback['strMode'] = flask.request.form['strMode']

    return ''
# end

@objFlask.route(rule='/update_from', methods=[ 'POST' ])
def update_from():
    objPlayback['intTime'] = objPlayback['fltTime'].index(0.0)
    objPlayback['strCache'] = {}
    objPlayback['objFrom']['fltCenterU'] = float(flask.request.form['fltCenterU'])
    objPlayback['objFrom']['fltCenterV'] = float(flask.request.form['fltCenterV'])
    objPlayback['objFrom']['intCropWidth'] = int(flask.request.form['intCropWidth'])
    objPlayback['objFrom']['intCropHeight'] = int(flask.request.form['intCropHeight'])

    return ''
# end

@objFlask.route(rule='/update_to', methods=[ 'POST' ])
def update_to():
    objPlayback['intTime'] = objPlayback['fltTime'].index(1.0)
    objPlayback['strCache'] = {}
    objPlayback['objTo']['fltCenterU'] = float(flask.request.form['fltCenterU'])
    objPlayback['objTo']['fltCenterV'] = float(flask.request.form['fltCenterV'])
    objPlayback['objTo']['intCropWidth'] = int(flask.request.form['intCropWidth'])
    objPlayback['objTo']['intCropHeight'] = int(flask.request.form['intCropHeight'])

    return ''
# end

@objFlask.route(rule='/get_live', methods=[ 'GET' ])
def get_live():
    def generator():
        fltFramelimiter = 0.0

        while True:
            for intYield in range(100): gevent.sleep(0.0)

            gevent.sleep(max(0.0, (1.0 / 25.0) - (time.time() - fltFramelimiter))); fltFramelimiter = time.time()

            if objPlayback['strImage'] is None:
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + cv2.imencode(ext='.jpg', img=np.ones([ 768, 1024, 3 ], np.uint8) * 29, params=[ cv2.IMWRITE_JPEG_QUALITY, 80 ])[1].tobytes() + b'\r\n'; continue
            # end

            if objPlayback['intTime'] > len(objPlayback['fltTime']) - 1:
                objPlayback['intTime'] = 0
            # end

            intTime = objPlayback['intTime']
            fltTime = objPlayback['fltTime'][intTime]

            if objPlayback['strMode'] == 'automatic':
                objPlayback['intTime'] += 1
            # end

            if str(fltTime) not in objPlayback['strCache']:
                # Debug by Francis
                npyKenburns,_ = KPIPE.process_kenburns({
                    'fltSteps': [ fltTime ],
                    'objFrom': objPlayback['objFrom'],
                    'objTo': objPlayback['objTo'],
                    'boolInpaint': False
                }, KCFG, inpaint=False)[0]

                objPlayback['strCache'][str(fltTime)] = b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + cv2.imencode(ext='.jpg', img=npyKenburns, params=[ cv2.IMWRITE_JPEG_QUALITY, 80 ])[1].tobytes() + b'\r\n'
            # end

            yield objPlayback['strCache'][str(fltTime)]
        # end
    # end

    return flask.Response(response=generator(), mimetype='multipart/x-mixed-replace; boundary=frame')
# end

@objFlask.route(rule='/get_result', methods=[ 'GET' ])
def get_result():
    # strTempdir = tempfile.gettempdir() + '/kenburns-' + str(os.getpid()) + '-' + str.join('', [ random.choice('abcdefghijklmnopqrstuvwxyz0123456789') for intCount in range(8) ]) + '-' + str(time.time()).split('.')[-1]

    # os.makedirs(name=strTempdir + '/', exist_ok=False)
    print('###########################################')
    print(objPlayback['objFrom'])
    print(objPlayback['objTo'])

    # Debug by Francis
    npyKenburns,_ = KPIPE.process_kenburns({
        'fltSteps': np.linspace(0.0, 1.0, KCFG.num_frame).tolist(),
        'objFrom': objPlayback['objFrom'],
        'objTo': objPlayback['objTo'],
        'boolInpaint': True
    }, KCFG)

    moviepy.editor.ImageSequenceClip(sequence=[ npyFrame[:, :, ::-1] for npyFrame in npyKenburns + list(reversed(npyKenburns))[1:-1] ], fps=25).write_videofile('interface_kenburns.mp4', preset='placebo')


    return ''
    # objKenburns = io.BytesIO(open(strTempdir + '/kenburns.mp4', 'rb').read())

    # shutil.rmtree(strTempdir + '/')

    # return flask.send_file(path_or_file=objKenburns, mimetype='video/mp4', as_attachment=True, download_name='kenburns.mp4')



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Anime Character Instance Segmentation')
    
    parser.add_argument('--cfg', type=str, default=None, help='KenBurns config file path')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    KPIPE = KenBurnsPipeline(args.cfg)
    print(f'running on http://localhost:8080')
    gevent.pywsgi.WSGIServer(listener=('0.0.0.0', 8080), application=objFlask).serve_forever()
