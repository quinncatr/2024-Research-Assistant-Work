import cv2
import sys
import vpi
import numpy as np
from math import sin, cos, pi
from argparse import ArgumentParser

#python3 PerspectiveWarpBenchmarking.py 
# ../../../../opt/nvidia/vpi1/samples/assets/noisy.mp4
def parseArgs():
    global args
    # ----------------------------
    # Parse command line arguments

    parser = ArgumentParser()
    #parser.add_argument('backend', choices=['cpu', 'cuda','vic'],
    #                    help='Backend to be used for processing')

    parser.add_argument('input',
                        help='Input video to be denoised')

    args = parser.parse_args();

def loadVideo(backend):
    global inVideo, outVideo
    # -----------------------------
    # Open input and output videos

    inVideo = cv2.VideoCapture(args.input)

    if int(cv2.__version__.split('.')[0]) >= 3:
        extOutputVideo = '.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        inSize = (int(inVideo.get(cv2.CAP_PROP_FRAME_WIDTH)), int(inVideo.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps = inVideo.get(cv2.CAP_PROP_FPS)
    else:
        # MP4 support with OpenCV-2.4 has issues, we'll use
        # avi/mpeg instead.
        extOutputVideo = '.avi'
        fourcc = cv2.cv.CV_FOURCC('M','P','E','G')
        inSize = (int(inVideo.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)), int(inVideo.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))
        fps = inVideo.get(cv2.cv.CV_CAP_PROP_FPS)

    outVideo = cv2.VideoWriter('perspwarp_python'+str(sys.version_info[0])+'_'+str(args.backend)+extOutputVideo,
                                fourcc, fps, inSize)
    

def loadCudaVideo():
    args.backend = vpi.Backend.CUDA
    backend = args.backend
    loadVideo(backend)

def loadCpuVideo():
    args.backend = vpi.Backend.CPU
    backend = args.backend
    loadVideo(backend)

def loadVicVideo():
    args.backend = cpi.Backend.VIC
    backend = args.backend
    loadVideo(backend)

def runCUDA():
    loadCudaVideo()

    #--------------------------------------------------------------
    # Main processing loop
    curFrame = 1
    while True:
        #print("Frame: {}".format(curFrame))
        curFrame+=1

        # Read one input frame
        ret, cvFrame = inVideo.read()
        if not ret:
            break

        # Convert it to NV12_ER format to be used by VPI
        with vpi.Backend.CUDA:
            frame = vpi.asimage(cvFrame).convert(vpi.Format.NV12_ER)

        # Calculate the transformation to be applied ------------

        # Move image's center to origin of coordinate system
        T1 = np.array([[1, 0, -frame.width/2.0],
                        [0, 1, -frame.height/2.0],
                        [0, 0, 1]])

        # Apply some time-dependent perspective transform
        v1 = sin(curFrame/30.0*2*pi/2)*0.0005
        v2 = cos(curFrame/30.0*2*pi/3)*0.0005
        P = np.array([[0.66, 0, 0],
                    [0, 0.66, 0],
                    [v1, v2, 1]])

        # Move image's center back to where it was
        T2 = np.array([[1, 0, frame.width/2.0],
                        [0, 1, frame.height/2.0],
                        [0, 0, 1]])

        # Do perspective warp using the backend passed in the command line.
        with vpi.Backend.CUDA:
            frame = frame.perspwarp(np.matmul(T2, np.matmul(P, T1)))

        # Convert it to RGB8 for output using the CUDA backend
        with vpi.Backend.CUDA:
            frame = frame.convert(vpi.Format.RGB8)

        # Write the denoised frame to the output video
        with frame.rlock():
            outVideo.write(frame.cpu())



parseArgs()
runCUDA()

# vim: ts=8:sw=4:sts=4:et:ai
