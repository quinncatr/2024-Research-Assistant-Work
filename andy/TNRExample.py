import cv2
import sys
import vpi
import numpy as np
from argparse import ArgumentParser

# ----------------------------
# Parse command line arguments

parser = ArgumentParser()
parser.add_argument('backend', choices=['cuda','vic'],
                    help='Backend to be used for processing')

parser.add_argument('input',
                    help='Input video to be denoised')

args = parser.parse_args();

if args.backend == 'cuda':
    backend = vpi.Backend.CUDA
else:
    assert args.backend == 'vic'
    backend = vpi.Backend.VIC

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

outVideo = cv2.VideoWriter('denoised_python'+str(sys.version_info[0])+'_'+args.backend+extOutputVideo,
                            fourcc, fps, inSize)

#--------------------------------------------------------------
# Create the TNR object using the backend specified by the user
with backend:
    tnr = vpi.TemporalNoiseReduction(inSize, vpi.Format.NV12_ER)

#--------------------------------------------------------------
# Main processing loop
curFrame = 0
while True:
    curFrame+=1
    print("Frame: {}".format(curFrame))

    # Read one input frame
    ret, cvFrame = inVideo.read()
    if not ret:
        break

    # Convert it to NV12_ER format to be used by VPI
    with vpi.Backend.CUDA:
        frame = vpi.asimage(cvFrame).convert(vpi.Format.NV12_ER)

    # Retrieve the corresponding denoised frame
    denoised = tnr(frame, preset=vpi.TNRPreset.INDOOR_MEDIUM_LIGHT, strength=1)

    # Convert it to RGB8 for output using the CUDA backend
    with vpi.Backend.CUDA:
        denoised = denoised.convert(vpi.Format.RGB8)

    # Write the denoised frame to the output video
    with denoised.rlock():
        outVideo.write(denoised.cpu())

# vim: ts=8:sw=4:sts=4:et:ai