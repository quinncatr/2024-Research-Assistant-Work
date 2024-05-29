import sys
import vpi
import numpy as np
#Must install PIL from command line (sudo pip3 install pillow)
from PIL import Image
from argparse import ArgumentParser

# Parse command line arguments
parser = ArgumentParser()
parser.add_argument('backend', choices=['cpu','cuda','vic'],
                    help='Backend to be used for processing')

parser.add_argument('input',
                    help='Image to be used as input')

args = parser.parse_args();

if args.backend == 'cpu':
    backend = vpi.Backend.CPU
elif args.backend == 'cuda':
    backend = vpi.Backend.CUDA
else:
    assert args.backend == 'vic'
    backend = vpi.Backend.VIC

# Load input into a vpi.Image
input = vpi.asimage(np.asarray(Image.open(args.input)))

# Using the chosen backend,
with backend:
    # First convert input to NV12_ER.
    # We're overriding the default backend with CUDA.
    temp = input.convert(vpi.Format.NV12_ER, backend=vpi.Backend.CUDA)

    # Rescale the image using the chosen backend
    temp = temp.rescale((input.width//2, input.height//3))

    # Convert result back to input's format
    output = temp.convert(input.format, backend=vpi.Backend.CUDA)

# Save result to disk
Image.fromarray(output.cpu()).save('scaled_python'+str(sys.version_info[0])+'_'+args.backend+'.png')

# vim: ts=8:sw=4:sts=4:et:ai