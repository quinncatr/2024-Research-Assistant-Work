import sys
import vpi
import numpy as np
from PIL import Image
from argparse import ArgumentParser
from jtop import jtop
from timeit import default_timer as timer

#with jtop() as jetson:
    #print(jetson.stats)

# Parse command line arguments
parser = ArgumentParser()

parser.add_argument('input',
                    help='Image to be used as input')

args = parser.parse_args();

# Load input into a vpi.Image
input = vpi.asimage(np.asarray(Image.open(args.input)))

# Using the chosen backend,
with vpi.Backend.VIC:
    start = timer()
    # First convert input to NV12_ER.
    # We're overriding the default backend with CUDA.
    temp = input.convert(vpi.Format.NV12_ER, backend=vpi.Backend.CUDA)

    # Rescale the image using the chosen backend
    temp = temp.rescale((input.width//2, input.height//3))

    # Convert result back to input's format
    output = temp.convert(input.format, backend=vpi.Backend.CUDA)
    end = timer()
primaryTime = end - start

# Save result to disk
Image.fromarray(output.cpu()).save('scaled_python_VIC'+'.jpeg')

print("Rescaling using VIC was completed in " + str(end - start) + " seconds.\n")

with vpi.Backend.CPU:
    start = timer()
    temp = input.convert(vpi.Format.NV12_ER, backend=vpi.Backend.CPU)

    temp = temp.rescale((input.width//2, input.height//3))

    output = temp.convert(input.format, backend=vpi.Backend.CPU)
    end = timer()
primaryTime = end - start

Image.fromarray(output.cpu()).save('scaled_python_CPU'+'.jpeg')

print("Rescaling using CPU was completed in " + str(end - start) + " seconds.\n")

with vpi.Backend.CUDA:
    start = timer()
    temp = input.convert(vpi.Format.NV12_ER, backend=vpi.Backend.CUDA)

    temp = temp.rescale((input.width//2, input.height//3))

    output = temp.convert(input.format, backend=vpi.Backend.CUDA)
    end = timer()
primaryTime = end - start

Image.fromarray(output.cpu()).save('scaled_python_CUDA'+'.jpeg')

print("Rescaling using CUDA was completed in " + str(end - start) + " seconds.\n")

# vim: ts=8:sw=4:sts=4:et:ai