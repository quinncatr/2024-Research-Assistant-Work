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
vicTime = end - start

# Save result to disk
Image.fromarray(output.cpu()).save('scaled_python_VIC'+'.jpeg')

print("\nRescaling using VIC was completed in " + str(vicTime) + " seconds.\n")

with vpi.Backend.CPU:
    start = timer()
    temp = input.convert(vpi.Format.NV12_ER, backend=vpi.Backend.CPU)

    temp = temp.rescale((input.width//2, input.height//3))

    output = temp.convert(input.format, backend=vpi.Backend.CPU)
    end = timer()
cpuTime = end - start

Image.fromarray(output.cpu()).save('scaled_python_CPU'+'.jpeg')

print("Rescaling using CPU was completed in " + str(cpuTime) + " seconds.\n")

with vpi.Backend.CUDA:
    start = timer()
    temp = input.convert(vpi.Format.NV12_ER, backend=vpi.Backend.CUDA)

    temp = temp.rescale((input.width//2, input.height//3))

    output = temp.convert(input.format, backend=vpi.Backend.CUDA)
    end = timer()
cudaTime = end - start

Image.fromarray(output.cpu()).save('scaled_python_CUDA'+'.jpeg')

print("Rescaling using CUDA was completed in " + str(cudaTime) + " seconds.\n")

#TODO: add more print statments about differences in seconds and percent

print("-----Comparisons-----\n")

vicCpuTimeDiff = abs(vicTime - cpuTime)
vicCudaTimeDiff = abs(vicTime - cudaTime)
cpuCudaTimeDiff = abs(cpuTime - cudaTime)

vicCpuPercentDiff = (vicCpuTimeDiff / ((vicTime + cpuTime) / 2)) * 100
vicCudaPercentDiff = (vicCudaTimeDiff / ((vicTime + cudaTime) / 2)) * 100
cpuCudaPercentDiff = (cpuCudaTimeDiff / ((cudaTime + cpuTime) / 2)) * 100

print("Rescaling using VIC is " + str(vicCpuTimeDiff) + " seconds (" + str(vicCpuPercentDiff) + "%) slower than using the CPU.")
print("Rescaling using VIC is " + str(vicCudaTimeDiff) + " seconds (" + str(vicCudaPercentDiff) + "%) slower than using CUDA.")
print("Rescaling using the CPU is " + str(cpuCudaTimeDiff) + " seconds (" + str(cpuCudaPercentDiff) + "%) slower than using the CUDA.")
print("Rescaling using the CPU is " + str(vicCpuTimeDiff) + " seconds (" + str(vicCpuPercentDiff) + "%) faster than using VIC.")
print("Rescaling using CUDA is " + str(vicCudaTimeDiff) + " seconds (" + str(vicCudaPercentDiff) + "%) faster than using the VIC.")
print("Rescaling using CUDA is " + str(cpuCudaTimeDiff) + " seconds (" + str(cpuCudaPercentDiff) + "%) slower than using the CPU.")

# vim: ts=8:sw=4:sts=4:et:ai