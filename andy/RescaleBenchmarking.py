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

with vpi.Backend.CUDA:
    input = vpi.asimage(np.asarray(Image.open(args.input)))
    
    temp = input.convert(vpi.Format.NV12_ER, backend=vpi.Backend.CUDA)
    start = timer()
    temp = temp.rescale((input.width//2, input.height//3))
    end = timer()
    output = temp.convert(input.format, backend=vpi.Backend.CUDA)

    cudaTime = end - start

#Image.fromarray(output.cpu()).save('scaled_python_CUDA'+'.jpeg')

print("Rescaling using CUDA was completed in " + str(cudaTime) + " seconds.\n")


# Using the chosen backend,
with vpi.Backend.VIC:
    # Load input into a vpi.Image
    input = vpi.asimage(np.asarray(Image.open(args.input)))
    
    # First convert input to NV12_ER.
    # We're overriding the default backend with CUDA.
    temp = input.convert(vpi.Format.NV12_ER, backend=vpi.Backend.CUDA)
    start = timer()
    # Rescale the image using the chosen backend
    temp = temp.rescale((input.width//2, input.height//3))
    end = timer()
    # Convert result back to input's format
    output = temp.convert(input.format, backend=vpi.Backend.CUDA)
    
    vicTime = end - start

# Save result to disk
#Image.fromarray(output.cpu()).save('scaled_python_VIC'+'.jpeg')

print("\nRescaling using VIC was completed in " + str(vicTime) + " seconds.\n")


with vpi.Backend.CPU:
    input = vpi.asimage(np.asarray(Image.open(args.input)))

    temp = input.convert(vpi.Format.NV12_ER, backend=vpi.Backend.CPU)
    start = timer()
    temp = temp.rescale((input.width//2, input.height//3))
    end = timer()
    output = temp.convert(input.format, backend=vpi.Backend.CPU)
    
    cpuTime = end - start

#Image.fromarray(output.cpu()).save('scaled_python_CPU'+'.jpeg')

print("Rescaling using CPU was completed in " + str(cpuTime) + " seconds.\n")








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
print("Rescaling using CUDA is " + str(cpuCudaTimeDiff) + " seconds (" + str(cpuCudaPercentDiff) + "%) faster than using the CPU.")

#TODO: Find out why the order of the backends code changes elapsed time
#TODO: ex. vic last is .0006 seconds, but vic first is .01 seconds
#TODO: Uncomment save to disk line, just did that so it wouldn't keep creating images all the time
#TODO: My theory about the bug is that each chip is running at the same speed every time, but 
#TODO:                      that they get mislabeled in the print 
#TODO:                      statments depending on which one is first in the code
#TODO: When run 1 at a time, with the other 2 commented out, results are very very close to each other
# vim: ts=8:sw=4:sts=4:et:ai