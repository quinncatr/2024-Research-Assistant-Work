import sys
import vpi
import numpy as np
from PIL import Image
from argparse import ArgumentParser
from jtop import jtop
from timeit import default_timer as timer

#python3 RescaleBenchmarking.py 
# ../../../../opt/nvidia/vpi1/samples/assets/kodim08.png 
# ../../../../opt/nvidia/vpi1/samples/assets/kodim09.png 
# ../../../../opt/nvidia/vpi1/samples/assets/kodim10.png

#with jtop() as jetson:
    #print(jetson.stats)

def parseArgs():
    global args
    # Parse command line arguments
    parser = ArgumentParser()

    parser.add_argument('vicInput',
                        help='Image to be used as input')
    parser.add_argument('cudaInput',
                        help='Image to be used as input')
    parser.add_argument('cpuInput',
                        help='Image to be used as input')

    args = parser.parse_args();

def vicOperations():
    global vicTime
    # Using the chosen backend,
    with vpi.Backend.VIC:
        # Load input into a vpi.Image
        input = vpi.asimage(np.asarray(Image.open(args.vicInput)))
    
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

    print("\nRescaling using VIC was completed in " + str(vicTime) + " seconds.")

def cudaOperations():
    global cudaTime
    with vpi.Backend.CUDA:
        input = vpi.asimage(np.asarray(Image.open(args.cudaInput)))
    
        temp = input.convert(vpi.Format.NV12_ER, backend=vpi.Backend.CUDA)
        start = timer()
        temp = temp.rescale((input.width//2, input.height//3))
        end = timer()
        output = temp.convert(input.format, backend=vpi.Backend.CUDA)
        
        cudaTime = end - start

    #Image.fromarray(output.cpu()).save('scaled_python_CUDA'+'.jpeg')

    print("Rescaling using CUDA was completed in " + str(cudaTime) + " seconds.")

def cpuOperations():
    global cpuTime
    with vpi.Backend.CPU:
        input = vpi.asimage(np.asarray(Image.open(args.cpuInput)))

        temp = input.convert(vpi.Format.NV12_ER, backend=vpi.Backend.CPU)
        start = timer()
        temp = temp.rescale((input.width//2, input.height//3))
        end = timer()
        output = temp.convert(input.format, backend=vpi.Backend.CPU)

        cpuTime = end - start

    #Image.fromarray(output.cpu()).save('scaled_python_CPU'+'.jpeg')

    print("Rescaling using CPU was completed in " + str(cpuTime) + " seconds.")


parseArgs()

print("-----Operations-----\n")

vicOperations()
cudaOperations()
cpuOperations()

print("\n")

print("-----Comparisons-----\n")

vicCpuTimeDiff = abs(vicTime - cpuTime)
vicCudaTimeDiff = abs(vicTime - cudaTime)
cpuCudaTimeDiff = abs(cpuTime - cudaTime)

vicCpuPercentDiff = (vicCpuTimeDiff / ((vicTime + cpuTime) / 2)) * 100
vicCudaPercentDiff = (vicCudaTimeDiff / ((vicTime + cudaTime) / 2)) * 100
cpuCudaPercentDiff = (cpuCudaTimeDiff / ((cudaTime + cpuTime) / 2)) * 100

print("Rescaling using VIC is " + str(vicCpuTimeDiff) + " seconds (" + str(vicCpuPercentDiff) + "%) slower than using the CPU.")
print("Rescaling using VIC is " + str(vicCudaTimeDiff) + " seconds (" + str(vicCudaPercentDiff) + "%) slower than using CUDA.\n")
print("Rescaling using the CPU is " + str(cpuCudaTimeDiff) + " seconds (" + str(cpuCudaPercentDiff) + "%) slower than using the CUDA.")
print("Rescaling using the CPU is " + str(vicCpuTimeDiff) + " seconds (" + str(vicCpuPercentDiff) + "%) faster than using VIC.\n")
print("Rescaling using CUDA is " + str(vicCudaTimeDiff) + " seconds (" + str(vicCudaPercentDiff) + "%) faster than using the VIC.")
print("Rescaling using CUDA is " + str(cpuCudaTimeDiff) + " seconds (" + str(cpuCudaPercentDiff) + "%) faster than using the CPU.\n")

# vim: ts=8:sw=4:sts=4:et:ai