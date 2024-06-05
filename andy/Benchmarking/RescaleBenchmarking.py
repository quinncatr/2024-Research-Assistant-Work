import sys
import vpi
import numpy as np
from PIL import Image
from argparse import ArgumentParser
from jtop import jtop
from timeit import default_timer as timer
from jtop import jtop
import pandas as pd

#python3 RescaleBenchmarking.py 
# ../../../../../opt/nvidia/vpi1/samples/assets/kodim08.png 

def parseArgs():
    global args
    # Parse command line arguments
    parser = ArgumentParser()
    parser.add_argument('Input',
                        help='Image to be used as input')
    args = parser.parse_args();

def vicOperations():
    global vicTime
    global vicPower
    global vicCurrent
    global vicVoltage

    with jtop() as jetson:
        vicData = pd.DataFrame(jetson.stats, index = [0])
        vicEnergy = pd.DataFrame(jetson.power)
    
    vicCurrent = vicEnergy['tot'][4]
    vicPower = vicEnergy['tot'][7]
    vicVoltage = vicEnergy["tot"][8]

    # Using the chosen backend,
    with vpi.Backend.VIC:
        # Load input into a vpi.Image
        input = vpi.asimage(np.asarray(Image.open(args.Input)))
    
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
    Image.fromarray(output.cpu()).save('scaled_python_VIC'+'.jpeg')

    print("Rescaling using VIC was completed in " + str(vicTime) + " seconds.")
    print("\n-----Usage-----")
    print(vicData[['APE', 'CPU1', 'CPU2', 'CPU3', 'CPU4', 'EMC', 'GPU', 'Temp AO', 'Temp CPU', 'Temp GPU', 'Temp PLL', 
                'Temp thermal', 'VIC03', 'time']])
    print("\n-----Power Consumption-----")
    print(vicEnergy[['tot']])
    print("\n")

def cudaOperations():
    global cudaTime
    global cudaPower
    global cudaCurrent
    global cudaVoltage

    with jtop() as jetson:
        cudaData = pd.DataFrame(jetson.stats, index = [0])
        cudaEnergy = pd.DataFrame(jetson.power)
    
    cudaCurrent = cudaEnergy['tot'][4]
    cudaPower = cudaEnergy['tot'][7]
    cudaVoltage = cudaEnergy["tot"][8]

    with vpi.Backend.CUDA:
        input = vpi.asimage(np.asarray(Image.open(args.Input)))
        temp = input.convert(vpi.Format.NV12_ER, backend=vpi.Backend.CUDA)

        start = timer()
        temp = temp.rescale((input.width//2, input.height//3))
        end = timer()

        output = temp.convert(input.format, backend=vpi.Backend.CUDA)
        cudaTime = end - start

    Image.fromarray(output.cpu()).save('scaled_python_CUDA'+'.jpeg')

    print("Rescaling using CUDA was completed in " + str(cudaTime) + " seconds.")
    print("\n-----Usage-----")
    print(cudaData[['APE', 'CPU1', 'CPU2', 'CPU3', 'CPU4', 'EMC', 'GPU', 'Temp AO', 'Temp CPU', 'Temp GPU', 'Temp PLL', 
                'Temp thermal', 'VIC03', 'time']])
    print("\n-----Power Consumption-----")
    print(cudaEnergy[['tot']])
    print("\n")

def cpuOperations():
    global cpuTime
    global cpuPower
    global cpuCurrent
    global cpuVoltage

    with jtop() as jetson:
        cpuData = pd.DataFrame(jetson.stats, index = [0])
        cpuEnergy = pd.DataFrame(jetson.power)

    cpuCurrent = cpuEnergy['tot'][4]
    cpuPower = cpuEnergy['tot'][7]
    cpuVoltage = cpuEnergy["tot"][8]

    with vpi.Backend.CPU:
        input = vpi.asimage(np.asarray(Image.open(args.Input)))
        temp = input.convert(vpi.Format.NV12_ER, backend=vpi.Backend.CPU)

        start = timer()
        temp = temp.rescale((input.width//2, input.height//3))
        end = timer()

        output = temp.convert(input.format, backend=vpi.Backend.CPU)
        cpuTime = end - start

    Image.fromarray(output.cpu()).save('scaled_python_CPU'+'.jpeg')

    print("\nRescaling using CPU was completed in " + str(cpuTime) + " seconds.")
    print("\n-----Usage------")
    print(cpuData[['APE', 'CPU1', 'CPU2', 'CPU3', 'CPU4', 'EMC', 'GPU', 'Temp AO', 'Temp CPU', 'Temp GPU', 'Temp PLL', 
                'Temp thermal', 'VIC03', 'time']])
    print("\n-----Power Consumption-----")
    print(cpuEnergy[['tot']])
    print("\n")

#-----Warmup methods store everything in cache in order for operations to not 
#-----have to do that themselves, which causes speed discrepencies
def cpuWarmup():
    with vpi.Backend.CUDA:
        input = vpi.asimage(np.asarray(Image.open(args.Input)))
        temp = input.convert(vpi.Format.NV12_ER, backend=vpi.Backend.CUDA)
        temp = temp.rescale((input.width//2, input.height//3))
        output = temp.convert(input.format, backend=vpi.Backend.CUDA)

def cudaWarmup():
    with vpi.Backend.CPU:
        input = vpi.asimage(np.asarray(Image.open(args.Input)))
        temp = input.convert(vpi.Format.NV12_ER, backend=vpi.Backend.CPU)
        temp = temp.rescale((input.width//2, input.height//3))
        output = temp.convert(input.format, backend=vpi.Backend.CPU)
    
    with vpi.Backend.VIC:
        input = vpi.asimage(np.asarray(Image.open(args.Input)))
        temp = input.convert(vpi.Format.NV12_ER, backend=vpi.Backend.CUDA)
        temp = temp.rescale((input.width//2, input.height//3))
        output = temp.convert(input.format, backend=vpi.Backend.CUDA)


parseArgs()

print("\n-----Operations-----\n")

cpuWarmup()
cpuOperations()

cudaWarmup()
cudaOperations()
vicOperations()

vicCpuTimeDiff = abs(vicTime - cpuTime)
vicCudaTimeDiff = abs(vicTime - cudaTime)
cpuCudaTimeDiff = abs(cpuTime - cudaTime)

vicCpuPercentDiff = round(((vicCpuTimeDiff / ((vicTime + cpuTime) / 2)) * 100), 2)
vicCudaPercentDiff = round(((vicCudaTimeDiff / ((vicTime + cudaTime) / 2)) * 100), 2)
cpuCudaPercentDiff = round(((cpuCudaTimeDiff / ((cudaTime + cpuTime) / 2)) * 100), 2)

print("-----Speed Comparisons-----\n")

print("Rescaling using the CPU is " + str(cpuCudaPercentDiff) + "% slower than using CUDA.")
print("Rescaling using the CPU is " + str(vicCpuPercentDiff) + "% faster than using VIC.\n")
print("Rescaling using CUDA is " + str(vicCudaPercentDiff) + "% faster than using the VIC.")
print("Rescaling using CUDA is " + str(cpuCudaPercentDiff) + "% faster than using the CPU.\n")
print("Rescaling using VIC is " + str(vicCpuPercentDiff) + "% slower than using the CPU.")
print("Rescaling using VIC is " + str(vicCudaPercentDiff) + "% slower than using CUDA.\n")

vicCpuPowerDiff = abs(cpuPower - vicPower)
vicCudaPowerDiff = abs(cudaPower - vicPower)
cpuCudaPowerDiff = abs(cpuPower - cudaPower)

vicCpuPowerPercentDiff = round(((vicCpuPowerDiff / ((cpuPower + vicPower) / 2)) * 100), 2)
cpuCudaPowerPercentDiff = round(((cpuCudaPowerDiff / ((cpuPower + cudaPower) / 2)) * 100), 2)
vicCudaPowerPercentDiff = round(((vicCudaPowerDiff / ((vicPower + cudaPower) / 2)) * 100), 2)

print("-----Energy Comparisons-----\n")
print("Rescaling using the CPU used " + str(vicCpuPowerPercentDiff) + "% more power than VIC.")
print("Rescaling using the CPU used " + str(cpuCudaPowerPercentDiff) + "% more power than CUDA.\n")
print("Rescaling using CUDA used " + str(vicCudaPowerPercentDiff) + "% more power than VIC.")
print("Rescaling using CUDA used " + str(cpuCudaPowerPercentDiff) + "% less power than the CPU.\n")
print("Rescaling using VIC used " + str(vicCudaPowerPercentDiff) + "% less power than CUDA.")
print("Rescaling using VIC used " + str(vicCpuPowerPercentDiff) + "% less power than the CPU.\n")


# vim: ts=8:sw=4:sts=4:et:ai