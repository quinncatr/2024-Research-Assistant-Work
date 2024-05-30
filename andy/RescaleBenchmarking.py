import sys
import vpi
import numpy as np
from PIL import Image
from argparse import ArgumentParser
from jtop import jtop
from timeit import default_timer as timer

with jtop() as jetson:
    #print(jetson.stats)

# Parse command line arguments
    parser = ArgumentParser()
parser.add_argument('backend', choices=['cpu','cuda','vic'],
                    help='Backend to be used for processing')

parser.add_argument('input',
                    help='Image to be used as input')

args = parser.parse_args();

if args.backend == 'cpu':
    backend = vpi.Backend.CPU
    print("-----CPU-----")
elif args.backend == 'cuda':
    backend = vpi.Backend.CUDA
    print("-----CUDA-----")
else:
    assert args.backend == 'vic'
    backend = vpi.Backend.VIC
    print("-----VIC-----")

# Load input into a vpi.Image
input = vpi.asimage(np.asarray(Image.open(args.input)))

# Using the chosen backend,
start = timer()
with backend:
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
Image.fromarray(output.cpu()).save('scaled_python'+str(sys.version_info[0])+'_'+args.backend+'.png')

print("Rescaling using " + args.backend + " was completed in " + str(end - start) + " seconds.\n")

# Using the second and third backends:
timeList = []
i = 0
while i < 2:
    if args.backend == 'cpu':
        cpuTime = primaryTime
        timeList.append(cpuTime)
        args.backend = 'cuda'
        print("-----CUDA-----")
    elif args.backend == 'cuda':
        cudaTime = primaryTime
        timeList.append(cudaTime)
        args.backend = 'vic'
        print("-----VIC-----")
    elif args.backend == 'vic':
        vicTime = primaryTime
        timeList.append(vicTime)
        args.backend = 'cpu'
        print("-----CPU-----")
    startOtherChip = timer()
    with backend:
        # First convert input to NV12_ER.
        # We're overriding the default backend with CUDA.
        temp = input.convert(vpi.Format.NV12_ER, backend=vpi.Backend.CUDA)

        # Rescale the image using the chosen backend
        temp = temp.rescale((input.width//2, input.height//3))

        # Convert result back to input's format
        output = temp.convert(input.format, backend=vpi.Backend.CUDA)
    endOtherChip = timer()
    timeList.append((endOtherChip - startOtherChip))
    
    print("Rescaling using " + args.backend + " was completed in " + str(endOtherChip - startOtherChip) + " seconds.\n")
    i += 1



print("-----Comparisons-----")
print("VIC is" + " seconds (x%) slower/faster than CUDA and x seconds (x%) slower/faster than CPU")
del timeList[2]
print(timeList)






# vim: ts=8:sw=4:sts=4:et:ai