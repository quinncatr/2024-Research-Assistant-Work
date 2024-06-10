import sys
import vpi
import numpy as np
from PIL import Image
from argparse import ArgumentParser
import time
  
 # Parse command line arguments
parser = ArgumentParser()
  
parser.add_argument('input', help='Image to be used as input')
  
args = parser.parse_args();
  
 # Load input into a vpi.Image
try:
    input = vpi.asimage(np.asarray(Image.open(args.input)))
except IOError:
    sys.exit("Input file not found")
except:
    sys.exit("Error with input file")
    
with vpi.Backend.CPU:

    start = time.time()
     # First convert input to NV12_ER. 
    temp = input.convert(vpi.Format.NV12_ER, backend=vpi.Backend.CPU)
  
     # Rescale the image
    temp = temp.rescale((input.width//2, input.height//3))
  
     # Convert result back to input's format
    output = temp.convert(input.format, backend=vpi.Backend.CPU)
    end = time.time()
 # Save result to disk
Image.fromarray(output.cpu()).save("CPU"+'.jpeg')
print("Benchmark for CPU " + str(end - start))

with vpi.Backend.CUDA:

    start = time.time()

    temp = input.convert(vpi.Format.NV12_ER, backend=vpi.Backend.CUDA)
    temp = temp.rescale((input.width//2, input.height//3))
    output = temp.convert(input.format, backend=vpi.Backend.CUDA)

    end = time.time()

Image.fromarray(output.cpu()).save("CUDA"+'.jpeg')
print("Benchmark for CUDA " + str(end - start))

with vpi.Backend.VIC:

    start = time.time()
    # Image conversion example
    temp = input.convert(vpi.Format.NV12_ER, backend=vpi.Backend.CUDA)
    temp = temp.rescale((input.width//2, input.height//3))
    output = temp.convert(input.format, backend=vpi.Backend.CUDA)

    end = time.time()

Image.fromarray(output.cpu()).save("VIC"+'.jpeg')
print("Benchmark for VIC " + str(end - start))