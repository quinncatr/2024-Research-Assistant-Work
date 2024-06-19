## Getting Started with the Jetson Nano:
- Unbox the Jetson Nano
- Make sure you have one of each: at least 32GB microSD card, a USB keyboard and mouse of you're planning to use it on a monitor, monitor with HDMI / DP cord, and a micro USB power supply.
- Flash the Jetson Nano Developer Kit SD Card Image onto the microSD by following the intructions on Nvidia's website: [link](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit#write) you have to download both the formatting software and etcher which flashes the Image zip file to your nano. This'll take a few minutes.
- Once your microSD card has the Image on it. You're gonna want to plug everything in, including the microSD card which can be inserted on the opposite side of the USB ports under the black plastic thing.
- After plugging everything in, you should see the monitor turn on with some setup information for you to fill out. 
- You should now be in!

## How to achieve SSH access to the Jetson Nano:
- You can use this command on your Virtual Machine to install client applications for SSH: 
`sudo apt install openssh-client`  
- And this command on the Nano to install server applications for SSH: 
`sudo apt install openssh-server`
- Once these are done, you can SSH into the Nano via the command: `ssh yournanousername@nanoip`
- This article on Ubuntu's website can help: [link](https://ubuntu.com/server/docs/openssh-server)

## How to Install Tensorflow on the Jetson Nano:
- After SSH-ing into the Nano, the next step is to install Tensorflow
- The webpage [Jetson_Zoo](https://elinux.org/Jetson_Zoo) which also is on the desktop of the nano when first booting is very helpful with installing necessary packages like Tensorflow.
- You're gonna want to do these commands in order:
```
sudo apt-get install libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran
sudo apt-get install python3-pip
sudo pip3 install -U pip testresources setuptools=49.6.0
sudo pip3 install -U numpy==1.19.4 future==0.18.2 mock==3.0.5 h5py==2.10.0 keras_preprocessing==1.1.1 keras_applications==1.0.8 gast==0.2.2 futures protobuf pybind11
```
- Depending on what version of Jetpack you have you'll have to change the number next to the v in the command below in my case it's 461 because I have JetPack v4.6.1
```
sudo pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v461 tensorflow
```

## VPI (Vision Programming Interface) Nvidia Module
- The VPI module is already installed on the Nano if you got the Jetpack
- The following are the operations/algorithms supported by VIC (Video and Image Compositor):

| VIC Algorithms |
| -------------- |
| Convert Image Format |
| Rescale |
| Remap |
| Perspective Warp |
| Lens Distortion Correction |
| Stereo Disparity Estimator |
| Temporal Noise Reduction |
| Image Flip |

- All the backends and the algorithms they work for are listed on: [link](https://docs.nvidia.com/vpi/algorithms.html)

## Convert Image Format/Rescale Example
- The code listed here: [link](https://github.com/network-synthesis/jetson-toolkit/blob/main/quinn/rescale.py) shows an example of both the Convert Image Format and Rescale algorithms
- It tests the rescale function with the CPU, CUDA, AND VIC backends and benchmarks their times
- It also uses the convert image format algorithm to change the intial image (I used a JPEG) to one the program can use in the program's case NV12_ER
- Rescale.py code is mostly from the Nivida Image Resampling sample code: [link](https://docs.nvidia.com/vpi/sample_rescale.html)
- From this program, I saw that the VIC is much faster at preforming the rescaling than the CPU or CUDA

## Mesh TensorFlow
- To get mesh tensorflow running, run this command:
`pip install mesh-tensorflow`
- Make sure you also have normal tensorflow installed
- The offical mesh tensorflow github repository is also helpful in getting things started: [link](https://github.com/tensorflow/mesh/blob/master/README.md)
- Mnist mesh example [here](https://github.com/network-synthesis/jetson-toolkit/tree/main/quinn/mesh) could not get to work on Jetson Nano but got working on Virtual Machine

## Distributed Tensorflow example between Jetson Nano and a Linux Virtual Machine
- Distributed training involves using multiple processors or 'workers' to divide the workload while training a model
- In the [distributedtraining](https://github.com/network-synthesis/jetson-toolkit/tree/main/quinn/distributedtraining) folder in the repository, there is distributed training code for each the Nano and the virtual machine
- This works by assigning each machine a worker index, 0 and 1 in this case, and opening a port for each machine's ip to connect to