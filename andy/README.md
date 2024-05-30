# Nvidia Jetson Nano Research
## Jetson Nano Setup
### About Jetson Nano
- The Nvidia Jetson Nano is a small Raspberry Pi type machine to experiment with machine learning, GPU utilization and more. The Nano is a few years old and as a result, is only compatible with older versions of most software needed.
- The newest supported version of Jetpack for the Nano is Jetpack 4.6.1. All instructions in this document work for the Nvidia Jetson Nano running Jetpack 4.6.1. Any other machine or Jetpack version likley will have different steps than the ones laid out below.
### About Jetpack:
- The Nvidia Jetpack SDK is a software package that includes a linux based operating system in addition to various tools that can take advantage of the Nvidia Jetson Nano's CPU, GPU, and accelerators. Because Jetpack includes an operating system, it can be downloaded and used to boot the Nano. Once booted, the Nano will have an operating system, various development packages and tools, and examples of how those tools can be used.
- One caveat of the the Jetson Nano is that it does not support the newest versions of most software. For example, Jetpack 6 is not supported so Jetpack 4 is needed. In addition, the newest version of VScode that is supported is 1.83. 
### Installing Jetpack:
- To download Jetpack 4.6.1 from Nvidia, navigate to the Jetpack 4 page, which can be found [here](https://developer.nvidia.com/embedded/jetpack-sdk-461).
- Once on the Jetpack page, navigate to "Installing Jetpack" and click on "Jetson Nano Developer Kits". Once there, download the correct version based on the amount of RAM your Nano has. The default version is for a Nano with 4 GB of RAM. 
- To install the file that has been downloaded in the previous step, flash it to a micro sd card. Further instructions can be found [here](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit#write).
### Booting Jetson Nano:
- Once Jetpack is flashed to a microSD card, insert the microSD into the slot on the Nano. Once the Nano has power and is connected to a display, it should automatically boot and take you through standard OS setup. Further instructions can be found [here](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit#setup).
  - Note: Versions of Jetpack newer than 4.6.1 will not boot.
## Additional Information
### The following links provide more information about the Jetson Nano and other devices.
- [Nvidia Jetson Developer Guide](https://docs.nvidia.com/jetson/archives/r36.3/DeveloperGuide/index.html)
- [Nvidia Jetson DownLoad Center](https://developer.nvidia.com/embedded/downloads)
- [Getting Started with Nvidia Jetson Nano](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit)
- [Nvidia Jetson Nano Supported Components](https://developer.download.nvidia.com/assets/embedded/secure/jetson/Nano/docs/Jetson_Nano_Supported_Components_List_NV.pdf?76ypPONhPK8pnVWKKmtrcDs8GSSXFzGyO5o03ipRp0qyxzI9_ahJyPqFqwQqMsE_j49n7W-NixTZ_dqhQBKy0qT6Tm1nbsY9KRl-ZnN4kLiqOXWDAjaSzrse1t7U2wanxvJOOMoAg7xmbNcomI3d5rtrpKlv80qgbpkyCsAGFRO4SVbpcT4_uxlogxHmtV9wIHQu&t=eyJscyI6InJlZiIsImxzZCI6IlJFRi1kdWNrZHVja2dvLmNvbS8ifQ==)
- [Nvidia Jetson Nano Developer Kit User Guide](https://developer.download.nvidia.com/assets/embedded/secure/jetson/Nano/docs/NV_Jetson_Nano_Developer_Kit_User_Guide.pdf?RiV_dF65QGf4OFgHiKstg8sPxBmWBNHMvXjOzp87KE9QISQQpuBoE9b8Zyh5RCTVXABr_97DdoGWEmfUdL3KXSOmAER0PmRvZ3Vdsp-U98lsIBCj_l_f9lp5yqGdUGx2Z6ROhlbcvf2G3Sfxt6RQgidEeUdz7Vr1dQg8276fuhcaNFgYpNP45hslUgP3Rn4dG7s=&t=eyJscyI6InJlZiIsImxzZCI6IlJFRi1kdWNrZHVja2dvLmNvbS8ifQ==)
- [Nvidia Jetson Webinar Video Series](https://www.youtube.com/playlist?list=PL5B692fm6--tMDTpxk_5akMWlyq2n-qmW)
- [RidgeRun Developer Jetson Nano Resources](https://developer.ridgerun.com/wiki/index.php/Category:JetsonNano)
  - Note: Some of these links have files that are not compatible with the Nano. For downloading supported software, only use information above. 

## SSH into Jetson Nano
- Once Jetpack is installed and the Nano is booted, it is easier to run everything else through an SSH connection. The Nano struggles when using a monitor and external IO devices. Once an SSH connection is established, run the rest of the commands from another linux machine that is remotely connected via SSH to the Jetson Nano.
### SSH Setup
- To ensure that the Nvidia Jetson Nano is able to act as a remote server, run `sudo apt install openssh-client openssh-server` in the terminal of the Nano. Type yes when prompted and follow the instructions given. Once that is done, run the same exact command on the linux machine that will remotely connect to the Nano. By running both openssh-client and openssh-server, both machines can ssh into each other. If you want only one machine to ssh into another, run `sudo apt install openssh-server` on the Nano and `sudo apt install openssh-client` on your linux machine.
### Creating an SSH Connection
- Once openssh is set up, connecting to the Nano is as simple as running `ssh username@IPaddress` on your linux machine. The username and IP address are the username and IP address associated with the Nano. You can find the IP address of the Nano by running `hostname -I`. The IP address needed will be the first IP address displayed. It will ask for the password of the Nano and once logged in, the SSH connection is active.
### Remote SSH via VScode
- Once an SSH connection is setup through the terminal, an SSH connection can be setup through VScode. Download the Remote SSH extension in VScode. Then, click on the two arrows in the bottom left hand corner. Click on "Connect to Host". The first time connecting you will have to type `ssh username@IPaddress` in the search bar and hit enter. If it asks to configure SSH hosts, click the first option. Once VScode is able to complete the connection, it will prompt you for the password of the remote machine. Once you have logged in, you are now connected remotley via SSH to the Jetson Nano. 
- Once you connect the first time, you will not have to spcify the server again. If you need to reconnect, click the arrows in the bottom left, then "Connect to Host" and the IP address of the Nano will be displayed. Click the IP address liusted and once logged in, the SSH connection is active. 

## Installing TensorFlow
### TensorFlow version
- Due to the Jetson Nano supporting only up to Jetpack 4.6.1, TensorFlow 2.7.0 is the version that works best. In order to install TensorFlow 2.7.0, Python 3.6 is needed. Once the correct version of Python is installed, TensorFlow can be installed.
### TensorFlow Install
- To install TensorFlow 2.7.0 with Jetpack 4.6.1, run the following commands in order.
  ```
  sudo apt-get update
  sudo apt-get install -y python3-pip pkg-config
  sudo apt-get install -y libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran
  sudo ln -s /usr/include/locale.h /usr/include/xlocale.h
  sudo pip3 install --verbose 'protobuf<4' 'Cython<3'
  sudo wget --no-check-certificate https://developer.download.nvidia.com/compute/redist/jp/v461/tensorflow/tensorflow-2.7.0+nv22.1-cp36-cp36m-linux_aarch64.whl
  sudo pip3 install --verbose tensorflow-2.7.0+nv22.1-cp36-cp36m-linux_aarch64.whl
  ```
  - Note: These commands can take 10 - 15 minutes to complete. 
### Verifying TensorFlow Install
- Once the previous commands are finished running, run `pip show tensorflow` to verify that tensorflow has been installed.

## Using TensorFlow
### Initial Setup
- Because the Jetson Nano is slightly outdated, the most recent version of Keras supported is 2.6. Since Tensorflow automatically comes with a newer version than 2.6, it must be downgraded. To downgrade Keras, run `pip install keras==2.6` in the Nano's terminal.
- Once TensorFlow is installed, follow the quickstart [tutorial](https://www.tensorflow.org/tutorials/quickstart/beginner).
  - Note: If this tutorial throws an error about the GPU memory being maxed out, add this code block to reallocate memory.
 ```
    device = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(device[0], True)
    tf.config.experimental.set_virtual_device_configuration(device[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
```
  - By default, Tensorflow has a fixed amount of GPU memory it can use. The code above allows TensorFlow to dynamically allocate GPU memory, which allows it to increase the amount of memory it uses if the dataset takes up more than the default amount. 
  - Note: If the tutorial core dumps, a different version of numpy may be needed. The most stable version for the Nano is 1.19.4. To install the stable version of numpy, run `pip3 install numpy==1.19.4` in the Nano's terminal.

## VPI (Video Programming Interface)
### Installation
- Due to the Jetson Nano running Jetpack 4.6.1, VPI 1.2 is the newest supported version of VPI. Jetpack 4.6.1 includes Ubuntu 18.04, which determines the version of VPI that can be used.
- To install VPI 1.2 on the Nano, run the following commands in the terminal. 
  - Note: These commands will only work for Jetpack 4.6.1 running Ubuntu 18.04
```
sudo apt-get update
sudo apt-get install libnvvpi1
sudo apt install gnupg
sudo apt-key adv --fetch-key https://repo.download.nvidia.com/jetson/jetson-ota-public.asc
sudo apt install software-properties-common
sudo add-apt-repository 'deb https://repo.download.nvidia.com/jetson/x86_64 bionic r32.7'
sudo apt update
sudo apt install libnvvpi1 vpi1-dev vpi1-samples
sudo apt install vpi1-demos
```
- Examples of how VPI can be used can be found in `/opt/nvidia/vpi1`
### VPI 1.2 Operations
- Despite being an older version of VPI, many operations are available and supported in both C++ and Python.
- For more information on how VPI works read the [Nvidia VPI architecture document](https://docs.nvidia.com/vpi/1.2/architecture.html).
- The following table lists the algorithms inlcuded in Nvidia VPI along with the backends that they can be run on.
  - Note: The table is only compatibale with a Jetson Nano running VPI 1.2.

### VIC Operations

| Algorithm | CPU | CUDA | PVA | VIC |
| --------- | --- | ---- | --- | --- |
| Temporal Noise Reduction | no | yes | no | yes |
| Convert Image Format | yes | yes | no | yes |
| Rescale | yes | yes | no | yes |

### Other Operations

| Algorithm | CPU | CUDA | PVA | VIC |
| --------- | --- | ---- | --- | --- |
| Box Filter | yes | yes | yes | no |
| Bilateral Filter | yes | yes | no | no |
| Gaussian Filter | yes | yes | yes | no |
| Laplacian Pyramid Generator | yes | yes | no | no |
| Erode | yes | yes | no | no |
| Dilate | yes | yes | no | no |
| Convolution | yes | yes | yes | no |
| Seperable Convolution | yes | yes | yes | no |
| Remap | yes | yes | no | no |
| Perspective Warp | yes | yes | no | no |
| FFT | yes | yes | no | no |
| Inverse FFT | yes | yes | no | no |
| Lens Distortion Correction | yes | yes | no | no |
| Stereo Disparity Estimator | yes | yes | yes | no |
| KLT Feature Tracker | yes | yes | yes | no |
| Harris Corner Detector | yes | yes | yes | no |
| Pyramidal LK Optical Flow | yes | yes | no | no |
| Image Histogram | yes | yes | no | no |
| Equalize Histogram | yes | yes | no | no |
| Background Subtractor | yes | yes | no | no |
| Min/Max Location | yes | yes | no | no |

  - To find more information about what these operations do and how they work, go to [Nvidia VPI1.2 Algorithms](https://docs.nvidia.com/vpi/1.2/algorithms.html).

## JTOP
### JTOP overview
- JTOP is a program that can be run in the terminal to dsiplay data about the Jetson Nano hardware including CPU, GPU, VIC, etc usage and temperatures, along with individual cores usage and temperatures. It has many more readings to show the state of the Jetson Nano
### Installing JTOP
- To install JTOP, make sure PIP is installed along with python3. Once those are installed run `sudo pip3 install -U jetson-stats` in the terminal. You may need to logout and/or reboot the Nano before running JTOP. Once the Nano has been rebooted, simply run `jtop` in the terminal.
### Using JTOP in a python program
- JTOP is capable of being imported into a python program and displaying a text summary of the Nanos stats
- To import, add `from jtop import jtop` to the program. Then add `with jtop90 as jetson:` to call jtop methods on the jetson.
  - For example, once the imports are complete `print(jetson.stats)` displays CPU, GPU and accelerator data like usage, temps, power draw, etc in the terminal when the program is run.

## Initial VPI Benchmakring
- In order to learn more about the Nano and how its various accelerators work, initial benchmarking must be done.
### Temporal Noise Reduction Benchmarking
- The `TNRBenchmarking.py` file uses the sample temporal noise reduction program supplyed by VPI with added benchmarking features to compare the speed at which CUDA and VIC complete a basic temporal noise reduction program.
  - The added features keep track of elapsed time and report associated data.
- `TNRBenchmarking.py` provides data about which accelorator is used, how fast it completed, and how fast it completed relative to the other accelorators.
- To run `TNRBenchmarking,py` run the following commands in the directory where the program is located.

```
python TNRBenchmarking.py <accelorator> <input video filepath>
```
  - where `<accelorator>` is the specific accelorator you want to use and `<input video filepath>` is the filepath where the original video is located.
- Example command:

```
python TNRBenchmarking.py cuda ../../../../opt/nvidia/vpi1/samples/assets/noisy.mp4
```
- Note: This program will create a new mp4 file and store it in the directory that the program was run from.

### Rescaling Benchmarking
- To benchmark the `Rescaling.py` sample, pillow must be installed. to do this, run ` sudo pip3 install pillow`.
- The `RescaleBenchmarking.py` program runs the rescaling operation on the CPU, CUDA and VIC and displays data about the speed each operation took and how it compares to the other accelerators.
- Interestingly, after initial benchmarking tests were written, the speed of each chip would change significantly depending on which chip ran the operation first.
