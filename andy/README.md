## Jetson Nano Setup
### About Jetpack:
- The Nvidia Jetpack SDK is a software package that includes a linux based operating system in addition to various tools that can take advantage of the Nvidia Jetson Nano's CPU, GPU, and accelerators. Because Jetpack includes an operating system, it can be downloaded and used to boot the Nano. Once booted, the Nano will have an operating system, various development packages and tools, and examples of how those tools can be used.
- One caveat of the the Jetson Nano is that it does not support the newest versions of most software. For example, Jetpack 6 is not supported so Jetpack 4 is needed. In addition, the newest version of VScode that is supported is 1.83. 
### Installing Jetpack:
- To download Jetpack 4 from Nvidia, navigate to the Jetpack 4 page, which can be found [here](https://developer.nvidia.com/embedded/jetpack-sdk-461).
- Once on the Jetpack page, navigate to "Installing Jetpack" and click on "Jetson Nano Developer Kits". Once there, download the correct version based on the amount of RAM your Nano has. The default version is for a Nano with 4 GB of RAM. 
- To install the file that has been downloaded in the previous step, flash it to a micro sd card. Further instructions can be found [here](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit#write).
### Booting Jetson Nano:
- Once Jetpack is flashed to a microSD card, insert the microSD into the slot on the Nano. Once the Nano has power and is connected to a display, it should automatically boot and take you through standard OS setup. Furhter instructions can be found [here](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit#setup)
- Note: Versions of Jetpack newer than 4.6.1 will not boot.

## SSH into Jetson Nano
###

## Installing TensorFlow
