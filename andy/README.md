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
- Once TensorFlow is installed, follow the quickstart tutorial [here](https://www.tensorflow.org/tutorials/quickstart/beginner).
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
- Nvidia VPI is included in the Jetpack SDK. Jetpack 4.6.1 comes with VPI 1.2 already installed


