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
