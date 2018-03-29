# RESDOG: A transfer-learning based approach to dog breed identification using Keras/CNTK

v. 0.0.1
Very much a work in progress!

This project is derived from an assignment completed as part of the [Udacity Deep Learning Nanodegree](https://www.udacity.com/course/deep-learning-nanodegree--nd101), in which we develop a dog breed classifier using transfer learning, starting with a pre-trained (on ImageNet) ResNet-50 and then modifying it for our task, re-training only the new fully-connected layer(s).

## Obtaining the data
1. Download the dog image dataset from [here](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip). Extract it in your repo/working directory such that there is now a directory `foo/dogImages`.

2. Download the human image dataset from [here](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip). Extract as above so there is a directory `foo/lfw`.

3. Download the VGG-16 bottleneck features for the dog images [here](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG16Data.npz) and place the `.npz` file in a new directory `foo/bottleneck_features`.

4. Download the actual ResNet-50 bottleneck features [here](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogResnet50Data.npz). Also place this file in the `bottleneck_features` directory.

## Setting up the environment

This project was created and run on a Windows 10 machine with one GTX 1080Ti, using Microsoft's [CNTK](https://docs.microsoft.com/en-us/cognitive-toolkit/) 2.4 (GPU-enabled) with Keras as a frontend, with CUDA 9.0 and cudNN 7. The python version was Anaconda 3.5.3. This was also briefly tested using `tensorflow-gpu` as the Keras backend and seemed to work fine, as no CNTK-specific functionality is directly invoked. This also appears to work on Linux with no modification, at the very least on a headless Ubuntu 16.04 cloud instance with 1 Tesla K80, such as an Azure NC6 or AWS EC2 p2.xlarge.

#### CUDA

I am assuming here that you have a working CUDA/cudNN setup on your host OS, and that you have a working Anaconda 3.5/6 python installation available to you even if it is not your system python. If you do not, detailed instructions may be found elsewhere on the internet (in the meantime, though I may write them up soon as well), but for Windows and Linux, the general principle for everything not to explode/complain at you is to:

1. First, install CUDA by downloading the installers (Windows binaries and *nix shell scripts) **from the NVIDIA website**, not via, say, your OS's package manager. It's more annoying to update, yes, but I can't count the number of times I've had the distro-supplied CUDA-related package blow up everything when using Linux (I'm on Windows as my primary OS right now so it's less of a pain).

2. Second, install cudNN the same way.

3. If you're on Windows, CUDA may try to install a different driver version than you have if you have "GeForce Experience" and "game-ready drivers," specifically an older version. I let it. You can try your luck with the "game ready" drivers, but for me they usually break things massively and I have to purge everything NVIDIA-related from my system and do a clean CUDA install.

4. A similar principle applies on Linux with respect to distro-packaged proprietary nvidia drivers (or ones in community repos like the AUR, RPMFusion, etc.), but is often easier to fix if something goes wrong.

#### Installing and configuring environment

1. Create (and activate) a new Anaconda environment.

	- __Windows__ (to install __without__ GPU support, change `requirements/dog-windows-gpu-cntk.yml` to `requirements/dog-windows.yml`, replace `cntk` with `tf` to use tensorflow instead):  
	```
	conda env create -n "[name]" -f requirements/dog-windows-gpu-cntk.yml
	activate "[name]"
	```

	- __Linux__ (to install __without__ GPU support, change `requirements/dog-linux-gpu.yml` to `requirements/dog-linux.yml`): 
	```
	conda env create -n "[name]" -f requirements/dog-linux-gpu.yml
	source activate "[name]"
	```  
	
2. Install your preferred DL framework (Keras backend). Currently only CNTK and TensorFlow are explicitly supported.

  - __CNTK 2.4__ : It is easiest to simply follow the instructions for a python-only installation for python 3.5 [here](https://docs.microsoft.com/en-us/cognitive-toolkit/Setup-Windows-Python?tabs=cntkpy24) for Windows and [here](https://docs.microsoft.com/en-us/cognitive-toolkit/Setup-Linux-Python?tabs=cntkpy24) for Linux (they are mostly the same). Remember to choose the correct wheel file (python 3.5 GPU, whether you use 1-bit SGD is up to you and whether your project allows its use according to the license.) In either case we *do* need to install OpenCV, even though it is not a strict dependency for CNTK 2.3+.

  - __TensorFlow__ : GPU-enabled TensorFlow should be installed correctly by default if one uses the appropriate `requirements/*.yml` file, but if not, for Windows the instructions are [here](https://www.tensorflow.org/install/install_windows) and for Linux [here](https://www.tensorflow.org/install/install_linux). While the documentation says "Ubuntu," this is only because only Ubuntu is publicly, officially supported. The installation instructions are easily adapted to most other distros with the obvious modifications (different package naming conventions in the Red Hat world, for example).

3. Only *now* install Keras. 
	
	pip install Keras
	
4. Switch [Keras backend](https://keras.io/backend/) to CNTK. The default is TensorFlow, so if you wish to use it you may leave this default. There are two ways of doing this. The first way is to simply set the KERAS_BACKEND environment variable:
	- __Windows__: 
		```
		set KERAS_BACKEND=cntk
		python -c "from keras import backend"
		```
	- __Linux__: 
		```
		KERAS_BACKEND=tensorflow 
		python -c "from keras import backend"
		```
The other method is to set a system-wide default in the `keras.json` file, which is found in `%USERPROFILE%\.keras\` on Windows and `~/.keras` on *nix, by default. While in principle explicitly setting the environment variable should override this default, in practice I have run into circumstances where the backend would not change regardless. In case of any such environment variable issues, it's easy to just quickly edit the config file. Change:
	
	
	"backend": "tensorflow"
	
which is the default, to

	"backend": "cntk"



5. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for your environment environment. 
```
python -m ipykernel install --user --name "[name]" --display-name "display name"
```

6. Open the notebook.
```
jupyter notebook display name.ipynb
```

7. Change the kernel to match your environment by using the drop-down menu (**Kernel > Change kernel**). Then, follow the instructions in the notebook.

## TODO

* Perform proper hyperparameter search, particularly evolutionary search on learning rate.
* Augment training data
* Implement much more fine-grained face/dog detectors
* Add functionality for mixed-breeds
* Improve UI/UX beyond just "edit this code to specify a file and get text"

