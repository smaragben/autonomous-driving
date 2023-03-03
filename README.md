## Autonomous driving using Udacity simulator and Nvidia model
This project is an implementation of [Udacity's project assignment "Behavioral Cloning"](https://github.com/udacity/CarND-Behavioral-Cloning-P3) that uses [Udacity's simulator](https://github.com/udacity/self-driving-car-sim).
Using the [project's description template](https://github.com/udacity/CarND-Behavioral-Cloning-P3) this solution was generated according to nvidia's model described in Boyarski's et al paper "End to End Learning for Self-Driving Cars" using PyTorch.

**Instructions to run simulation with pretrained model**

1. Setup anaconda environment :
```
conda env create -f conda_env.yml 
```
2. Activate anaconda environment :
```
conda activate torch_env
```
3. Download Simulation for your OS from here and choose a version from Version 2, 2/07/17
4. Download this repository and on the directory run:
```
python drive_new.py nvidia_working
```
5. Open the simulator by clicking on the file beta_simulation
6. Choose Autonomous mode and you're done!

**Instructions to train model**
1. Load the jupyter notebook provided here : training.ipynb to kaggle and load the dataset named udacity-self-driving-car-behavioural-cloning.
2. Download the model produced and save it to your local directory of this project. To use it you can run :
```
python drive_new.py <yoursavedmodel>
```


