### Dataset Generation
* How to generate expert path dataset from PRM* with smoothing 
```
python dataset_generation.py
```
* How to generate image dataset
```
python image_dataset_generation.py
```

### Training Network Models
* How to train MPNet based on obstacle cloud and path
```
python mpnet_train.py
```
* How to train visual network
```
python visuomotor_train.py
```

### Testing Network Models
* mpnet_test can test multi paths same as assignment 3
* new_net_test can test the network based on specific start and goal configurations
```
python new_net_test.py
```
* How to test the accuracy for the trained visual network
```
python visuomotor_test.py
```
* How to run the demo for using the trained visual network in the simulator (testing for one config + image)
```
python visuomotor_demo.py
```

### Testing whole architecture
* How to test the success rate of the whole architecture
```
python Multi_finalDemo.py

Note: datasets are not included due to the size limit
