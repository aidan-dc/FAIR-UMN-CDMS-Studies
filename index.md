# Welcome to FAIR-UMN-CDMS

Welcome!

This is the project web-page for our project---FAIR-UMN-CDMS: Identifying Interaction Location in SuperCDMS Detectors.

In this project, we address the problem of accurately reconstruct the locations of interactions in theSuperCDMS detectors using machine learning methods. The approach is to use data collected with aradioactive source at known locations to train and qualify machine learning models.


## Detector and Its Data 

### Detector Information

A prototype SuperCDMS germanium detector was tested at the University of Minnesota with a radioactive source mounted on a movable stage that can scan from the edge to the center of the detector. The detector is disk-shaped with sensors placed on the top and bottom surfaces to detect the particles emitted by the radioactive source, which is shown in [Figure 1](https://).

<div align="center">
<figure><img src="figures/detector_photo.jpg" width="456"></figure>
<br>
<figcaption>Figure 1: A SuperCDMS dark matter detector.</figcaption>

</div>


The sensors measure phonons (quantized vibrations of the crystal lattice) that are produced by the interacting particle and travel from the interaction location to the sensors. The number of phonons and the relative time of arrival at a particular sensor depends on the positions of the interaction and
the sensor. The sensors are grouped into six regions on each side of the detector and each of these “channels” produces a waveform for every interaction. For the test performed at Minnesota, five channels on one side of the detector were used ([Figure 2](https://)). The movable radioactive source was used to produce interactions at thirteen different locations on the detector along a radial path from the central axis to close to the the detector’s outer edge ([Figure 3](https://)).

<div align="center">
<figure><img src="figures/detector_3d.png" width="300"><img src="figures/waveforms_example_12.55mm.png" width="300"></figure>
 <br>
<figcaption>Figure 2: Pulses from an interaction in a SuperCDMS detector.</figcaption>

</div>

<br>
<div align="center">
<figure><img src="figures/run73_source_positions.png" width="356"></figure>
<br>
<figcaption>Figure 3: Interaction locations included in the dataset.</figcaption>
</div>


### Data from the Detector
For each interaction a set of parameters was extracted from the signals from each of the five sensors. These sets have been divided into two datasets which we will refer to as the **full** and **reduced** datasets.

The full dataset provides 85 input parameters for each interaction. These include 1 amplitude parameter and 16 timing/shape parameters for the waveforms for each of the 5 channels. The timing parameters represent time points during the rise and fall of the waveform at which the waveform reaches a given percentage of its maximum height. The parameters are given names such as PCr40 - the 40%-point for channel C as the waveform is rising; and PFf80 - the 80%-point for channel F as the waveform is falling. These times are referenced to PAr20, an early point on the waveform for Channel A, the outer channel. Thus PAr20 is always zero, reducing the number of independent parameters to 84. The amplitude parameter is a measure of the size of each waveform, based on a comparison to a normalized waveform template. The parameters included are:

- P[A,B,C,D,F]r[10,20,30,40,50,60,70,80,90,95,100]
- P[A,B,C,D,F]f[95,90,80,40,20]
- P[A,B,C,D,F]amp

The reduced dataset represents the input parameters that have been publicly released; it contains 19 input parameters which represent information known to be sensitive to interaction location, including the relative timing between pulses in different channels, and features like the pulse shape. The parameters included in the reduced dataset for each interaction are:

- P[B,C,D,F]start:
  - The time at which the pulse rises to 20% of its peak with respect to Channel A 
- P[A,B,C,D,F]rise:
  - The time it takes for a pulse to rise from 20% to 50% of its peak
- P[A,B,C,D,F]width:
  - The width (in seconds) of the pulse at 80% of the pulse height
- P[A,B,C,D,F]fall:
  - The time it takes for a pulse to fall from 40% to 20% of its peak

These parameters are illustrated in Figure 4.

<div align="center">
<figure><img src="figures/r73_pulse_c_pos9_ev1253_starttime.png" width="300"><img src="figures/r73_pulse_c_pos9_ev1253_shapequants3.png" width="280"></figure>
 <br>
<figcaption>Figure 4: Visualization of pulse timing and shape parameters in the reduced dataset.</figcaption>

</div>

The reduced dataset does not include amplitude parameters. Although the amplitudes are relevant to position reconstruction (particularly the relative amplitudes), they were omitted for technical reasons. During data taking it became clear that the gains of the sensors varied significantly from one data period to the next, and since each period represents a different source position, this introduces a bias in the dataset. For the full dataset, an attempt was made to adjust the amplitudes to correct for the variations with time. For this reason any studies involving the full dataset will be repeated both with and without pulse amplitude information.


## Machine Learning Solution

### Dataset for Machine Learning
We need to a large number of (x,y) paris to train our machine learning model. In our current experimental data, we have 19 and 85 informative features for the reduced and full datasets respectively, extracted
from 5 observed signals (pulses) and 13 different interaction locations (see our [document](https://github.com/FAIR-UMN/FAIR-UMN-CDMS/blob/main/doc/FAIR%20Document%20-%20Identifying%20Interaction%20Location%20in%20SuperCDMS%20Detectors.pdf) for details). In total, we obtain 7151 (x, y) pairs, of which the details are shown in [Table 1](https://). We further split our dateset to *Model-learning subset (MLS)* and *Held-out subset (HOS)*, of which the detailed definitions are provided in our [document](https://github.com/FAIR-UMN/FAIR-UMN-CDMS/blob/main/doc/FAIR%20Document%20-%20Identifying%20Interaction%20Location%20in%20SuperCDMS%20Detectors.pdf).

<div align="center">
<figure><img src="figures/dnn_dataset.png" width="456"></figure>
 <br>
</div>

### Classical Regression Techniques

We use the classical linear regression techniques to obtain benchmark results for the impact location estimation problem. Along with the ordinary least-squared error (LSE) regression, we also look at regularized models using Ridge and Lasso Regression. Additionally, we use Principal Component Analysis (PCA) for dimensionality reduction and perform regression on PCA-transformed inputs.

<div align="center">
<figure><img src="figures/image13.png" width="300"><img src="figures/image14.png" width="300"></figure>
 <br>
<figcaption>Figure 5: The plot on the left shows how the model-predicted value compares with the true value for training, validation, and test/held-out datasets for an ordinary LSE regression. The plot on the right compares the RMSE losses from Lasso and Ridge regression models for the three datasets with the RMSE losses from the ordinary LSE regression model. These results are from the reduced dataset.</figcaption>

</div>

<div align="center">
<figure><img src="figures/image1.png" width="300"><img src="figures/image6.png" width="300"></figure>
 <br>
<figcaption>Figure 6: These plots compare the RMSE losses from Lasso and Ridge regression models for the training, validation, and test/held-out datasets with the RMSE losses from the ordinary LSE regression model. These results are from the full dataset without amplitude information (left) and with amplitude information (right).</figcaption>

</div>
  
  
<div align="center">
<figure><img src="figures/image11.png" width="200"><img src="figures/image8.png" width="200"><img src="figures/image4.png" width="200"></figure>
 <br>
<figcaption>Figure 7: These plots compare the RMSE losses from Lasso and Ridge regression models for the training, validation, and test/held-out datasets with the RMSE losses from the ordinary LSE regression model when PCA is used to reduce the input dataset. These results are from the extended dataset (left) w/o amplitude information and (right) with amplitude information. All models are use as many PCs as needed to account for 99% of the observed variance in the training data. For the three datasets considered, this required using 13, 14, and 16 PCs respectively.</figcaption>

</div>


<div align="center">
<figure><img src="figures/image3.png" width="600"></figure>
 <br>
<figcaption>Figure 8: Summary of validation (green) and test/held-out (white) RMSE Losses from different choices of dataset and regression model. The distributions of the losses are obtained from 20 different random splitting for the training and validation dataset while the HOS is kept the same for each case.</figcaption>

</div>
  


### Deep Neural Network Model
We implement our neural network with Pytorch 1.9.0 3 . The framework of our neural network model is shown in [Figure 5](https://). It is a fully-connected network with non-linearity activation functions. In particular, in each hidden layer except the output layer, we employ a linear layer followed by the batch normalization, leaky rectified activation, and dropout. For the output layer, we simply pass the learned features through a linear layer and obtain its prediction directly. For other settings, please refer to our [document](https://github.com/FAIR-UMN/FAIR-UMN-CDMS/blob/main/doc/FAIR%20Document%20-%20Identifying%20Interaction%20Location%20in%20SuperCDMS%20Detectors.pdf).


<div align="center">
<figure><img src="figures/dnn.png" width="356"></figure>
 <br>
<figcaption>Figure 5: The framework of deep neural network models.</figcaption>

</div>

### Results of Deep Neural Network Models

We show the test performance on our test set (held-out subset) in [Table 2](https://). We can observe that simply increasing the model complexity does not boost the performance on our dataset, rather it hurts the performance. Therefore, we argue that to achieve better performance, it is worth exploring novel network architectures or training paradigms. For more experimental results, please refer to our [document](https://github.com/FAIR-UMN/FAIR-UMN-CDMS/blob/main/doc/FAIR%20Document%20-%20Identifying%20Interaction%20Location%20in%20SuperCDMS%20Detectors.pdf).


<div align="center">
<figure><img src="figures/dnn_results.png" width="456"></figure>
 <br>

</div>



## Support or Contact

Having trouble with codes or setting up? Check out our [documentation](https://github.com/FAIR-UMN/FAIR-UMN-CDMS/blob/main/doc/FAIR%20Document%20-%20Identifying%20Interaction%20Location%20in%20SuperCDMS%20Detectors.pdf) or [contact support](https://) and we’ll help you sort it out.
