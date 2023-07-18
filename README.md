# Multi-Scale Attention Based Channel Estimation for RIS-Aided Massive MIMO Systems.
Authors: Jian Xiao, Ji Wang, Zhaolin Wang, Wenwu Xie, and Yuanwei Liu.

This work has been submitted for possible publication. We highly respect reproducible research, so we try to provide the simulation codes for our submitted papers.


## Updating version 2023.7.17 
### 1. Dataset updating
(1) Quasi-static channel estimation

In this version, we have reinvestigated the hardware imperfections in the system model, and have modified the pre-estimation operations in the dataset construction according to the insightful comments. The paired samples have been provided in the following link for the quasi-static channel estimation, in which the rar files are the updated dataset.

DOI Link: https://dx.doi.org/10.21227/3c2t-dz81

(2) Time-varying channel estimation

In this version, we have also increased the time-varying channel estimation scenario for the proposed LPAN-L model, in which the user mobility is considered in the cascaded channel modeling. The time-varying channel dataset has been provided in the following link.

DOI Link: https://dx.doi.org/10.21227/pz7h-q132

In this time-varying channel dataset, we consdier the number of pilot block is set to $B^\text{p}=2$ within a frame with $B=6$ blocks. The sampling period $T_b$ of each time block is fixed as $T_b \approx 0.24$ ms. The specific simulation parameters of this dataset have been elaborated in our submitted paper. The dataset in the DOI link is composed of training, vadilation and test data. Please put them in the desired folder.

### 2. Model updating
(1) Quasi-static channel estimation

The provided code in the original version is still applicable to the updated dataset for the quasi-static channel estimation.

(2) Time-varying channel estimation

In the file of 'Mobility_LPAN_L1.py', we have provided the improved LPAN-L architecture for the time-varying channel estimation. 

In the file of 'main_Mobility.py', we have provided the execute function for the time-varying channel estimation, in which the transfer learning framework has also been presented for the domain adpation of the proposed LPAN-L architecture.

## Original version 2023.1.19

How to use this simulation code package?

### 1.Data Generation and Download

We have provided the paired samples in the following link, where the LS-based pre-estimation processing and data normalization have been completed.

DOI Link: https://dx.doi.org/10.21227/3c2t-dz81

You can download the dataset and put it in the desired folder. The “LS_64_256R_6users_32pilot.mat” file includes the training and validation dataset, while the “LS_64_256R_test_6users_32pilot” file is used in the test phase.

Remark: We refer to the channel modeling scheme in [1] for RIS-aided mmWave Massive MIMO systems, in which we further extend the single-user communication scenarios to the multi-user setup. Besides, we complement the NLOS channel modeling for the RIS-user link in the InH indoor office communication scenarios. 

Ackonwledge: We are very grateful for the author of following reference paper. Our dataset construction code is improved based on the open-source SimRIS Channel Simulator MATLAB package [2]. 

[1] E. Basar, I. Yildirim, and F. Kilinc, “Indoor and outdoor physical channel modeling and efficient positioning for reconfigurable intelligent surfaces in mmWave bands,” IEEE Trans. Commun., vol. 69, no. 12, pp. 8600-8611, Dec. 2021.

[2] E. Basar, I. Yildirim, “Reconfigurable Intelligent Surfaces for Future Wireless Networks: A Channel Modeling Perspective“, IEEE Wireless Commun., vol. 28, no. 3, pp. 108–114, June 2021.

### 2.The Training and Testing of LPAN/LPAN-L model

We have integrated the model training and test code, and you can run the “main.py” file to obtain the channel estimation result of the LPAN or LPAN-L model. The detailed network model is given in the “LPAN.py” and “LPAN-L.py”.

Notes: 

(1)	Please confirm the required library files have been installed.

(2)	Please switch the desired data loading path and network models.

(3) In the training stage, the different hyper-parameters setup will result in slight difference for final channel estimation perfromance. According to our training experiences and some carried attempts, the hyper-parameters and network architecture can be further optimized to obtain better channel estimation performance gain, e.g., the dividing ratio between training samples and vadilation samples, the number of convolutional filters, the training learning rate, batchsize and epochs.

(4) Since the limitation of sample space (e.g., the fixed number of channel samples is collected for each user), the inevitable overfitting phenomenon may occur in the network training stage with the increase of epochs.

The author in charge of this simulation code pacakge is: Jian Xiao (email: jianx@mails.ccnu.edu.cn). If you have any queries, please don’t hesitate to contact me.

Copyright reserved by the WiCi Lab, Department of Electronics and Information Engineering, Central China Normal University, Wuhan 430079, China.
