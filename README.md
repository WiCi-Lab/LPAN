# Multi-Scale Attention Based Channel Estimation for RIS-Aided Massive MIMO Systems.
# Authors: Jian Xiao, Ji Wang, Zhaolin Wang, Wenwu Xie, and Yuanwei Liu.

This work has been submitted for possible publication. We highly respect reproducible research, so we try to provide the simulation codes for our submitted papers.

How to use this simulation code package?

1.Data Generation and Download

We have provided the paired samples in following link, where the LS-based pre-estimation processing and data normalization have been completed.

DOI Link: https://dx.doi.org/10.21227/3c2t-dz81

You can download the dataset and put it in the desired folder. The “LS_64_256R_6users_32pilot.mat” file includes the training and validation dataset, while the “LS_64_256R_test_6users_32pilot” file is used in the test phase.

Remark: We refer the channel modeling scheme in [1] for RIS-aided mmWave Massive MIMO systems, in which we further extend the single-user communication scenarios to the multi-user setup. Besides, we complement the NLOS channel modeling for the RIS-user link in the InH indoor office communication scenarios. 

Ackonwledge: We are very grateful for the author of following reference paper. Our dataset construction code is improved based on the open-source SimRIS Channel Simulator MATLAB package [2]. 

[1] E. Basar, I. Yildirim, and F. Kilinc, “Indoor and outdoor physical channel modeling and efficient positioning for reconfigurable intelligent surfaces in mmWave bands,” IEEE Trans. Commun., vol. 69, no. 12, pp. 8600-8611, Dec. 2021.

[2] E. Basar, I. Yildirim, “Reconfigurable Intelligent Surfaces for Future Wireless Networks: A Channel Modeling Perspective“, IEEE Wireless Commun., vol. 28, no. 3, pp. 108–114, June 2021.

2.The Training and Testing of LPAN/LPAN-L model

We have integrated the model training and test code, and you can run the “main.py” file to obtain the channel estimation result of the LPAN or LPAN-L model. The detailed network model is given in the “LPAN.py” and “LPAN-L.py”.

Notes: 

(1)	Please confirm the required library files have been installed.

(2)	Please switch the desired data loading path and network models.

(3)	In this work, our goal is to propose a general multi-scale channel estimation network backbone for RIS-aided communication systems. In the model training phase, we did not carefully find the optimal hyper-parameters. Intuitively, hyper-parameters can be further optimized to obtain better channel estimation performance gain.

The author in charge of this simulation code pacakge is: Jian Xiao (email: jianx@mails.ccnu.edu.cn). If you have any queries, please don’t hesitate to contact me.

Copyright reserved by the Holographic Communications Laboratory, Department of Electronics and Information Engineering, Central China Normal University, Wuhan 430079, China.
