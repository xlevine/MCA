# MCA
Maximum Covariance Analysis for CMIP datasets

Maximum covariance analysis (MCA) is powerful tool to study remote connection between climatic variables [see Bretherton et al., 1992; Wallace et al., 1992, for a description of MCA as an exploratory method]; MCA determines the spatial pattern of 2 climatological variables (e.g. precipitation and sea surface temperature) that can explain the largest fraction of their covariance (e.g. interannual covariance). In particular, it can be used to find the dominant mode of interaction between those 2 variables globally or over a specific region. This is done by performing a singular value decomposition (SVD) on the covariance matrix of the 2 variables.

A MCA returns left (right) singular vectors, which correspond to the spatial pattern of the leading modes of intermodel variability in the first (second) variable that is most correlated with the interannual variability of the second (first) variable (the eigenvectors). Associated with each Leading Mode is a timeseries (the eigenvalues).

Inputs of MCA: X_1[x,y,t], X_2[x,y,t], lat1[x,y], lat2[x,y], lon1[x,y], lon2[x,y], areas1[x,y], areas2[x,y], region1[x,y], region2[x,y]

Outputs of MCA: E1_N[x,y]; E2_N[x,y]; P_1[t]; P_2[t] (for each leading modes)    
