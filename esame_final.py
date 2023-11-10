from astropy.io import fits
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
import os 
path = os.getcwd()

TEST_COVARIANCE=False
PLOTS=True

Nbins=200
Nmeasures=10000


for test_i in range(3):

    test = test_i + 1 

    measures0=[]
    measures2=[]
    measures4=[]

    for i in np.arange(Nmeasures)+1:
        fname = path + f'/data/MockMeasures_2PCF_Test{test}/MockMeasures_2PCF_Correlation_MULTIPOLES_Test{test}_{i}.fits'

        file = fits.open(fname)
        table = file[1].data.copy()

        measures0.append(table['XI0']) 
        measures2.append(table['XI2']) 
        measures4.append(table['XI4'])

        if i==1:
            scale = table['SCALE']
        del table
        file.close()

    measures0=np.asarray(measures0).transpose()
    measures2=np.asarray(measures2).transpose()
    measures4=np.asarray(measures4).transpose() 


    # MEDIA E COVARIANZA NUMERICA

    mean_xi0   = np.mean(measures0,axis=1) 
    cov_xi0 = np.cov(measures0) 

    mean_xi2   = np.mean(measures2,axis=1) 
    cov_xi2 = np.cov(measures2)

    mean_xi4   = np.mean(measures4,axis=1) 
    cov_xi4 = np.cov(measures4)

    cov_mis = [cov_xi0, cov_xi2, cov_xi4]

    
    if TEST_COVARIANCE:
    
        print('Running test to see if I understand the covariance:')
    
        AVE0 = np.zeros((Nbins,),dtype=float) 
        COV0 = np.zeros((Nbins,Nbins),dtype=float) 
    
        AVE2 = np.zeros((Nbins,),dtype=float) 
        COV2 = np.zeros((Nbins,Nbins),dtype=float) 
    
        AVE4 = np.zeros((Nbins,),dtype=float) 
        COV4 = np.zeros((Nbins,Nbins),dtype=float) 
    
        COV02 = np.zeros((Nbins,Nbins),dtype=float) 
        COV04 = np.zeros((Nbins,Nbins),dtype=float) 
        COV24 = np.zeros((Nbins,Nbins),dtype=float) 
    
        for i in range(Nmeasures):
            AVE0 += measures0[:,i]
            AVE2 += measures2[:,i]
            AVE4 += measures4[:,i]
        AVE0 /= Nmeasures 
        AVE2 /= Nmeasures 
        AVE4 /= Nmeasures 
    
        for i in range(Nbins):
            for j in range(Nbins):
                COV0[i,j] = (np.sum(measures0[i]*measures0[j]) - AVE0[i]*AVE0[j]*Nmeasures) / (Nmeasures-1) # covarianza 
                COV2[i,j] = (np.sum(measures2[i]*measures2[j]) - AVE2[i]*AVE2[j]*Nmeasures) / (Nmeasures-1)
                COV4[i,j] = (np.sum(measures4[i]*measures4[j]) - AVE4[i]*AVE4[j]*Nmeasures) / (Nmeasures-1)

        print('Largest deviation between my calculation and numpy (multipole 0):') 
        print('multipole 0: {}'.format(np.max(np.abs(COV0-cov_xi0)))) 
        print('multipole 2: {}'.format(np.max(np.abs(COV2-cov_xi2)))) 
        print('multipole 4: {}'.format(np.max(np.abs(COV4-cov_xi4)))) 


    # MATRICE DI AUTOCORRELAZIONE MISURATA

    corr_xi0 = np.zeros((Nbins,Nbins),dtype=float)
    corr_xi2 = np.zeros((Nbins,Nbins),dtype=float)
    corr_xi4 = np.zeros((Nbins,Nbins),dtype=float)

    for i in range(Nbins):
        for j in range(Nbins):
            corr_xi0[i,j]=cov_xi0[i,j]/(cov_xi0[i,i]*cov_xi0[j,j])**0.5   
            corr_xi2[i,j] = cov_xi2[i,j]/(cov_xi2[i,i]*cov_xi2[j,j])**0.5 
            corr_xi4[i,j] = cov_xi4[i,j]/(cov_xi4[i,i]*cov_xi4[j,j])**0.5 
        


    # PARAMETRI  

    if test==1:
        sigs = [0.02, 0.02, 0.02] 
        ls = [25, 50, 75]        

    elif test==2:
        sigs = [0.02, 0.01, 0.005]
        ls = [50, 50, 50]

    else:
        sigs = [0.02, 0.01, 0.005]
        ls = [5, 5, 5]



    # COVARIANZA TEORICA 

    # per 1 singolo multipolo
    def covf(x1, x2, sig, l):
        return sig**2.*np.exp(-(x1 - x2)**2./(2.*l**2.)) 

    # per più multipoli 
    def covf1f2(x1, x2, sig1, l1, sig2, l2): 
        return (np.sqrt(2.*l1*l2)*np.exp(-(np.sqrt((x1 - x2)**2.)**2./(l1**2. + l2**2.)))*sig1*sig2)/np.sqrt(l1**2. + l2**2.) 


    # auto-covarianza teorica

    cov_th00 = np.zeros((Nbins,Nbins),dtype=float)
    cov_th22 = np.zeros((Nbins,Nbins),dtype=float)
    cov_th44 = np.zeros((Nbins,Nbins),dtype=float)

    for i in range(Nbins):
        for j in range(Nbins):
            cov_th00[i,j] = covf(scale[i],scale[j],sigs[0],ls[0]) 
            cov_th22[i,j] = covf(scale[i],scale[j],sigs[1],ls[1])
            cov_th44[i,j] = covf(scale[i],scale[j],sigs[2],ls[2]) 

    cov_th = [cov_th00, cov_th22, cov_th44]


    # cross-covarianza teorica 

    cross_th02 = np.zeros((Nbins,Nbins),dtype=float)
    cross_th04 = np.zeros((Nbins,Nbins),dtype=float)
    cross_th24 = np.zeros((Nbins,Nbins),dtype=float)

    for i in range(Nbins):
        for j in range(Nbins):
            cross_th02[i][j] = covf1f2(scale[i], scale[j], sigs[0], ls[0], sigs[1], ls[1])
            cross_th04[i][j] = covf1f2(scale[i], scale[j], sigs[0], ls[0], sigs[2], ls[2])
            cross_th24[i][j] = covf1f2(scale[i], scale[j], sigs[1], ls[1], sigs[2], ls[2]) 
    

    # CORRELAZIONE TEORICA 

    # matrice di auto-correlazione teorica

    corr_th00 = np.zeros((Nbins,Nbins),dtype=float)
    corr_th22 = np.zeros((Nbins,Nbins),dtype=float)
    corr_th44 = np.zeros((Nbins,Nbins),dtype=float) 

    for i  in range(Nbins):
        for j in range(Nbins):
            corr_th00[i][j] = cov_th00[i][j]/(cov_th00[i][i]*cov_th00[j][j])**0.5
            corr_th22[i][j] = cov_th22[i][j]/(cov_th22[i][i]*cov_th22[j][j])**0.5
            corr_th44[i][j] = cov_th44[i][j]/(cov_th44[i][i]*cov_th44[j][j])**0.5



    # matrice di cross correlazione teorica 

    cross_corr_th02 = np.zeros((Nbins,Nbins),dtype=float)
    cross_corr_th04 = np.zeros((Nbins,Nbins),dtype=float)
    cross_corr_th24 = np.zeros((Nbins,Nbins),dtype=float) 

    for i  in range(Nbins):
        for j in range(Nbins):
            cross_corr_th02[i][j] = cross_th02[i][j]/(cross_th02[i][i]*cross_th02[j][j])**0.5
            cross_corr_th04[i][j] = cross_th04[i][j]/(cross_th04[i][i]*cross_th04[j][j])**0.5
            cross_corr_th24[i][j] = cross_th24[i][j]/(cross_th24[i][i]*cross_th24[j][j])**0.5


    # matrice unica delle cross-correlazioni teoriche  

    l_1 = np.hstack((corr_th00, cross_corr_th02, cross_corr_th04 ))
    l_2 = np.hstack((cross_corr_th02, corr_th22, cross_corr_th24)) 
    l_3 = np.hstack((cross_corr_th04, cross_corr_th24 , corr_th44)) 
    crossmatrix_th = np.vstack((l_1,l_2,l_3))


    # COVARIANZA MISURATA 

    def cross_cov_m(mat_0,mat_2):

        rows,cols = mat_0.shape
        average_0 = np.zeros((rows,),dtype=float)
        average_2 = np.zeros((rows,),dtype=float)
        covariance = np.zeros((rows,rows),dtype=float)
        
        for i in range(cols):
            average_0 += mat_0[:,i]
            average_2 += mat_2[:,i]
        average_0 /= cols                     # average = average / cols
        average_2 /= cols                     # average = average / cols

            
        for i in range(rows):
            for j in range(rows):
                covariance[i,j] = (np.sum(mat_0[i]*mat_2[j]) - average_0[i]*average_2[j]*cols) / (cols-1)
            
        return covariance    
        

    crosscov_m02 = cross_cov_m(measures0,measures2)
    crosscov_m04 = cross_cov_m(measures0,measures4)
    crosscov_m24 = cross_cov_m(measures2,measures4)    


    # MATRICE DI CROSS-CORRELAZIONE MISURATA 

    cross_corr_m02 = np.zeros((Nbins,Nbins),dtype=float)
    cross_corr_m04 = np.zeros((Nbins,Nbins),dtype=float)
    cross_corr_m24 = np.zeros((Nbins,Nbins),dtype=float) 

    for i  in range(Nbins):
        for j in range(Nbins):
            cross_corr_m02[i][j] = crosscov_m02[i][j]/(crosscov_m02[i][i]*crosscov_m02[j][j])**0.5
            cross_corr_m04[i][j] = crosscov_m04[i][j]/(crosscov_m04[i][i]*crosscov_m04[j][j])**0.5
            cross_corr_m24[i][j] = crosscov_m24[i][j]/(crosscov_m24[i][i]*crosscov_m04[j][j])**0.5

    # matrice unica delle cross-correlazioni misurate 

    l1 = np.hstack((corr_xi0, cross_corr_m02, cross_corr_m04 ))
    l2 = np.hstack((cross_corr_m02, corr_xi2, cross_corr_m24)) 
    l3 = np.hstack((cross_corr_m04, cross_corr_m24 , corr_xi4)) 
    crossmatrix_m = np.vstack((l1,l2,l3))



    # Validazione: residui normalizzati  

    norm_residuals0 = np.zeros_like(cov_th00)
    norm_residuals2 = np.zeros_like(cov_th22)
    norm_residuals4 = np.zeros_like(cov_th44)
    norm_residuals02 = np.zeros_like(cross_th02)
    norm_residuals04 = np.zeros_like(cross_th04)
    norm_residuals24 = np.zeros_like(cross_th24)

    for i in range(Nbins):
        for j in range(Nbins):

            rho2 = cov_th00[i,j]/(np.sqrt(cov_th00[i,i]*cov_th00[j,j])) # Rij
            norm_residuals0[i,j] = (cov_th00[i,j]-cov_xi0[i,j]) * np.sqrt((Nmeasures-1.)/((1.+rho2)*cov_th00[i,i]*cov_th00[j,j]))

            rho2 = cov_th22[i,j]/(np.sqrt(cov_th22[i,i]*cov_th22[j,j])) # Rij
            norm_residuals2[i,j] = (cov_th22[i,j]-cov_xi2[i,j]) * np.sqrt((Nmeasures-1.)/((1.+rho2)*cov_th22[i,i]*cov_th22[j,j]))

            rho2 = cov_th44[i,j]/(np.sqrt(cov_th44[i,i]*cov_th44[j,j])) # Rij
            norm_residuals4[i,j] = (cov_th44[i,j]-cov_xi4[i,j]) * np.sqrt((Nmeasures-1.)/((1.+rho2)*cov_th44[i,i]*cov_th44[j,j]))

            rho2 = cross_th02[i,j]/(np.sqrt(cross_th02[i,i]*cross_th02[j,j])) # Rij
            norm_residuals02[i,j] = (cross_th02[i,j]-crosscov_m02[i,j]) * np.sqrt((Nmeasures-1.)/((1.+rho2)*cross_th02[i,i]*cross_th02[j,j]))

            rho2 = cross_th04[i,j]/(np.sqrt(cross_th04[i,i]*cross_th04[j,j])) # Rij
            norm_residuals04[i,j] = (cross_th04[i,j]-crosscov_m04[i,j]) * np.sqrt((Nmeasures-1.)/((1.+rho2)*cross_th04[i,i]*cross_th04[j,j]))

            rho2 = cross_th24[i,j]/(np.sqrt(cross_th24[i,i]*cross_th24[j,j])) # Rij
            norm_residuals24[i,j] = (cross_th24[i,j]-crosscov_m24[i,j]) * np.sqrt((Nmeasures-1.)/((1.+rho2)*cross_th24[i,i]*cross_th24[j,j]))


      
    rms_deviation0=np.std(norm_residuals0)
    rms_deviation2=np.std(norm_residuals2)
    rms_deviation4=np.std(norm_residuals4)

    rms_deviation02=np.std(norm_residuals02)
    rms_deviation04=np.std(norm_residuals04)
    rms_deviation24=np.std(norm_residuals24)


    print(f"rms deviation of normalized residuals:")
    print(f"multipole 0: {rms_deviation0}")
    print(f"multipole 2: {rms_deviation2}")
    print(f"multipole 4: {rms_deviation4}")

    print(f"rms deviation of normalized residuals (cross-covariance):")
    print(f"multipole 0: {rms_deviation02}")
    print(f"multipole 2: {rms_deviation04}")
    print(f"multipole 4: {rms_deviation24}")


    if rms_deviation0<1.1 and rms_deviation2<1.1 and rms_deviation4<1.1 :
        print("**********")
        print("* PASSED *")
        print("**********")
    else:
        print("!!!!!!!!!!")
        print("! FAILED !")
        print("!!!!!!!!!!")

    if rms_deviation02<1.1 and rms_deviation04<1.1 and rms_deviation24<1.1 :
        print("**********")
        print("* PASSED *")
        print("**********")
    else:
        print("!!!!!!!!!!")
        print("! FAILED !")
        print("!!!!!!!!!!")





    # PLOTS
        
    # multipolo 0 

    if PLOTS:

        gratio = (1. + 5. ** 0.5) / 2.

        dpi = 300
        cmin0 = -np.max(cov_th00)*0.05
        cmax0 =  np.max(cov_th00)*1.05

        # Matrix plot of measured covariance matrix  
        fig = plt.figure(figsize=(6,4))
        plt.title(f'Measured covariance matrix \n multipole 0 test {test}')
        plt.imshow(cov_xi0, vmin=cmin0, vmax=cmax0) 
        cbar = plt.colorbar(orientation="vertical", pad=0.02)
        cbar.set_label(r'$ C^{\xi}_{N}$')
        # PLOTNAME = 'Test%s_Measured_Matrix.png'%test
        # plt.savefig(PLOTNAME,dpi = dpi)
        plt.show()


        # Matrix plot of theoretical covariance matrix  
        fig = plt.figure(figsize=(6,4))
        plt.title(f'Theoretical covariance matrix \n multipole 0 test {test}')
        plt.imshow(cov_th00, vmin=cmin0, vmax=cmax0)
        cbar = plt.colorbar(orientation="vertical", pad=0.02)
        cbar.set_label(r'$ C^{\xi}_{N}$')
        # PLOTNAME = 'Test%s_Th_Matrix.png'%test
        # plt.savefig(PLOTNAME,dpi = dpi)
        plt.show()
    
        # Matrix plot of of residuals multipole 0 
        fig = plt.figure(figsize=(6,4))
        plt.title(f'Residuals \n multipole 0 test {test}')
        plt.imshow(norm_residuals0) # calcolo dei residui 
        cbar = plt.colorbar(orientation="vertical", pad=0.02)
        cbar.set_label(r'$ C^{\xi}_{N}$')
        # PLOTNAME = 'Test%s_Res_Matrix.png'%test
        # plt.savefig(PLOTNAME,dpi = dpi)
        plt.show()



    # multipolo 2

    if PLOTS:

        gratio = (1. + 5. ** 0.5) / 2.

        dpi = 300
        cmin2 = -np.max(cov_th22)*0.05
        cmax2 =  np.max(cov_th22)*1.05

        # Matrix plot of measured covariance matrix
        fig = plt.figure(figsize=(6,4))
        plt.title(f'Measured covariance matrix \n multipole 2 test {test}')
        plt.imshow(cov_xi2, vmin=cmin2, vmax=cmax2)
        cbar = plt.colorbar(orientation="vertical", pad=0.02)
        cbar.set_label(r'$ C^{\xi}_{N}$')
        # PLOTNAME = 'Test%s_Measured_Matrix.png'%test
        # plt.savefig(PLOTNAME,dpi = dpi)
        plt.show()


        # Matrix plot of theoretical covariance matrix
        fig = plt.figure(figsize=(6,4))
        plt.title(f'Theoretical covariance matrix \n multipole 2 test {test}')
        plt.imshow(cov_th22, vmin=cmin2, vmax=cmax2)
        cbar = plt.colorbar(orientation="vertical", pad=0.02)
        cbar.set_label(r'$ C^{\xi}_{N}$')
        # PLOTNAME = 'Test%s_Th_Matrix.png'%test
        # plt.savefig(PLOTNAME,dpi = dpi)
        plt.show()
    
        # Matrix plot of residuals multipole 4
        fig = plt.figure(figsize=(6,4))
        plt.title(f'Residuals \n multipole 2 TEST {test}')
        plt.imshow(norm_residuals2) # calcolo dei residui 
        cbar = plt.colorbar(orientation="vertical", pad=0.02)
        cbar.set_label(r'$ C^{\xi}_{N}$')
        # PLOTNAME = 'Test%s_Res_Matrix.png'%test
        # plt.savefig(PLOTNAME,dpi = dpi)
        plt.show()


    # multipolo 4 

    if PLOTS:

        gratio = (1. + 5. ** 0.5) / 2.

        dpi = 300
        cmin4 = -np.max(cov_th44)*0.05
        cmax4 =  np.max(cov_th44)*1.05

        # Matrix plot of measured covariance matrix
        fig = plt.figure(figsize=(6,4))
        plt.title(f'Measured covariance matrix \n multipole 4 test {test}')
        plt.imshow(cov_xi4, vmin=cmin4, vmax=cmax4)
        cbar = plt.colorbar(orientation="vertical", pad=0.02)
        cbar.set_label(r'$ C^{\xi}_{N}$')
        # PLOTNAME = 'Test%s_Measured_Matrix.png'%test
        # plt.savefig(PLOTNAME,dpi = dpi)
        plt.show()


        # Matrix plot of theoretical covariance matrix
        fig = plt.figure(figsize=(6,4))
        plt.title(f'Theoretical covariance matrix \n multipole 4 test {test}')
        plt.imshow(cov_th44, vmin=cmin4, vmax=cmax4)
        cbar = plt.colorbar(orientation="vertical", pad=0.02)
        cbar.set_label(r'$ C^{\xi}_{N}$')
        # PLOTNAME = 'Test%s_Th_Matrix.png'%test
        # plt.savefig(PLOTNAME,dpi = dpi)
        plt.show()
    
        # Matrix plot of residuals multipole 4
        fig = plt.figure(figsize=(6,4))
        plt.title(f'Residuals \n multipole 4 test {test}')
        plt.imshow(norm_residuals4) # calcolo dei residui 
        cbar = plt.colorbar(orientation="vertical", pad=0.02)
        cbar.set_label(r'$ C^{\xi}_{N}$')
        # PLOTNAME = 'Test%s_Res_Matrix.png'%test
        # plt.savefig(PLOTNAME,dpi = dpi)
        plt.show()
        plt.show()
        plt.show()



    # PLOT cross-covarianze TEORICHE 

    # multipoli 0,2  

    if PLOTS:

        gratio = (1. + 5. ** 0.5) / 2.

        dpi = 300
        cmin02 = -np.max(cross_th02)*0.05
        cmax02 =  np.max(cross_th02)*1.05

        # Matrix plot of cross covariance matrix multipole 0,2
        fig = plt.figure(figsize=(6,4))
        plt.title(f'Theoretical cross covariance matrix \n multipoles 0,2 test {test}')
        plt.imshow(cross_th02, vmin=cmin02, vmax=cmax02)
        cbar = plt.colorbar(orientation="vertical", pad=0.02)
        cbar.set_label(r'$ C^{\xi}_{N}$')
        # PLOTNAME = 'Test%s_Th_Matrix.png'%test
        # plt.savefig(PLOTNAME,dpi = dpi)
        plt.show()

    # multipoli 0,4 

    if PLOTS:

        gratio = (1. + 5. ** 0.5) / 2.

        dpi = 300
        cmin04 = -np.max(cross_th04)*0.05
        cmax04 =  np.max(cross_th04)*1.05

        # Matrix plot of cross covariance matrix multipole 0,4
        fig = plt.figure(figsize=(6,4))
        plt.title(f'Theoretical cross covariance matrix \n multipoles 0,4 test {test}')
        plt.imshow(cross_th04, vmin=cmin04, vmax=cmax04)
        cbar = plt.colorbar(orientation="vertical", pad=0.02)
        cbar.set_label(r'$ C^{\xi}_{N}$')
        # PLOTNAME = 'Test%s_Th_Matrix.png'%test
        # plt.savefig(PLOTNAME,dpi = dpi)
        plt.show()


    # multipoli 2,4 

    if PLOTS:

        gratio = (1. + 5. ** 0.5) / 2.

        dpi = 300
        cmin24 = -np.max(cross_th24)*0.05
        cmax24 =  np.max(cross_th24)*1.05

        # Matrix plot of cross covariance matrix multipole 2,4
        fig = plt.figure(figsize=(6,4))
        plt.title(f'Theoretical cross covariance matrix \n multipoles 2,4 test {test}')
        plt.imshow(cross_th24, vmin=cmin24, vmax=cmax24)
        cbar = plt.colorbar(orientation="vertical", pad=0.02)
        cbar.set_label(r'$ C^{\xi}_{N}$')
        # PLOTNAME = 'Test%s_Th_Matrix.png'%test
        # plt.savefig(PLOTNAME,dpi = dpi)
        plt.show()


    # PLOT cross-covarianze MISURATE

    # multipoli 0,2 

    if PLOTS:

        gratio = (1. + 5. ** 0.5) / 2.

        dpi = 300
        cmin02 = -np.max(crosscov_m02)*0.05
        cmax02 =  np.max(crosscov_m02)*1.05

        # Matrix plot of cross covariance matrix multipole 2,4
        fig = plt.figure(figsize=(6,4))
        plt.title(f'measured cross covariance matrix \n multipoles 2,4 test {test}')
        plt.imshow(crosscov_m02, vmin=cmin02, vmax=cmax02)
        cbar = plt.colorbar(orientation="vertical", pad=0.02)
        cbar.set_label(r'$ C^{\xi}_{N}$')
        # PLOTNAME = 'Test%s_Measured_Matrix.png'%test
        # plt.savefig(PLOTNAME,dpi = dpi)
        plt.show()
        
        # residuals 
        fig = plt.figure(figsize=(6,4))
        plt.title(f'Residuals cross covariance matrix \n multipoles 0,2 test {test}')
        plt.imshow(norm_residuals02) # calcolo dei residui 
        cbar = plt.colorbar(orientation="vertical", pad=0.02)
        cbar.set_label(r'$ C^{\xi}_{N}$')
        # PLOTNAME = 'Test%s_Measured_Matrix.png'%test
        # plt.savefig(PLOTNAME,dpi = dpi)
        plt.show()



    # multipoli 0,4

    if PLOTS:

        gratio = (1. + 5. ** 0.5) / 2.

        dpi = 300
        cmin24 = -np.max(crosscov_m04)*0.05
        cmax24 =  np.max(crosscov_m04)*1.05

        # Matrix plot
        fig = plt.figure(figsize=(6,4))
        plt.title(f'Measured cross covariance matrix \n multipoles 2,4 TEST {test}')
        plt.imshow(crosscov_m04, vmin=cmin04, vmax=cmax04)
        cbar = plt.colorbar(orientation="vertical", pad=0.02)
        cbar.set_label(r'$ C^{\xi}_{N}$')
        # PLOTNAME = 'Test%s_Measured_Matrix.png'%test
        # plt.savefig(PLOTNAME,dpi = dpi)
        plt.show()
        
        # residuals 
        fig = plt.figure(figsize=(6,4))
        plt.title(f'Residuals cross covariance matrix \n multipoles 0,4 test {test}')
        plt.imshow(norm_residuals04) # calcolo dei residui 
        cbar = plt.colorbar(orientation="vertical", pad=0.02)
        cbar.set_label(r'$ C^{\xi}_{N}$')
        # PLOTNAME = 'Test%s_Measured_Matrix.png'%test
        # plt.savefig(PLOTNAME,dpi = dpi)
        plt.show()

        

    # multipoli 2,4

    if PLOTS:

        gratio = (1. + 5. ** 0.5) / 2.

        dpi = 300
        #climit=max(np.max(theoretical_covariance),np.max(measured_covariance))
        cmin24 = -np.max(crosscov_m24)*0.05
        cmax24 =  np.max(crosscov_m24)*1.05

        # matrix plot
        fig = plt.figure(figsize=(6,4))
        plt.title(f'Measured cross correlation matrix \n multipoles 2,4 TEST {test}')
        plt.imshow(crosscov_m24, vmin=cmin24, vmax=cmax24)
        cbar = plt.colorbar(orientation="vertical", pad=0.02)
        cbar.set_label(r'$ C^{\xi}_{N}$')
        # PLOTNAME = 'Test%s_Measured_Matrix.png'%test
        # plt.savefig(PLOTNAME,dpi = dpi)
        plt.show()
        
        # residuals
        fig = plt.figure(figsize=(6,4))
        plt.title(f'Residuals cross correlation matrix \n multipoles 2,4 test {test}')
        plt.imshow(norm_residuals24) 
        cbar = plt.colorbar(orientation="vertical", pad=0.02)
        cbar.set_label(r'$ C^{\xi}_{N}$')
        # PLOTNAME = 'Test%s_Measured_Matrix.png'%test
        # plt.savefig(PLOTNAME,dpi = dpi)
        plt.show()

    # matrice complessiva delle cross-correlazioni teoriche
    
    if PLOTS:

        gratio = (1. + 5. ** 0.5) / 2.

        dpi = 300
        cmin24 = -np.max(crossmatrix_th)*0.05
        cmax24 =  np.max(crossmatrix_th)*1.05

        # matrix plot
        fig = plt.figure(figsize=(6,4))
        plt.title(f'Theoretical cross correlation matrix \n TEST {test}')
        plt.imshow(crossmatrix_th, vmin=cmin24, vmax=cmax24)
        cbar = plt.colorbar(orientation="vertical", pad=0.02)
        cbar.set_label(r'$ C^{\xi}_{N}$')
        # PLOTNAME = 'Test%s_Th_Matrix.png'%test
        # plt.savefig(PLOTNAME,dpi = dpi)
        plt.show()    


    # matrice complessiva delle cross-correlazioni misurate

    if PLOTS:

        gratio = (1. + 5. ** 0.5) / 2.

        dpi = 300
        cmin24 = -np.max(crossmatrix_m)*0.05
        cmax24 =  np.max(crossmatrix_m)*1.05

        # matrix plot
        fig = plt.figure(figsize=(6,4))
        plt.title(f'Measured cross correlation matrix \n TEST {test}')
        plt.imshow(crossmatrix_m, vmin=cmin24, vmax=cmax24)
        cbar = plt.colorbar(orientation="vertical", pad=0.02)
        cbar.set_label(r'$ C^{\xi}_{N}$')
        # PLOTNAME = 'Test%s_Measured_Matrix.png'%test
        # plt.savefig(PLOTNAME,dpi = dpi)
        plt.show()
        
    