import os
import sys
import numpy as np
import time
import GPy
import KernelFunctionUtilities as kpu
import GaussianProcessUtilities as gpu
from sklearn.metrics import mean_squared_error

    
## Example usage = python Figure5.py 1987 0 True 1000 200 10 True 5 500\
#    "UCI_Communities_Modified.csv"

def main():
    # get simulation parameter settings
    numpySeed = int( sys.argv[1] ) # seed for random number generator 
    kernelOrder = int(sys.argv[2]) # order of Cho+Saul base kernel
    kernelAntiSymmetric = (sys.argv[3]=='True' ) # flag to indicate if base kernel is anti-symmetric
    nTrainLinear = int(sys.argv[4])  # number of training points to use with linear kernel to determine ARD weights
    nTest = int(sys.argv[5]) # number of data points in test set to evaluate the generalization performance 
    nLayerMax = int(sys.argv[6]) # maximum number of iterations of base kernel function
    ARD_scaling = (sys.argv[7]=='True' ) # flag to indicate whether we should perform automatic relevance determination
    nTrainStep = int(sys.argv[8]) # Increment step for training set size
    nSample = int(sys.argv[9]) # Number of simulations to run per configuration
    inputDataPath = str(sys.argv[10]) # Path to input data

    # write command line arguments to logfile
    if( os.path.isdir( './Figure5' ) == False ):     
        os.mkdir( './Figure5' )

    logFilename = 'GeneralizationErrorRun-' + time.strftime("%Y%m%d-%H%M%S") + '-Log.txt'
    f1 = open( './Figure5/' + logFilename, 'w' )
    
    f1.write( "numpySeed = " + str( numpySeed ) + "\n" ) 
    f1.write( "kernelOrder = " + str(kernelOrder) + "\n" )
    f1.write( "kernelAntiSymmetric = " + str(kernelAntiSymmetric) + "\n" )
    f1.write( "nTrainLinear = " + str(nTrainLinear) + "\n" )
    f1.write( "nTest = " + str(nTest) + "\n" ) 
    f1.write( "nLayerMax = " + str(nLayerMax) + "\n" )
    f1.write( "ARD_scaling = " + str(ARD_scaling) + "\n" )
    f1.write( "nTrainStep = " + str(nTrainStep) + "\n" )
    f1.write( "nSample = " + str(nSample) + "\n" )
    f1.write( "inputDataPath = " + inputDataPath + "\n" ) 
    f1.flush()

    # read in data
    raw_data = np.genfromtxt(inputDataPath,
                             delimiter=",",
                             skip_header=1 )

    ### impute missing values
    # get column means
    col_mean = np.nanmean(raw_data,axis=0)

    #Find indicies that you need to replace
    inds = np.where(np.isnan(raw_data))

    #Place column means in the indices. Align the arrays using take
    raw_data[inds]=np.take(col_mean,inds[1])


    ### set seed
    np.random.seed( seed=numpySeed )

    ### divide data into 
    # i. data for optimizing linear kernel
    # ii. data for training 
    # iii. data for testing

    nTotal = raw_data.shape[0]  # total number of data points available
    pDim = raw_data.shape[1] - 2 # number of features
    nTrainNonLinear_All = nTotal - nTrainLinear - nTest # remaining number of data points that can be used to form training sets

    # generate random permutation of all the data points 
    sTmp = np.random.choice( nTotal, size=nTotal, replace=False )

    # extract training set for use with linear kernel to determin ARD weights 
    xTrainLinear = raw_data[sTmp[0:nTrainLinear], 0:pDim]
    xTrainLinear.resize( nTrainLinear, pDim )
    yTrainLinear = raw_data[sTmp[0:nTrainLinear], 1+pDim]
    yTrainLinear.shape = (nTrainLinear, 1)

    # extract test data set 
    xTest = raw_data[sTmp[nTrainLinear:(nTrainLinear+nTest)], 0:pDim]
    yTest = raw_data[sTmp[nTrainLinear:(nTrainLinear+nTest)], 1+pDim]

    # extract data points for training with neural network covariance functions
    xTrainNonLinear_All = raw_data[sTmp[(nTrainLinear+nTest)::], 0:pDim]
    yTrainNonLinear_All = raw_data[sTmp[(nTrainLinear+nTest)::], 1+pDim]


    ### standardize data based upon characteristics of linear training data
    xTrainLinear_mean = np.mean(xTrainLinear, axis=0)
    xTrainLinear_sd = np.std(xTrainLinear, axis=0)
    yTrainLinear_mean = np.mean( yTrainLinear )
    yTrainLinear_sd = np.std( yTrainLinear )

    for i in range( pDim ):
        xTrainLinear[ :, i] = (xTrainLinear[ :, i] - xTrainLinear_mean[i])/ xTrainLinear_sd[i]
        xTrainNonLinear_All[ :, i] = (xTrainNonLinear_All[ :, i] - xTrainLinear_mean[i])/ xTrainLinear_sd[i]
        xTest[ :, i] = (xTest[ :, i] - xTrainLinear_mean[i])/ xTrainLinear_sd[i]

    yTrainLinear = (yTrainLinear - yTrainLinear_mean) / yTrainLinear_sd
    yTrainNonLinear_All = (yTrainNonLinear_All - yTrainLinear_mean) / yTrainLinear_sd
    yTest = (yTest - yTrainLinear_mean) / yTrainLinear_sd

    ## rescale so that input vectors for linear kernel training have unit length  
    for i in range( nTrainLinear ):
        xTrainLinear_norm_tmp = np.dot( xTrainLinear[i, :], xTrainLinear[i, :] )
        xTrainLinear[i, :] /= np.sqrt( xTrainLinear_norm_tmp )    

    ### instantiate linear kernel and estimate scale factors using ARD
    linearKernel = GPy.kern.Linear( input_dim = pDim, ARD=ARD_scaling )
    linearGPModel = GPy.models.GPRegression(xTrainLinear,yTrainLinear, kernel=linearKernel)
    linearGPModel.optimize_restarts(num_restarts=10, messages=True)

    sumVar = 0.0
    for i in range( pDim ):
        sumVar += linearKernel.input_sensitivity()[i] * np.power( np.std( xTrainLinear[:, i ]), 2.0 )
    
    sumVar += linearGPModel.likelihood.variance[0]

    linearModelPredictions = linearGPModel.predict( xTest / np.sqrt( pDim ) )
    linearModelTestError = mean_squared_error( yTest, linearModelPredictions[0] )
    
    # write linear kernel test error to logfile
    f1.write( "Linear kernel RMSE on test set = " + str(np.sqrt(linearModelTestError)) + "\n" )
    f1.close()

    ### calculate variable importance weights
    var_wts =np.ones( pDim )
    var_wts_norm  = np.sum( np.abs( linearKernel.input_sensitivity() ) )
    for i in range( pDim ):
        var_wts[i] = np.abs( linearKernel.input_sensitivity()[i] ) / var_wts_norm
    
        if( ARD_scaling == False ):
            var_wts[i] = np.sqrt( 1.0/float(pDim) )

    # rescale input vectors of test and non-linear training data sets.
    # rescale so that they have expected unit length. Scaling may be non-isotropic if
    # ARD variable importance weights are used
    for i in range( pDim ):
        xTrainNonLinear_All[ :, i] = var_wts[i] * xTrainNonLinear_All[ :, i]
        xTest[ :, i] = var_wts[i] * xTest[ :, i]

    # set each input vector to have unit length
    for i in range( nTest ):
        xTest_norm_tmp = np.dot( xTest[i, :], xTest[i, :] )
        xTest[i, :] /= np.sqrt( xTest_norm_tmp )
    
    for i in range( nTrainNonLinear_All ):
        xTrainNonLinear_All_norm_tmp = np.dot( xTrainNonLinear_All[i, :], xTrainNonLinear_All[i, :] )
        xTrainNonLinear_All[i, :] /= np.sqrt( xTrainNonLinear_All_norm_tmp )


    noiseGP_linear = linearGPModel.likelihood.variance[0]

    # define GP covariance functions
    linearKernel = kpu.getLinearKernel() 
    baseKernel = kpu.getBaseKernelChoice( q=kernelOrder, antiSymmetric=kernelAntiSymmetric )
    kernelList = []
    kernelList.append( linearKernel )
    for l in range( nLayerMax ):
        kernelTmp = kpu.calcDeepKernelFunc( baseKernel, l+1 )    
        kernelList.append( kernelTmp )
    
    nKernels = len( kernelList )

    # set noise level for covariance kernels equal to the estimated noise when using 
    # a linear kernel on the initial training set
    noiseGP = noiseGP_linear


    # set sequence of training set sizes
    nTrain_nStep = int( np.floor( nTrainNonLinear_All / nTrainStep ) )
    nTrainNonLinear_seq = np.arange( nTrainStep, (nTrainStep * nTrain_nStep)+1, nTrainStep )
 
    # open results file
    resultsFilename = 'GeneralizationErrorRun-' + time.strftime("%Y%m%d-%H%M%S") + '-Results.txt'
    f2 = open( './Figure5/' + resultsFilename, 'w' )

    for nTrainNonLinear_idx in range( len(nTrainNonLinear_seq ) ):
        print( "Generating results for sample " + str(nTrainNonLinear_idx) + "\n" )
        nTrainNonLinear = nTrainNonLinear_seq[nTrainNonLinear_idx]
        for s in range( nSample ):
            sTmp = np.random.choice( nTrainNonLinear_All, size=nTrainNonLinear, replace=False )
            xTrain = xTrainNonLinear_All[sTmp, ]
            yTrain = yTrainNonLinear_All[sTmp]
    
            for l in range(nKernels):
                testTmpGP = gpu.trainAndPredictGP( xTrain, yTrain, xTest, yTest, kernelList[l], noiseGP, 1.0 - noiseGP )
                f2.write( str( nTrainNonLinear ) + "\t" + str( s ) + "\t" + str(l) + "\t" + str(testTmpGP) + "\n" )
            f2.flush()
            
    f2.close()
    
if __name__ == '__main__':
    main()     