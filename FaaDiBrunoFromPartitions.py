## Original author: David Hoyle (david.hoyle@hoyleanalytics.org)
## 
## Date: 2019-05-20
##
## Licence: CC-BY

from sympy.utilities.iterables import partitions
from scipy.special import gammaln 
import numpy as np


def calcHardyRamanujanApprox( n ):
    """
    Function to calculate the Hardy-Ramanujan asymptotic approximation
    to the number of partitions of n.
    Function returns the log of the approximate number of partitions

    :param n: Integer for which number of partitions is required
    :return: Returns log of approximate number of partitions of n
    """

    log_nPartitions = np.pi * np.sqrt( 2.0 * float( n )/ 3.0 ) - np.log( 4.0 * float( n ) * np.sqrt( 3.0 ) )
	
    return( log_nPartitions )


def calcFaaDiBrunoFromPartitionsFromFunction( baseDerivativeFunction, l, n, x, thresholdForExp=30.0, verbose=True):
    """
    Function to calculate derivatives of iterated base function
    
    :param baseDerivativeFunction: Function that returns derivaties of base function
    :param l: Order of function iteration
    :param n: Highest order of derivative to be evaluated for iterated function 
    :param x: Point at which derivatives are to be evaluated
    :param thresholdForExp: Threshold beyond which differences in logarithms will be capped
    :param verbose: Boolean flag. If true then print messages
    :return: Dictionary containing logarithms and signs of derivates of iterated base function
    
    """
    
    # set up arrays for holding derivatives (on log scale)
    # of the base function, the current iteration and next iteration  
    
    evaluationPoint = x
    
    baseDerivativesLog = np.zeros( n )
    baseDerivativesSign = np.ones( n )
    currentDerivativesLog = np.zeros( n )
    currentDerivativesSign = np.ones( n )
    for k in range( n ):		
        baseDerivativeTmp = baseDerivativeFunction( x, k+1 )
        if( np.abs( baseDerivativeTmp ) > 1.0e-12 ):
            baseDerivativesLog[k] = np.log( np.abs(baseDerivativeTmp) )
            baseDerivativesSign[k] = np.sign( baseDerivativeTmp)

            # get current derivatives
            currentDerivativesLog[k] = np.log( np.abs( baseDerivativeTmp ) )
            currentDerivativesSign[k] = np.sign( baseDerivativeTmp )
        else:
            baseDerivativesLog[k] = float( '-Inf' )
            baseDerivativesSign[k] = 1.0

            currentDerivativesLog[k] = float( '-Inf' )
            currentDerivativesSign[k] = 1.0
            


    nextDerivativesLog = np.zeros( n )
    nextDerivativesSign = np.ones( n )


    # initialize array for holding derivatives at each iteration
    iteratedFunctionDerivativesLog = np.zeros( (l, n) )
    iteratedFunctionDerivativesSign = np.ones( (l, n) )

    # store base derivatives in first row of array
    iteratedFunctionDerivativesLog[0, :] = baseDerivativesLog
    iteratedFunctionDerivativesSign[0, :] = baseDerivativesSign

    # set number of function iterations
    nIterations = l-1

    # if we need to iterate then do so
    if( nIterations > 0 ): 


        # evaluate approximate number of paritions required
        log_nPartitions = calcHardyRamanujanApprox( n )

        if( verbose==True ):
            print( "You have requested evaluation up to derivative " + str(n) )
            if( log_nPartitions > np.log( 1.0e6 ) ):
                print( "Warning: There are approximately " + str(int(np.round(np.exp(log_nPartitions)))) + " partitions of " + str(n) )
                print( "Warning: Evaluation will be costly both in terms of memory and run-time" )

        # store partitions
        pStore = {}
        for k in range( n ):
            # get partition iterator
            pIterator = partitions(k+1)
            pStore[k] = [p.copy() for p in pIterator]

        # loop over function iterations    
        for iteration in range( nIterations ):
            evaluationPoint = baseDerivativeFunction( evaluationPoint, 0 )

            if( verbose==True ):
                print( "Evaluating derivatives for function iteration " + str(iteration+1)  )

            for k in range( n ):
                faaSumLog = float( '-Inf' )
                faaSumSign = 1
			
                # get partitions
                partitionsK = pStore[k]
                for pidx in range( len(partitionsK) ):
                    p = partitionsK[pidx]
                    sumTmp = 0.0
                    sumMultiplicty = 0
                    parityTmp = 1
                    for i in p.keys():
                        value = float(i)
                        multiplicity = float( p[i] )
                        sumMultiplicty += p[i]
                        sumTmp += multiplicity * currentDerivativesLog[i-1]
                        sumTmp -= gammaln( multiplicity + 1.0 )
                        sumTmp -= multiplicity * gammaln( value + 1.0 )
                        parityTmp *= np.power( currentDerivativesSign[i-1], multiplicity )	

                    
                    #evaluationPointTmp = np.exp( currentDerivativesLog[0] ) * currentDerivativesSign[0]
                    baseDerivativeTmp = baseDerivativeFunction( evaluationPoint , sumMultiplicty )
                    if( np.abs( baseDerivativeTmp ) > 1.0e-12 ):
                        sumTmp += np.log( np.abs( baseDerivativeTmp ) )
                        parityTmp *= np.sign( baseDerivativeTmp )
                    else:
                        sumTmp = float( '-Inf' )
                        parityTmp = 1.0

                    # now update faaSum on log scale
                    if( sumTmp > float( '-Inf' ) ):
                        if( faaSumLog > float( '-Inf' ) ):
                            diffLog = sumTmp - faaSumLog
                            if( np.abs(diffLog) <= thresholdForExp ):
                                if( diffLog >= 0.0 ):
                                    faaSumLog = sumTmp
                                    faaSumLog += np.log( 1.0 + (float(parityTmp*faaSumSign) * np.exp( -diffLog )) )
                                    faaSumSign = parityTmp
                                else:
                                    faaSumLog += np.log( 1.0 + (float(parityTmp*faaSumSign) * np.exp( diffLog )) )
                            else:
                                if( diffLog > thresholdForExp ):
                            	    faaSumLog = sumTmp
                            	    faaSumSign = parityTmp
                        else:
                            faaSumLog = sumTmp
                            faaSumSign = parityTmp

                nextDerivativesLog[k] = faaSumLog + gammaln( float(k+2) )
                nextDerivativesSign[k] = faaSumSign

            # update accounting for proceeding to next function iteration
            currentDerivativesLog[0:n] = nextDerivativesLog[0:n]
            currentDerivativesSign[0:n] = nextDerivativesSign[0:n]
            iteratedFunctionDerivativesLog[iteration+1, 0:n] = currentDerivativesLog[0:n]
            iteratedFunctionDerivativesSign[iteration+1, 0:n] = currentDerivativesSign[0:n]

    return( {'logDerivatives':iteratedFunctionDerivativesLog, 'signDerivatives':iteratedFunctionDerivativesSign} )
