# -*- coding: utf-8 -*-

import os
import math
import numpy as np
from scipy.special import iv, gammaln

#################### Function definitions #############################
          
def calcRBFlogEigenvalue( n, p, width ):
    """
    Function to calculate eigenvalues of an RBF kernel, up to a
    specified order
    
    :param n: Order of eigenvalues required
    :param p: Data dimension
    :param width: RBF kernel width
    :return: Returns log of eigenvalue
    """
    
    log_eigenvalue = np.zeros(n)

    for i in range(n):
        log_eigenvalue[i] = np.log(iv(float(i+1) + 0.5*float(p) - 1.0, 1/math.pow(width, 2.0) ))
        log_eigenvalue[i] -= 1.0/math.pow( width, 2.0 )
        log_eigenvalue[i] += 0.5*float(p-2) * (np.log( 2.0 ) + 2.0*np.log( width ))
        log_eigenvalue[i] += gammaln( 0.5*float(p) )
    
    return( log_eigenvalue )
    


def getBaseDerivativeAtZero_ChoSaulq0( n, antiSymmetric, logValue=False ): 
    """
    Function to compute a single specified derivative at z=0 for lowest
    order kernel of Cho and Saul. Note: z = cos(theta)
    The zeroth derivative is defined as the function value itself
    
    :param n: Order of the derivative required
    :param antiSymmetric: Boolean flag to indicate if the kernel function should be made antisymmetric, i.e. have domain [-1, 1] , range [-1, 1], and f(-z) = -f(z)
    :param logValue: If true returns log of derivative
    :return: Returns derivative at z=0
    """

    # Test if order is even
    if( n%2 == 0 ):
        return( 0.0 )
    else:
        i = int( np.floor( 0.5 * float(n-1) ) ) 
        derivative_atZero = 0.0
        derivative_atZero = gammaln( 2.0*i + 1.0) - 2.0*gammaln( 1.0*i + 1.0 )
        derivative_atZero = derivative_atZero + gammaln( 2.0 * i + 2.0 )
        derivative_atZero = derivative_atZero - (float(i)*np.log(4.0)) - np.log( 2.0 * i + 1.0 )
        derivative_atZero += np.log( 2.0/math.pi )
        
        if( antiSymmetric == False ):
            derivative_atZero += np.log( 0.5 )
            if( n==0 ):
                derivative_atZero = np.log( 0.5 )
                
        if( logValue == False ):
            derivative_atZero = np.exp( derivative_atZero )
    
        return( derivative_atZero )

    
    
def calcExactEigenvalue_ChoSaulq0(n, p, ftol, maxIter):
    """
    Function to compute exact eigenvalue of lowest order kernel of 
    Cho and Saul. This is done by summation of contributions from 
    Taylor expansion of the kernel function around z=0 (z=cos(theta)).
    Summation is continued until convergence tolerance criterion is met.
    
    :param n: Order of eigenvalue required
    :param p: Data dimension
    :param ftol: Convergence criterion. This is the tolerance in the fractional change in the running summation used to define the eigenvalue
    :param maxIter: Maximum number of terms in Taylor expansion to be included
    :return: Returns log of eigenvalue
    """
    
    iter = 1
    m = n
    sum_tmp = 0.0
    fracChange = 2*ftol
    while( (fracChange > ftol) & (iter <= maxIter) ):
        if( m%2 == 1 ):
            x1_tmp = gammaln(0.5*float(p)) + gammaln( 0.5*(m-n+1) ) - gammaln( 0.5*(m+n+p) ) - gammaln( m-n+1.0 )
            x2_tmp = getBaseDerivativeAtZero_ChoSaulq0( m, antiSymmetric=False, logValue=True )
            delta_tmp = np.exp( x1_tmp + x2_tmp )
        
            if( iter > 5 ):
                fracChange = np.abs( delta_tmp / sum_tmp ) # only calculate the fractional change once we have summed at least 5 terms

            sum_tmp += delta_tmp
            iter += 1
        m += 1
        
    logEigenvalue = -0.5*np.log( np.pi ) - (float(n)*np.log(2.0)) + np.log( sum_tmp )
    return( logEigenvalue )

    
#################### End - Function definitions #######################
    
def main():    
    n=15 # number of eigenvalues
    pSeq=np.arange(10, 210, 10) # data dimension (number of features)

    ### Compare exact RBF kernel eigenvalues and exact application of general 
    ### asymptotic approximation formula for eigenvalues
    width = 1.0 # width of RBF kernel 

    # get log of eigenvalues
    logPopEigenvalues = np.zeros( (len(pSeq), n) )
    for i in range(len(pSeq)):
        logPopEigenvalues[i, ] = calcRBFlogEigenvalue( n, pSeq[i], width )

    logPopEigenvalues_asymptotic = np.zeros( (len(pSeq), n) )
    for i in range( n ):
        for j in range(len(pSeq)):
            logPopEigenvalues_asymptotic[j,i] = -float( i+1 ) * ( 2.0 * math.log( width ) + math.log(pSeq[j]) ) - (1.0/(width*width))
          
    # write results to file
    if( os.path.isdir( './Figure1' ) == False ):     
        os.mkdir( './Figure1')

    # log eigenvalues
    f = open( './Figure1/RBFEigenvalues_Exact_vs_Asymptotic.txt', 'w+' )

    for j in range(len(pSeq)):
        f.write( str(pSeq[j]) + "\t" )
        for i in range( n-1 ):
            f.write( str(logPopEigenvalues[j, i]) + "\t" + str(logPopEigenvalues_asymptotic[j, i]) + "\t" )
        f.write( str(logPopEigenvalues[j, n-1]) + "\t" + str(logPopEigenvalues_asymptotic[j, n-1]) + "\n" )
    f.close()   

    # differences in log eigenvalues
    f = open( './Figure1/RBFEigenvalues_Exact_vs_Asymptotic_Diffs.txt', 'w+' )

    for j in range(len(pSeq)):
        f.write( str(pSeq[j]) + "\t" )
        for i in range( n-1 ):
            f.write( str(-logPopEigenvalues[j, i] + logPopEigenvalues_asymptotic[j, i]) + "\t" )
        f.write( str(-logPopEigenvalues[j, n-1] + logPopEigenvalues_asymptotic[j, n-1]) + "\n" )
    f.close()         
          
         
    ### Compare exact Cho + Saul q=0 kernel eigenvalues and exact application of general 
    ### asymptotic approximation formula for eigenvalues

    n=15 # number of eigenvalues
    pSeq=np.arange(10, 210, 10) # data dimension (number of features) 

    # get log of eigenvalues
    logPopEigenvalues = np.zeros( (len(pSeq), n) )
    for i in range(len(pSeq)):
        for j in range( n ):
            logPopEigenvalues[i, j] = calcExactEigenvalue_ChoSaulq0(2*j+1, pSeq[i], 1.0e-3, 500 )

    logPopEigenvalues_asymptotic = np.zeros( (len(pSeq), n) )
    for i in range( n ):
        for j in range(len(pSeq)):
            logPopEigenvalues_asymptotic[j,i] = getBaseDerivativeAtZero_ChoSaulq0( 2*i+1, antiSymmetric=False, logValue=True )
            logPopEigenvalues_asymptotic[j,i] += -float( 2*i+1 )*math.log(pSeq[j])
          
    # write results to file

    # log eigenvalues
    f = open( './Figure1/ChoSaulq0Eigenvalues_Exact_vs_Asymptotic.txt', 'w+' )

    for j in range(len(pSeq)):
        f.write( str(pSeq[j]) + "\t" )
        for i in range( n-1 ):
             f.write( str(logPopEigenvalues[j, i]) + "\t" + str(logPopEigenvalues_asymptotic[j, i]) + "\t" )
        f.write( str(logPopEigenvalues[j, n-1]) + "\t" + str(logPopEigenvalues_asymptotic[j, n-1]) + "\n" )
    f.close()   

    # differences in log eigenvalues
    f = open( './Figure1/ChoSaulq0Eigenvalues_Exact_vs_Asymptotic_Diffs.txt', 'w+' )

    for j in range(len(pSeq)):
        f.write( str(pSeq[j]) + "\t" )
        for i in range( n-1 ):
            f.write( str(-logPopEigenvalues[j, i] + logPopEigenvalues_asymptotic[j, i]) + "\t" )
        f.write( str(-logPopEigenvalues[j, n-1] + logPopEigenvalues_asymptotic[j, n-1]) + "\n" )
    f.close()
 

if __name__ == '__main__':
    main()
