import numpy as np
import math
import time
import os
from scipy.special import gammaln
import FaaDiBrunoFromPartitions as faaDiBruno

def createDerivativeFunction(  tol, nMinTerms, antiSymmetric ):
    """
    Function to return a function that evaluates derivatives of the 
    Cho+Saul q=0 kernel. Note z=cos(theta).
    
    :param tol: The tolerance to which the derivative is should be 
    calculated
    :param nMinTerms: Minimum number of terms in Taylor expansion that 
    we should sum, irrespective of whether we have met the tolerance
    :param antiSymmetric: Boolean flag to indicate if the 
    antiSymmetric version of the q=0 Cho+Saul kernel function should 
    be used
    :return: Returns the function calcDerivative
    """
    
    a = 1.0
    if( antiSymmetric == True ):
        a = 2.0
        
    def calcDerivative( x, n ):
        """
        Function to evaluate the n^th derivative of the q=0 Cho+Saul 
        kernel function
        
        :param x: Point at which derivative is required
        :param n: Order of derivative required
        :return: Returns derivative
        """
        
        # initialize derivative
        derivative = 0.0

        if( n == 0 ): # n=0 is just evaluation of the function itself
            derivative = 1.0 - (a*math.acos(x)/math.pi)
        else: # evaluation for n >=1 is done by summation of Taylor expansion
              # of arcos(x)
            
            # initialize converged flag
            converged = False
            # calculate index of starting index in Taylor expansion for 
            # n^th derivative
            kStart = np.round( 0.5 * float( n - 1 ) + 0.25 )

            # sum terms in Taylor expansion whilst we have not met the
            # tolerance criterion and whilst we have not summed at least
            # nMinTerms terms
            k = kStart
            delta_log = gammaln( float(2*k + 2) ) - gammaln( float( 2*k + 1 - n + 1 ) )
            delta_log += gammaln( float(2*k + 1) ) - (2.0*gammaln(float(k+1) ))
            delta_log -= (float( k )*np.log( 4.0 ))
            delta_log -= np.log( 2.0*float(k) + 1.0 )
            delta = np.power( x, float(2*k+1-n) ) * np.exp( delta_log )
            delta *= a
            derivative += delta
            
            while( (converged==False) & ( k < kStart + nMinTerms ) ):
                k += 1
                delta_log = gammaln( float(2*k + 2) ) - gammaln( float( 2*k + 1 - n+ 1 ) )
                delta_log += gammaln( float(2*k + 1) ) - (2.0*gammaln(float(k+1) ))
                delta_log -= (float( k )*np.log( 4.0 ))
                delta_log -= np.log( 2.0*float(k) + 1.0 )
                delta = np.power( x, float(2*k+1-n) ) * np.exp( delta_log )
                delta *= a
                
                fracChange = abs( delta )
                derivative += delta

                if( fracChange < tol ):
                    converged = True
        
            derivative /= math.pi

        return derivative

    return calcDerivative




def calcApproxDerivativesAtZero_ChoSaulN0( n, l ):
    
    """
    Function to calculate leading order term in asymptotic 
    approximation (w.r.t. n) of the nth derivative of the 
    antis-symmetric Cho+Saul q=0 kernel
    
    :param n: Order of derivative required
    :param l: Number of layers (iterations of kernel)
    :return: Returns the leading order asymptotic approximation to 
    the nth derivative
    """

    chi_l = np.exp( float(l)*np.log(0.5) )
    q_l = 1.0 - chi_l
    a = 2.0 * math.sqrt( 2.0 ) / np.pi
    derivativesAtZero_log = np.log( 2.0/np.pi ) + np.log( np.sin( np.pi*chi_l ) ) + gammaln( chi_l )
    derivativesAtZero_log += gammaln(float(n)) -chi_l*np.log( float( n ) ) 
    derivativesAtZero_log += (2.0*q_l * np.log(a)) + np.log( chi_l )
    
    return( derivativesAtZero_log )
    
    
def main():
    # write command line arguments to logfile
    if( os.path.isdir( './Figure7' ) == False ):     
        os.mkdir( './Figure7' )
        
    maxDerivativeOrder = 50 # number of derivatives to evaluate for each l
    nLayer = 6 # number of iterations of the kernel function
    antiSymmetric = True # whether we should us an antiSymmetric kernel

    # get function that calculates derivatives of base kernel
    ftmp = createDerivativeFunction(  tol=1.0e-10, nMinTerms = maxDerivativeOrder + 10, antiSymmetric=antiSymmetric )

    # evaluate iterated derivatives using the Faa Di Bruno formula
    iteratedDerivatives_FromPartitions = faaDiBruno.calcFaaDiBrunoFromPartitionsFromFunction( ftmp, nLayer, maxDerivativeOrder, 0.0, 40.0 )
    
    
    derivatives_atZero_asymptotic_log = np.zeros( (nLayer, maxDerivativeOrder ) )
    for l in range( 1, nLayer ):
        for n in range( 1, maxDerivativeOrder+1 ):
            derivatives_atZero_asymptotic_log[l-1, n-1] = calcApproxDerivativesAtZero_ChoSaulN0( n, l )

    # open results file
    resultsFilename = 'logDerivatives_ChoSaul_q0-' + time.strftime("%Y%m%d-%H%M%S") + '-Results.csv'
    f1 = open( './Figure7/' + resultsFilename, 'w' )
    

    odd_indices = np.arange(1, maxDerivativeOrder, 2)
    odd_indices -= 1
    headerString = "l,"
    for i in range( len(odd_indices)-1 ):
        headerString += "SignOfDerivative_" + str( 1+odd_indices[i] ) + ","
        headerString += "ExactLogDerivative_" + str( 1+odd_indices[i] )+ ","
        headerString += "ApproxLogDerivative_" + str( 1+odd_indices[i] )+ ","
        
    i = len(odd_indices) - 1 
    headerString += "SignOfDerivative_" + str( 1+odd_indices[i] ) + ","
    headerString += "ExactLogDerivative_" + str( 1+odd_indices[i] ) + ","
    headerString += "ApproxLogDerivative_" + str( 1+odd_indices[i] ) + "\n"
    f1.write( headerString )

    for l in range( 1, nLayer+1 ):
        outString = str( l ) + ","
        for i in range( len(odd_indices)-1 ):
            outString += str( iteratedDerivatives_FromPartitions['signDerivatives'][l-1,odd_indices[i]]  ) + ","
            outString += str( iteratedDerivatives_FromPartitions['logDerivatives'][l-1,odd_indices[i]] ) + ","
            outString += str( derivatives_atZero_asymptotic_log[l-1, odd_indices[i]] ) + ","
        
        i = len(odd_indices) - 1 
        outString += str( iteratedDerivatives_FromPartitions['signDerivatives'][l-1,odd_indices[i]] ) + ","
        outString += str( iteratedDerivatives_FromPartitions['logDerivatives'][l-1,odd_indices[i]] ) + ","
        outString += str( derivatives_atZero_asymptotic_log[l-1, odd_indices[i]] ) + "\n"
        f1.write( outString )
        f1.flush()

    f1.close()    


if __name__ == '__main__':
    main() 
