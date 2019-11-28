import numpy as np
import math
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



def main():
    maxDerivativeOrder = 40 # number of derivatives to evaluate for each l
    nLayer = 40 # number of iterations of the kernel function
    antiSymmetric = False # whether we should us an antiSymmetric kernel
    pDim = 500 # data dimension

    # get function that calculates derivatives of base kernel
    ftmp = createDerivativeFunction(  tol=1.0e-10, nMinTerms = 20, antiSymmetric=antiSymmetric )

    # evaluate iterated derivatives using the Faa Di Bruno formula
    iteratedDerivatives_FromPartitions = faaDiBruno.calcFaaDiBrunoFromPartitionsFromFunction( ftmp, nLayer, maxDerivativeOrder, 0.0, 40.0 )

    # create directory and open file handle for writing results
    if( os.path.isdir( './Figure2' ) == False ):     
        os.mkdir( './Figure2')

    if( antiSymmetric==True ):
        f = open( './Figure2/FaaDiBruno_ChoSaulq0_AntiSym.csv', 'w+' )
    else:
        f = open( './Figure2/FaaDiBruno_ChoSaulq0_Sym.csv', 'w+' )

    
    # write header line
    f.write( "l" )
    for n in range( 1, maxDerivativeOrder ):
        writeOut=True
        if( (n%2==0) & (antiSymmetric==True) ):
            writeOut=False
        
        if( writeOut==True ):
            f.write( "," + "Eigenvalue_FaaDoBruno" + "_" + str( n ) + "," 
                    + "Eigenvalue_Approx" + "_" + str( n ) + "," 
                    + "Delta_" + str(n) + "," 
                    + "Eigenvalue_Approx2" + "_" + str( n ) + "," 
                    + "Delta2_" + str(n) )
    f.write( "\n" )

    # evaluate simple approximations to Faa Di Bruno formula      
    for l in range( 1, nLayer+1 ):
        f.write( str(l) )

        for n in range( 1, maxDerivativeOrder ):
            writeOut=True
            if( n%2==0 & antiSymmetric==True ):
                writeOut=False
            
            if( writeOut==True ):
                # calculate approximation
                approxPreFactor = 0.0
                approxOffset = 1
            
            
                if( antiSymmetric==True ):
                    evaluationPointTmp = 0.0
                    fixedPoint = 0.0
                    approxOffset = 3
                    a= 2.0
                elif( antiSymmetric == False ):
                    approxOffset = 3
                    evaluationPointTmp = 0.0
                    fixedPoint = 0.7898
                    a = 1.0
                
                approxPreFactor = iteratedDerivatives_FromPartitions['logDerivatives'][approxOffset-1, n-1]         
                approxScalingFactor2 = ftmp( fixedPoint, 1 )
                
                # calculate first form of approximation to Faa Di Bruno result
                simpleApprox = approxPreFactor - (float(n)*np.log( float( pDim ) ) )
                evaluationPointTmp = 0.0
                for k in range( l ):
                    xtmp = ftmp( evaluationPointTmp, 1 )
                    if( np.abs( xtmp ) > 1.0e-30 ):
                        simpleApprox += np.log( xtmp )
                    evaluationPointTmp = ftmp( evaluationPointTmp, 0 )
                
                evaluationPointTmp = 0.0    
                for k in range( approxOffset ):
                    xtmp = ftmp( evaluationPointTmp, 1)
                    if( np.abs( xtmp ) > 1.0e-30 ):
                        simpleApprox -= np.log( xtmp )
                    evaluationPointTmp = ftmp( evaluationPointTmp, 0 )
                    
                # calculate second form of approximation to Faa Di Bruno result
                simpleApprox2 = (float( l - approxOffset ) * np.log( approxScalingFactor2 ) ) + approxPreFactor - (float(n)*np.log( float( pDim ) ) )
                
                # calculate differences between Faa Di Bruno result and the 
                # two simple approximations
                delta = iteratedDerivatives_FromPartitions['logDerivatives'][l-1, n-1] -float(n)*np.log( float( pDim ) ) - simpleApprox
                delta2 = iteratedDerivatives_FromPartitions['logDerivatives'][l-1, n-1] -float(n)*np.log( float( pDim ) ) - simpleApprox2
                
                # write results to file
                f.write( "," + str(iteratedDerivatives_FromPartitions['logDerivatives'][l-1, n-1] -float(n)*np.log( float( pDim ) ) ) + "," 
                        + str( simpleApprox ) + "," 
                        + str(delta) + "," 
                        + str( simpleApprox2 ) + "," 
                        + str(delta2) )
        f.write( "\n" )

    f.close()        
        
if __name__ == '__main__':
    main()



