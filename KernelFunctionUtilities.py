import numpy as np
import math

def getLinearKernel():
    """
    Function to return a linear kernel function
    
    :return: Returns linear kernel function
    """
    def linearKernel( x ):
        return( x )
    
    return( linearKernel )

def getBaseKernel_q0( antiSymmetric=False ):
    """
    Function to return normalized version Cho+Saul q=0 kernel function
    
    :param antiSymmetric:Boolean flag. If True then anti-symmetric form of 
    kernel is returned
    :return: Returns q=0 kernel function
    """
    def baseKernel( x ):
        theta = math.acos( x )
        kernelTmp = 1.0 - (theta/math.pi)

        if( antiSymmetric ):      
            theta_anti = math.acos( -x )
            kernelTmp_anti = math.pi - theta_anti
            kernelTmp_anti = kernelTmp_anti / math.pi
            
            kernelTmp -= kernelTmp_anti
            
        return( kernelTmp )
    return( baseKernel )    
    

def getBaseKernel_q1( antiSymmetric=False ):
    """
    Function to return normalized version of Cho+Saul q=1 kernel function
    
    :param antiSymmetric:Boolean flag. If True then anti-symmetric form of 
    kernel is returned
    :return: Returns q=1 kernel function
    """
    def baseKernel( x ):
        theta = math.acos( x )
        kernelTmp = math.sin( theta ) + ((math.pi - theta)*math.cos( theta ))
        kernelTmp = kernelTmp / math.pi
               
        if( antiSymmetric ):      
            theta_anti = math.acos( -x )            
            kernelTmp_anti = math.sin( theta_anti ) + ((math.pi - theta_anti)*math.cos( theta_anti ))
            kernelTmp_anti = kernelTmp_anti / math.pi
            
            kernelTmp -= kernelTmp_anti
            
        return( kernelTmp )
    return( baseKernel ) 
    
def getBaseKernel_q2( antiSymmetric=False ):
    """
    Function to return normalized version of Cho+Saul q=2 kernel function
    
    :param antiSymmetric:Boolean flag. If True then anti-symmetric form of 
    kernel is returned
    :return: Returns q=2 kernel function
    """
    def baseKernel( x ):
        theta = math.acos( x )
        kernelTmp = (3.0*math.sin( theta )*math.cos(theta)) +((math.pi - theta)*(1.0+2.0*np.power(math.cos(theta), 2.0)))
        kernelTmp = kernelTmp / 3.0 
        kernelTmp = kernelTmp / math.pi
               
        if( antiSymmetric ):      
            theta_anti = math.acos( -x )       
            kernelTmp_anti = (3.0*math.sin( theta_anti )*math.cos(theta_anti)) +((math.pi - theta_anti)*(1.0+2.0*np.power(math.cos(theta_anti), 2.0)))
            kernelTmp_anti = kernelTmp_anti / 3.0
            kernelTmp_anti = kernelTmp_anti / math.pi
            
            kernelTmp -= kernelTmp_anti
            
        return( kernelTmp )
    return( baseKernel )
  
def getBaseKernel_q3( antiSymmetric=False ):
    """
    Function to return normalized version of Cho+Saul q=3 kernel function
    
    :param antiSymmetric:Boolean flag. If True then anti-symmetric form of 
    kernel is returned
    :return: Returns q=3 kernel function
    """
    def baseKernel( x ):
        theta = math.acos( x )
        sin_theta = math.sin( theta )
        
        kernelTmp = (4.0*np.power( sin_theta, 3.0 )) + (15.0*np.power(x, 2.0)*sin_theta)
        kernelTmp += ( (math.pi - theta)*( (4.0*x*np.power(sin_theta, 2.0)) + (5.0*x) + (10.0*np.power(x, 3.0)) ) )
        kernelTmp = kernelTmp / 15.0  
        kernelTmp = kernelTmp / math.pi
               
        if( antiSymmetric ):      
            theta_anti = math.acos( -x ) 
            sin_theta_anti = math.sin( theta_anti )
            
            kernelTmp_anti = (4.0*np.power( sin_theta_anti, 3.0 )) + (15.0*np.power(-x, 2.0)*sin_theta_anti)
            kernelTmp_anti += ( (math.pi - theta_anti)*( (4.0*-x*np.power(sin_theta_anti, 2.0)) + (5.0*-x) + (10.0*np.power(-x, 3.0)) ) )
            kernelTmp_anti = kernelTmp_anti / 15.0
            kernelTmp_anti = kernelTmp_anti / math.pi
            kernelTmp -= kernelTmp_anti
            
        return( kernelTmp )
    return( baseKernel )
    
def getBaseKernel_q4( antiSymmetric=False ):
    """
    Function to return normalized version of Cho+Saul q=4 kernel function
    
    :param antiSymmetric:Boolean flag. If True then anti-symmetric form of 
    kernel is returned
    :return: Returns q=4 kernel function
    """
    def baseKernel( x ):
        theta = math.acos( x )
        sin_theta = math.sin( theta )
        
        kernelTmp = (15.0*sin_theta  - 30.0*np.power( sin_theta, 3.0 )) + (12.0*np.power(sin_theta, 2.0)*x)
        kernelTmp -= (9*sin_theta*x) + (6.0*sin_theta*np.power( x, 3.0))
        kernelTmp -= ( (math.pi - theta)*( (9.0*np.power(sin_theta, 2.0)) + (18.0*np.power(x*sin_theta, 2.0)) ) )
        kernelTmp += (7.0*(math.pi-theta)*x*(9.0 + (6.0*np.power(x, 2.0))))
        kernelTmp += (7.0*(15.0*sin_theta*x) + (4.0*np.power(sin_theta, 3.0)))
        kernelTmp = kernelTmp / 105.0  
        kernelTmp = kernelTmp / math.pi
               
        if( antiSymmetric ):      
            theta_anti = math.acos( -x )
            sin_theta_anti = math.sin( theta_anti )
            kernelTmp_anti = (15.0*sin_theta_anti  - 30.0*np.power( sin_theta_anti, 3.0 )) + (12.0*np.power(sin_theta_anti, 2.0)*-x)
            kernelTmp_anti -= (9*sin_theta_anti*-x) + (6.0*sin_theta_anti*np.power( -x, 3.0))
            kernelTmp_anti -= ( (math.pi - theta_anti)*( (9.0*np.power(sin_theta_anti, 2.0)) + (18.0*np.power(-x*sin_theta_anti, 2.0)) ) )
            kernelTmp_anti += (7.0*(math.pi-theta_anti)*x*(9.0 + (6.0*np.power(-x, 2.0))))
            kernelTmp_anti += (7.0*(15.0*sin_theta_anti*-x) + (4.0*np.power(sin_theta_anti, 3.0)))
            kernelTmp_anti = kernelTmp_anti / 105.0
            kernelTmp_anti = kernelTmp_anti / math.pi
            kernelTmp -= kernelTmp_anti
            
        return( kernelTmp )
    return( baseKernel )
    
def getBaseKernelChoice( q, antiSymmetric ):
    """
    Function to return selected Cho+Saul kernel function
    
    :param q: Order of kernel function required
    :param antiSymmetric:Boolean flag. If True then anti-symmetric form of 
    kernel is returned
    :return: Returns kernel function
    
    """
    
    baseKernel = getBaseKernel_q0( antiSymmetric ) 
    if( q==1 ):
        baseKernel = getBaseKernel_q1( antiSymmetric )
    elif( q==2 ):
        baseKernel = getBaseKernel_q2( antiSymmetric )
    elif( q==3 ):
        baseKernel = getBaseKernel_q3( antiSymmetric )
    elif( q==4 ):
        baseKernel = getBaseKernel_q4( antiSymmetric )

    return( baseKernel )
    
def getPolynomialKernel( linearCoefficient, balancingPower ):
    """
    Function to return a polynomial kernel consisting of a linear term and 
    a term of form t^balancingPower. The coefficient for the balancing power 
    is chosen so that k(t=1) = 1
    
    :param linearCoefficient: Coefficient for the linear term in the kernel
    :param balancingPower: Exponent for the balancing non-linear term in the 
    kernel
    :return: Returns kernel function
    """
    
    def baseKernel( x ):
        kernelValue = (linearCoefficient * x)
        kernelValue += (( 1.0 - linearCoefficient ) * (np.power( x, balancingPower )))

        return( kernelValue )
    return( baseKernel )
    
   
def calcDeepKernelFunc( baseKernel, l ):
    """
    Function to construct iterated kernel function from base kernel
    
    :param baseKernel: Base kernel function
    :param l: Order of iteration of base kernel
    :return: Returns function to calculate iterated kernel
    """
   
    def finalLayerFunc( x ):
        # get base function value
        lCurrent = 1
        currentLayerFuncValue = baseKernel(x)
       
        while( lCurrent < l ):
            currentLayerFuncValueTmp = baseKernel( currentLayerFuncValue )
            currentLayerFuncValue = currentLayerFuncValueTmp
            lCurrent = lCurrent + 1
           
        return( currentLayerFuncValue )
    return( finalLayerFunc )