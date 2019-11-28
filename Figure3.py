import numpy as np
import os
import KernelFunctionUtilities as kpu
import GaussianProcessUtilities as gpu

## Example usage: python3 Figure3.py 42 

def main():
    
    # initialize random number generator
    numpySeed = int( sys.argv[1] ) # seed for random number generator 
    np.random.seed( seed = numpySeed )
    
    # set simulation parameters
    pDim = 500 # data dimension
    nLayer_teacher_max = 5 # highest order of iteration of teacher base kernel function
    sigma2_teacher = 0.1 # variance of noise in teacher data
    sigma2_student = sigma2_teacher # variance of noise in student data
    nSample = 1000 # number of simulation samples for each setting

    nLayerMax = 4*nLayer_teacher_max # maximum number of student layers
    lmult_lower = [0, 0, 0, 0, 0] # multiplier for lower value of student layer size
    lmult_higher = [2, 2, 2, 2, 2] # multiplier for upper value of student layer size
    trainingSetSize = np.arange( 20, 200, 20 ) # sequence of training set sizes

    # base kernels
    base_kernel_teacher = kpu.getBaseKernelChoice( q=0, antiSymmetric=True )
    base_kernel_student = kpu.getBaseKernelChoice( q=0, antiSymmetric=True )
    
    # construct teacher kernel functions
    kernel_func_teacher = []
    for l in range( nLayer_teacher_max ):
        kernel_func_teacher_tmp = kpu.calcDeepKernelFunc( base_kernel_teacher, l+1 )    
        kernel_func_teacher.append( kernel_func_teacher_tmp )

    # construct student kernel functions
    kernel_func_student = []
    for l in range( nLayerMax ):
        kernel_func_student_tmp = kpu.calcDeepKernelFunc( base_kernel_student, l+1 )    
        kernel_func_student.append( kernel_func_student_tmp )

    # open files
    if( os.path.isdir( './Figure3' ) == False ):     
        os.mkdir( './Figure3')
    
    f1 = open( './Figure3/StudentGP_TestErrorProfile.txt', 'w' )
    f2 = open( './Figure3/TeacherGP_TestErrorProfile.txt', 'w' )

    for setSize_idx in range( len( trainingSetSize ) ):
        for sample_idx in range( nSample ):
            # generate training data
            xTrainData = gpu.sampleInput( pDim, trainingSetSize[setSize_idx] )
            xTestData = gpu.sampleInput( pDim, 1 )
        
            print( trainingSetSize[setSize_idx], sample_idx )
        
            for nLayer_teacher in range( nLayer_teacher_max ):
                genErrorInstance_teacher = gpu.calcGenErrorInstance( xTrainData, xTestData, kernel_func_teacher[nLayer_teacher], sigma2_teacher, kernel_func_teacher[nLayer_teacher], sigma2_teacher )
                f2.write( str(nLayer_teacher) + "," + str( trainingSetSize[setSize_idx] ) + "," + str( sample_idx ) + "," + str( genErrorInstance_teacher ) + "\n" )
  
                for l in range( lmult_lower[nLayer_teacher]*nLayer_teacher_max, lmult_higher[nLayer_teacher]*nLayer_teacher_max ):
                    genErrorInstance_student = gpu.calcGenErrorInstance( xTrainData, xTestData, kernel_func_teacher[nLayer_teacher], sigma2_teacher, kernel_func_student[l], sigma2_student ) 
                    f1.write( str(nLayer_teacher) + "," + str(l) + "," + str( trainingSetSize[setSize_idx] ) + "," + str( sample_idx ) + "," + str( genErrorInstance_student ) + "\n" )
            
                f1.flush()          
                f2.flush()
            
    f1.close()  
    f2.close()  
    

if __name__ == '__main__':
    main()    


