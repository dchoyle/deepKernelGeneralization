import sys
import os
import time
import numpy as np
import KernelFunctionUtilities as kpu
import GaussianProcessUtilities as gpu

## Note: running times can be significant, so it advisable to run this code 
## as a background process on a separate machine. 
## Example usage: python3 Figure4a.py 1987 3 True 0 True 0.1 500 \
##                                   20 100 20 5 0 4 200


def main():
    # get simulation parameter settings
    numpySeed = int( sys.argv[1] ) # seed for random number generator 
    studentOrder = int(sys.argv[2]) # order of Cho+Saul student base kernel
    studentAntiSymmetric = (sys.argv[3]=='True' ) # flag to indicate if student base kernel is anti-symmetric
    teacherOrder = int( sys.argv[4] ) # order of Cho+Saul teacher base kernel
    teacherAntiSymmetric = ( sys.argv[5] == 'True' ) # flag to indicate if teacher base kernel is anti-symmetric
    sigma2_teacher = float( sys.argv[6] ) # noise level
    pDim = int( sys.argv[7] ) # input data dimension
    trainSetSize_lower = int( sys.argv[8] ) # smallest training set size
    trainSetSize_upper = int( sys.argv[9] ) # largest training set size
    trainSetSize_delta = int( sys.argv[10] ) # increment in training set size
    nLayer_teacher_max = int( sys.argv[11] ) # largest value for number of iterations of teacher kernel
    lmult_lower = int( sys.argv[12] ) # parameter controllest lower number of iterations of student base kernel
    lmult_higher = int( sys.argv[13] ) # parameter controllest upper number of iterations of student base kernel
    nSample = int( sys.argv[14] ) # number of simulation samples for each combination of parameter settings
    
    np.random.seed( seed = numpySeed )
    sigma2_student = sigma2_teacher # set noise level of student to match teacher
    lmult_lower_array = np.ones( nLayer_teacher_max, dtype = np.int8 ) * lmult_lower # array setting lowest value of student kernel iterations
    lmult_higher_array = np.ones( nLayer_teacher_max, dtype = np.int8 ) * lmult_higher # array setting highest value of student kernel iterations
    nLayerMax = np.amax( lmult_higher_array )*nLayer_teacher_max # maximum number of student layers needed
    trainingSetSize = np.arange( trainSetSize_lower, trainSetSize_upper, trainSetSize_delta ) # sequence of training set sizes
    
    # write command line arguments to logfile
    if( os.path.isdir( './Figure4a' ) == False ):     
        os.mkdir( './Figure4a' )
        
    logFilename = 'Figure4a-' + time.strftime("%Y%m%d-%H%M%S") + '-Log.txt'
    f0 = open( './Figure4a/' + logFilename, 'w' )
    f0.write( "numpySeed = " + str( numpySeed ) + "\n" ) 
    f0.write( "StudentKernelOrder = " + str(studentOrder) + "\n" )
    f0.write( "StudentKernelAntiSymmetric = " + str(studentAntiSymmetric) + "\n" )
    f0.write( "TeacherKernelOrder = " + str(teacherOrder) + "\n" )
    f0.write( "TeacherKernelAntiSymmetric = " + str(teacherAntiSymmetric) + "\n" )
    f0.write( "Noise teacher = " + str(sigma2_teacher) + "\n" )
    f0.write( "DataDimension = " + str(pDim) + "\n" ) 
    f0.write( "TrainSetSizeLower = " + str(trainSetSize_lower) + "\n" )
    f0.write( "TrainSetSizeUpper = " + str(trainSetSize_upper) + "\n" )
    f0.write( "TrainSetSizeDelta = " + str(trainSetSize_delta) + "\n" )
    f0.write( "nLayerTeacherMax = " + str(nLayer_teacher_max) + "\n" )
    f0.write( "l_MultiplierLower = " + str(lmult_lower) + "\n" )
    f0.write( "l_MultiplierHigher = " + str(lmult_higher) + "\n" )
    f0.write( "nSample = " + str(nSample) + "\n" )
    f0.flush()
    f0.close()

    # base kernels
    base_kernel_teacher = kpu.getBaseKernelChoice( q=teacherOrder, antiSymmetric=teacherAntiSymmetric )
    base_kernel_student = kpu.getBaseKernelChoice( q=studentOrder, antiSymmetric=studentAntiSymmetric )

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


    # construct filenames for results
    teacherFileName = 'TeacherGP_TestErrorProfile_TeacherN' + str(teacherOrder) + '_StudentN' + str(studentOrder) + '_sigma2=' + str(sigma2_teacher) + '_p=' + str(pDim) + '_'
    studentFileName = 'StudentGP_TestErrorProfile_TeacherN' + str(teacherOrder) + '_StudentN' + str(studentOrder) + '_sigma2=' + str(sigma2_student) + '_p=' + str(pDim) + '_'

    if( teacherAntiSymmetric==True ):
        teacherFileName = teacherFileName + 'AntiSym_'
        studentFileName =  studentFileName + 'AntiSym_'
    else:
        teacherFileName = teacherFileName + 'NotAntiSym_'
        studentFileName =  studentFileName + 'NotAntiSym_'

    if( studentAntiSymmetric==True ):
        teacherFileName = teacherFileName + 'AntiSym'
        studentFileName =  studentFileName + 'AntiSym'
    else:
        teacherFileName = teacherFileName + 'NotAntiSym'
        studentFileName = studentFileName + 'NotAntiSym'
        
    teacherFileName = teacherFileName + "_" + time.strftime("%Y%m%d-%H%M%S") + ".txt"
    studentFileName = studentFileName + "_" + time.strftime("%Y%m%d-%H%M%S") + ".txt"
        
    # open files
    f1 = open( './Figure4a/' + studentFileName, 'w' )
    f2 = open( './Figure4a/' + teacherFileName, 'w' )

    for setSize_idx in range( len( trainingSetSize ) ):
        for sample_idx in range( nSample ):
            # generate training data
            xTrainData = gpu.sampleInput( pDim, trainingSetSize[setSize_idx] )
            xTestData = gpu.sampleInput( pDim, 1 )
        
            print( trainingSetSize[setSize_idx], sample_idx )
        
            for nLayer_teacher in range( nLayer_teacher_max ):
                genErrorInstance_teacher = gpu.calcGenErrorInstance( xTrainData, xTestData, kernel_func_teacher[nLayer_teacher], sigma2_teacher, kernel_func_teacher[nLayer_teacher], sigma2_teacher )
                f2.write( str(nLayer_teacher) + "," + str( trainingSetSize[setSize_idx] ) + "," + str( sample_idx ) + "," + str( genErrorInstance_teacher ) + "\n" )
  
                for l in range( lmult_lower_array[nLayer_teacher]*nLayer_teacher_max, lmult_higher_array[nLayer_teacher]*nLayer_teacher_max ):
                    genErrorInstance_student = gpu.calcGenErrorInstance( xTrainData, xTestData, kernel_func_teacher[nLayer_teacher], sigma2_teacher, kernel_func_student[l], sigma2_student ) 
                    f1.write( str(nLayer_teacher) + "," + str(l) + "," + str( trainingSetSize[setSize_idx] ) + "," + str( sample_idx ) + "," + str( genErrorInstance_student ) + "\n" )
            
                f1.flush()          
                f2.flush()
            
    f1.close()  
    f2.close()

if __name__ == '__main__':
    main()    
