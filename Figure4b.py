import sys
import os
import numpy as np
import KernelFunctionUtilities as kpu
import GaussianProcessUtilities as gpu

## Note: running times can be significant, so it advisable to run this code 
## as a background process on a separate machine. 
## Example usage: python3 Figure4b.py 1987 3 True 0.1 0.4 10 True 3 0.1 500 \
##                                   20 100 20 10 200

def main():
    # get simulation parameter settings
    numpySeed = int( sys.argv[1] ) # seed for random number generator 
    studentOrder = int(sys.argv[2]) # order of Cho+Saul student base kernel
    studentAntiSymmetric = (sys.argv[3]=='True' ) # flag to indicate if student base kernel is anti-symmetric
    teacherLinearCoefficientStart = float( sys.argv[4] ) # starting value for teacher kernel linear coefficient
    teacherLinearCoefficientEnd = float( sys.argv[5] ) # final value for teacher kernel linear coefficient
    nTeacherSteps = int( sys.argv[6] ) # number of teacher kernel linear coefficient values
    logScale = ( sys.argv[7]=='True' ) # flag to indicate whether teacher kernel linear coefficient should vary on a logarithmic scale
    teacherBalancingPower = int( sys.argv[8] ) # balancing power of polynomial term in teacher kernel
    sigma2_teacher = float( sys.argv[9] ) # noise level of teacher and student GPs
    pDim = int( sys.argv[10] ) # input data dimension
    trainSetSize_lower = int( sys.argv[11] ) # smallest training set size
    trainSetSize_upper = int( sys.argv[12] ) # largest training set size
    trainSetSize_delta = int( sys.argv[13] ) # increment in training set size
    nLayerMax = int( sys.argv[14] ) # parameter controlling upper number of iterations of student base kernel
    nSample = int( sys.argv[15] ) # number of simulation samples for each combination of parameter settings

    np.random.seed( seed = numpySeed )
    sigma2_student = sigma2_teacher # set noise level of student to match teacher
    trainingSetSize = np.arange( trainSetSize_lower, trainSetSize_upper, trainSetSize_delta ) # sequence of training set sizes

    # student base kernel
    base_kernel_student = kpu.getBaseKernelChoice( q=studentOrder, antiSymmetric=studentAntiSymmetric )

    # construct student kernel functions
    kernel_func_student = []
    for l in range( nLayerMax ):
        kernel_func_student_tmp = kpu.calcDeepKernelFunc( base_kernel_student, l+1 )    
        kernel_func_student.append( kernel_func_student_tmp )

    # construct teacher kernels
    kernel_func_teacher = []
    teacherLinearCoefficient = []
    for l in range( nTeacherSteps+1 ):
        teacherLinearCoefficient_tmp = teacherLinearCoefficientStart + (float(l)*( teacherLinearCoefficientEnd - teacherLinearCoefficientStart ) / float( nTeacherSteps) )
        if( logScale == True ):
            teacherLinearCoefficient_log_tmp = np.log(teacherLinearCoefficientStart) + (float(l)*( np.log(teacherLinearCoefficientEnd) - np.log(teacherLinearCoefficientStart) ) / float( nTeacherSteps) )
            teacherLinearCoefficient_tmp = np.exp( teacherLinearCoefficient_log_tmp )
        teacherLinearCoefficient.append( teacherLinearCoefficient_tmp )

        kernel_teacher_tmp = kpu.getPolynomialKernel( teacherLinearCoefficient_tmp, teacherBalancingPower ) 
        kernel_func_teacher.append( kernel_teacher_tmp )

    # construct filenames for results
    teacherFileName = 'TeacherGP_TestErrorProfile_TeacherP' + str(teacherBalancingPower) + '_StudentN' + str(studentOrder) + '_sigma2=' + str(sigma2_teacher) + '_p=' + str(pDim) + '_'
    studentFileName = 'StudentGP_TestErrorProfile_TeacherN' + str(teacherBalancingPower) + '_StudentN' + str(studentOrder) + '_sigma2=' + str(sigma2_student) + '_p=' + str(pDim) + '_'

    if( studentAntiSymmetric==True ):
        teacherFileName = teacherFileName + 'AntiSym.txt'
        studentFileName =  studentFileName + 'AntiSym.txt'
    else:
        teacherFileName = teacherFileName + 'NotAntiSym.txt'
        studentFileName = studentFileName + 'NotAntiSym.txt'
        
    # open files
    if( os.path.isdir( './Figure4b' ) == False ):     
        os.mkdir( './Figure4b')

    f1 = open( './Figure4b/' + studentFileName, 'w' )
    f2 = open( './Figure4b/' + teacherFileName, 'w' )

    for setSize_idx in range( len( trainingSetSize ) ):
        for sample_idx in range( nSample ):
            # generate training data
            xTrainData = gpu.sampleInput( pDim, trainingSetSize[setSize_idx] )
            xTestData = gpu.sampleInput( pDim, 1 )

            for teacherStep in range( nTeacherSteps+1 ):
                linearCoefficient = teacherLinearCoefficient[teacherStep]
                genErrorInstance_teacher = gpu.calcGenErrorInstance( xTrainData, xTestData, kernel_func_teacher[teacherStep], sigma2_teacher, kernel_func_teacher[teacherStep], sigma2_teacher )

                for l in range( nLayerMax ):
                    genErrorInstance_student = gpu.calcGenErrorInstance( xTrainData, xTestData, kernel_func_teacher[teacherStep], sigma2_teacher, kernel_func_student[l], sigma2_student )

                    f1.write( str(linearCoefficient) + "," + str(l) + "," + str( trainingSetSize[setSize_idx] ) + "," + str( sample_idx ) + "," + str( genErrorInstance_student ) + "\n" )
                    f1.flush()

                f2.write( str(linearCoefficient) + "," + str( trainingSetSize[setSize_idx] ) + "," + str( sample_idx ) + "," + str( genErrorInstance_teacher ) + "\n" )
                f2.flush()

    f1.close()  
    f2.close()  

if __name__ == '__main__':
    main()    

