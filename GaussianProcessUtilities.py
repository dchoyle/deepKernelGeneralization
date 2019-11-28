import numpy as np
import math

def calcBayesError( nPoints, eig_student, noise_student, nIter=20 ):
    """
    Function to calculate Bayes error of student Gaussian Process
    given individual, possibly degenerate, eigenvalues.
    See - Sollich,'Gaussian Process Regression with Mismatched Models', 
    arXiv:cond-mat/0106475v1, 2001
    
    :param nPoints: Training set size
    :param eig_student: Array of student covariance eigenvalues
    :param noise_student: Variance of noise component of student 
    covariance
    :param nIter: Number of iterations to use when solving 
    self-consistent condition for the Bayes error
    :return: Returns the Bayes error
    """
    epsilon_hat = sum( eig_student )
    
    for iter in range( nIter ):
        print( iter, epsilon_hat )
        numerator = eig_student * (noise_student + epsilon_hat)
        denominator = noise_student + (nPoints*eig_student) + epsilon_hat
        
        epsilon_hat = sum( numerator / denominator )
                    
    return( epsilon_hat )
    
def calcBayesError_Degenerate( nPoints, 
                               eig_student, 
                               degeneracy, 
                               noise_student, 
                               nIter=20 ):
    """
    Function to calculate Bayes error of student Gaussian Process
    given eigenvalue and their multiplicities.
    See - Sollich,'Gaussian Process Regression with Mismatched Models', 
    arXiv:cond-mat/0106475v1, 2001
    
    :param nPoints: Training set size
    :param eig_student: Array of student covariance eigenvalues
    :param degeneracy: Array of level of degeneracy (multiplicty) for 
    each unique eigenvalue
    :param noise_student: Variance of noise component of student 
    covariance
    :param nIter: Number of iterations to use when solving 
    self-consistent condition for the Bayes error
    :return: Returns the Bayes error
    """
    
    # calculate alpha factors
    alpha = nPoints / degeneracy  
    
    epsilon_hat = sum( eig_student )
    
    for iter in range( nIter ):
        numerator = eig_student * (noise_student + epsilon_hat)
        denominator = noise_student + (alpha*eig_student) + epsilon_hat
        
        epsilon_hat = sum( numerator / denominator )
                    
    return( epsilon_hat )
    


def calcExpectedGeneralizationError( nPoints, 
                                     eig_student, 
                                     eig_teacher, 
                                     noise_student, 
                                     noise_teacher ):
    """ 
    Function to calculate expected generalization error of student 
    Gaussian Process.
    See - Sollich,'Gaussian Process Regression with Mismatched Models', 
    arXiv:cond-mat/0106475v1, 2001
    
    :param nPoints: Training set size
    :param eig_student: Array of student covariance eigenvalues
    :param eig_student: Array of teacher covariance eigenvalues
    :param noise_student: Variance of noise component of student 
    covariance
    :param noise_teacher: Variance of noise component of teacher 
    covariance
    self-consistent condition for the Bayes error
    :return: Returns array with the Bayes error and the 
    generalization error
    """
    
    # calculate Bayes error
    bayes_error = calcBayesError( nPoints, eig_student, noise_student )
    
    # determine g
    g_numerator =  eig_student * (noise_student + bayes_error)
    g_denominator = noise_student + (nPoints*eig_student) + bayes_error
    g = g_numerator / g_denominator
    g2 = np.power( g, 2.0 )
    trG2 = sum( g2 )
    
    # determine correction factor to Bayes error
    xtmp1 = np.power( (noise_student + bayes_error), 2.0 )
    correction_numerator = noise_teacher * trG2
    correction_numerator += (xtmp1*(sum( (g2*eig_teacher) / np.power( eig_student, 2.0 ) ) / nPoints))
    correction_denominator = noise_student * trG2
    correction_denominator += (xtmp1*(sum( g2 / eig_student ) / nPoints))
    
    generalization_error = bayes_error * (correction_numerator/correction_denominator)
    
    return( [bayes_error, generalization_error] )
    
def calcExpectedGeneralizationError_Degenerate( nPoints, 
                                                eig_student, 
                                                eig_teacher, 
                                                degeneracy, 
                                                noise_student, 
                                                noise_teacher, 
                                                nIter ):
    """ 
    Function to calculate expected generalization error of student 
    Gaussian Process, given multiplicities of student and teacher
    eigenvalues.
    See - Sollich,'Gaussian Process Regression with Mismatched Models', 
    arXiv:cond-mat/0106475v1, 2001
    
    :param nPoints: Training set size
    :param eig_student: Array of student covariance eigenvalues
    :param eig_teacher: Array of teacher covariance eigenvalues
    :param degeneracy: Array of level of degeneracy (multiplicty) for 
    each unique eigenvalue. We assume student and teacher eigenvalues 
    are matched and so have same level of degeneracy.
    :param noise_student: Variance of noise component of student 
    covariance
    :param noise_teacher: Variance of noise component of teacher 
    covariance
    self-consistent condition for the Bayes error
    :return: Returns array with the Bayes error and the 
    generalization error
    """
    
    # calculate alpha factors
    alpha = nPoints / degeneracy 
   
    # calculate Bayes error
    bayes_error = calcBayesError_Degenerate( nPoints, eig_student, degeneracy, noise_student, nIter )
    
    
    # determine correction factor to Bayes error
    xtmp1 = np.power( (noise_student + bayes_error), 2.0 )
    xtmp2 = np.power( (noise_student + bayes_error + (alpha*eig_student)), 2.0 )
    
    sum1 = 0.0
    sum2 = 0.0
    sum3 = 0.0

    for i in range( len(eig_student) ):
        sum1 += ((alpha[i] * np.power( eig_student[i], 2.0 )) / xtmp2[i])
        sum2 += (eig_teacher[i] / xtmp2[i])
        sum3 += (eig_student[i] / xtmp2[i])
    
    correction_numerator = noise_teacher * sum1
    correction_numerator += (xtmp1*sum2)
    correction_denominator = noise_student * sum1
    correction_denominator += (xtmp1*sum3)
    
    generalization_error = bayes_error * (correction_numerator/correction_denominator)
    
    return( [bayes_error, generalization_error] )    

def calcSollichLearningCurveFunction( alpha, 
                                      sigma2_effective_student, 
                                      sigma2_effective_teacher ):
    """
    Function to calculate Sollich effective learning curve for a single
    learning stage.
    See - Sollich,'Gaussian Process Regression with Mismatched Models', 
    arXiv:cond-mat/0106475v1, 2001
    
    :param alpha: Ratio of training set size to degeneracy of eigenvalue 
    being learnt
    :param sigma2_effective_student: Effective noise level of student 
    :param sigma2_effective_teacher: Effective noise level of teacher
    :return: Returns proportion of eigenvalue yet to be learnt
    """

    xtmp = 1.0 - alpha - sigma2_effective_student
    bayes_error = 0.5 * ( xtmp + np.sqrt( math.pow( xtmp, 2.0 ) + (4.0*sigma2_effective_student) ) )
    
    xtmp2 = math.pow( (sigma2_effective_student + bayes_error), 2.0 ) / alpha
    rfunction = bayes_error * (sigma2_effective_teacher + xtmp2) / ( sigma2_effective_student + xtmp2 )

    return( rfunction )
    
def calcSollichGeneralizationErrorFromEigs( alpha, 
                                            eigs_student, 
                                            eigs_teacher,
                                            eigSumRemainder_student, 
                                            eigSumRemainder_teacher,
                                            sigma2_base_student, 
                                            sigma2_base_teacher, 
                                            learningStage ):
    
    """ 
    Function to calculate approximate expected generalization error 
    of student Gaussian Process for a specified learning stage.
    See - Sollich,'Gaussian Process Regression with Mismatched Models', 
    arXiv:cond-mat/0106475v1, 2001
        
    :param alpha: Ratio of training set size to degeneracy of eigenvalue 
    being learnt
    :param eig_student: Array of leading student covariance eigenvalues
    :param eig_teacher: Array of leading teacher covariance eigenvalues
    :param eigSumRemainder_student: Remaining student variance in later 
    learning stages
    :param eigSumRemainder_teacher: Remaining teacher variance in later 
    learning stages
    :param sigma2_base_student: Variance of noise component of student 
    covariance
    :param sigma2_base_teacher: Variance of noise component of teacher 
    covariance
    :param learningStage: Integer specifying which eigenvalue we are
    learning
    :return: Returns Sollich's approximation to the expected 
    generalization error
    """
                                               
    # construct effective teacher noise level
    f_student_plus1 = np.sum( eigs_student[learningStage:len(eigs_student)] ) + eigSumRemainder_student
    f_teacher_plus1 = np.sum( eigs_teacher[learningStage:len(eigs_student)] ) + eigSumRemainder_teacher
    
    sigma2_effective_student = (f_student_plus1 + sigma2_base_student) / eigs_student[learningStage-1]
    sigma2_effective_teacher = (f_teacher_plus1 + sigma2_base_teacher) / eigs_teacher[learningStage-1]
    
    rLearningCurve = calcSollichLearningCurveFunction( alpha, sigma2_effective_student, sigma2_effective_teacher )
    
    generalizationError = f_teacher_plus1 + (rLearningCurve * eigs_teacher[learningStage-1])
    
    return( generalizationError )
    

def sampleInput( pDim, nSample ):
    """
    Function to generate samples drawn uniformly from surface of 
    hypersphere
    
    :param pDim: Data dimension
    :param nSample: Number of sample vectors required
    :return: Numpy array of sample vectors
    """
    
    # draw input vector samples
    xInput = np.random.randn( nSample, pDim )
    for i in range( nSample ):
        xInput_norm = np.dot( xInput[i, ], xInput[i, ] )
        xInput[i, ] /= np.sqrt( xInput_norm )

    return( xInput )
   

def calcGenErrorInstance( xTrain, xTest, teacherKernel, teacherNoise, studentKernel, studentNoise ):
    """
    Function to calculate the expected generalization error of a 
    Gaussian Process at a specified test point and given a specified 
    set of training feature vectors
    
    :param xTrain: Array of training feature vectors
    :param xTest: Feature vector for the test point
    :param teacherKernel: Kernel function for teacher Gaussian Process
    :param teacherNoise: Variance of noise process in teacher data
    :param studentKernel: Kernel function for student Gaussian Process
    :param studentNoise: Variance of noise process in student response
    :return: Return expected student prediction error on test point
    """
    
    nTrain = xTrain.shape[0]
    
    # calculate covariance matrices
    covMat = np.zeros( (nTrain, nTrain) )
    covMatStar = np.zeros( (nTrain, nTrain) )

    for i in range( nTrain ):
        for j in range( i, nTrain ):
            scalar_product = np.dot( xTrain[i, ], xTrain[j, ] )
            scalar_product = ( np.sign( scalar_product ) * np.min( [1.0, np.abs( scalar_product )] ) )
            covMat[i, j] = studentKernel( scalar_product )
            covMat[j, i] = covMat[i, j]
            
            covMatStar[i, j] = teacherKernel( scalar_product )
            covMatStar[j, i] = covMatStar[i, j]
            
            
        covMat[i, i] += studentNoise
        covMatStar[i, i] += teacherNoise
 
    
    # now invert the covariance matrix
    covMat_inv = np.linalg.inv( covMat )
    
    
    # compute k vectors
    kVec = np.zeros( nTrain )
    kVec_star = np.zeros( nTrain )
    for j in range( nTrain ):
        scalar_product = np.dot( xTest, xTrain[j, ] )
        scalar_product = ( np.sign( scalar_product ) * np.min( [1.0, np.abs( scalar_product )] ) )
        kVec[j] = studentKernel( scalar_product )
        kVec_star[j] = teacherKernel( scalar_product )
    
    
    
    # compute generalization error
    epsilonSample = 1.0 - (2.0 * np.inner( kVec_star, np.dot( covMat_inv, kVec ) ) )
    effective_vec = np.dot( covMat_inv, kVec )
    epsilonSample += np.inner( effective_vec , np.dot( covMatStar, effective_vec ) )
    
    return( epsilonSample )
    

def trainAndPredictGP( xTrain, yTrain, xTest, yTest, kernel, noise, scalingFactor ):
    """
    Function to calculate RMSE on a test data set for a GP, given an input 
    training set and a choice of covariance kernel.
    
    :param xTrain: Covariates of training set
    :param yTrain: Response variable values of training set
    :param xTest: Covariates of test set
    :param yTest: Response variable values of test set
    :param kernel: Covariance kernel function of GP
    :param noise: The variance of the additive noise. This gives an additional
    diagonal contribution to covariance
    :param scalinFactor: A vector of scaling factors to be applied to each 
    covariate
    :return: Returns RMSE on test set
    
    """
    nTrain = xTrain.shape[0]
    yTrain_mean = np.mean( yTrain )

    # calculate covariance matrix and kVec
    covMat = np.zeros( (nTrain, nTrain) )
    for i in range( nTrain ):
        for j in range( i, nTrain ):
            scalar_product = np.dot( xTrain[i, ], xTrain[j, ] )
            scalar_product = ( np.sign( scalar_product ) * np.min( [1.0, np.abs( scalar_product )] ) )
            covMat[i, j] = kernel( scalar_product ) * scalingFactor
            covMat[j, i] = covMat[i, j]
            
            
        covMat[i, i] += noise
 
    
    # now invert the covariance matrix
    covMat_inv = np.linalg.inv( covMat )
    effective_vec = np.dot( covMat_inv, (yTrain -yTrain_mean ) )
    
    # loop over test points and evaluate  prediction
    nTest = xTest.shape[0]
    predictError = 0.0

    for i in range( nTest ):
        # compute k vector
        kVec = np.zeros( nTrain )
        for j in range( nTrain ):
            scalar_product = np.dot( xTest[i, ], xTrain[j, ] )
            scalar_product = ( np.sign( scalar_product ) * np.min( [1.0, np.abs( scalar_product )] ) )
            kVec[j] = kernel( scalar_product ) * scalingFactor

        predictTmp = np.inner( kVec ,  effective_vec ) + yTrain_mean
        predictError += np.power( yTest[i] - predictTmp, 2.0 )

    predictError = np.sqrt( predictError / float( nTest ) )

    return( predictError )
    
