function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

x1=X((1:end),1);
x2=X((1:end),2);

Cterm=[0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]';

sigmaterm=[0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]';

matrsize=length(Cterm)*length(sigmaterm);

errormatr=(zeros(matrsize,3));
sum1=0;
for i=1:length(Cterm)
    for j=1:length(sigmaterm)
	sum1=sum1+1;
	model1=svmTrain(X, y, Cterm(i), @(x1, x2) gaussianKernel(x1, x2, sigmaterm(j)));
	pred1= svmPredict(model1, Xval);
	err=mean(double(pred1~= yval));
	errormatr(sum1,1)=err;
	errormatr(sum1,2)=Cterm(i);
	errormatr(sum1,3)=sigmaterm(j);
   endfor
endfor	

[w,k]=min(errormatr);

C=errormatr(k(1),2);
sigma=errormatr(k(1),3);	



% =========================================================================

end
