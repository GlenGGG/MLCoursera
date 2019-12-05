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
n=8;
cset=ones(n,1);
sset=ones(n,1);
cset=cset.*1e-4;
sset=sset.*1e-5;
cset(2:end)=10;
sset(2:end)=10;
cset=cumprod(cset);
sset=cumprod(sset);
cset=cset.*3;
sset=sset.*3;
set=[0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30]
bset=[set(1) set(1)];
min=10000000000;
for i=1:n
    for j=1:n
        model= svmTrain(X, y, set(i), @(x1, x2) gaussianKernel(x1, x2, set(j)));
        p=svmPredict(model,Xval);
        err=mean(double(p~=yval));
        if err<=min;
            min=err;
            bset=[set(i) set(j)];
        end
    end
end

C=bset(1, 1);
sigma=bset(1,2);



% =========================================================================

end
