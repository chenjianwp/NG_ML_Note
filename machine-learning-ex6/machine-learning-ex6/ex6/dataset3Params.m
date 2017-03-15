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

C_list=[0.01;0.03;0.1;0.3;1;3;10;30];
sigma_list=C_list(:);

index=1;
result=zeros(8,3);
for i=1:8
  for j=1:8
    result(index,1)=C_list(i);
    result(index,2)=sigma_list(j);
    model= svmTrain(X, y, C_list(i), @(x1, x2) gaussianKernel(x1, x2, sigma_list(j)));
    predictions=svmPredict(model,Xval);
    result(index,3)=mean(double(predictions~=yval));
    index++;
  end
end
    [min,col]=min(result);
    C=result(col(3),1);
    sigma=result(col(3),2);





% =========================================================================

end
