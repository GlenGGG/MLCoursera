function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
tempj=0;
for i=1:m
    tempj=tempj+(-y(i).*log(sigmoid(X(i,:)*theta))-(1-y(i)).*log(1-sigmoid(X(i,:)*theta)));
end
J=1/m*tempj+lambda/(2*m)*sum(theta(2:size(theta)).*theta(2:size(theta)));
tempg=0;
for j=1:size(theta)
    for i=1:m
        tempg=tempg+(sigmoid(X(i,:)*theta)-y(i)).*X(i,j);
    end
    grad(j)=1/m*tempg;
    if(j~=1)
        grad(j)=grad(j)+lambda/m*theta(j);
    end
    tempg=0;
end


% =============================================================

end
