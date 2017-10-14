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

J=1/(length(y))*(-y'*log(sigmoid(X*theta))-(ones(size(y))-y)'*log(ones(size(X*theta))-sigmoid(X*theta)))+lambda/2/length(y)*(theta(2:length(theta),:))'*theta(2:length(theta),:);
Xreg=X(:,2:end);
thetaReg=theta(2:end,:);
grad(2:end,:)=1/m*(Xreg'*(sigmoid(X*theta)-y))+lambda/m*thetaReg;
                             Xzero=X(:,1);
grad(1)=1/m*(Xzero'*(sigmoid(X*theta)-y));

% =============================================================

end
