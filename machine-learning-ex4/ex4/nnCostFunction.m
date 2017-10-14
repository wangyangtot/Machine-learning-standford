function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

for i=1:m
 temx=[1,X(i,:)];    
 tem1=sigmoid(Theta1*temx');
 tem2=vertcat(1,tem1);
 tem3=sigmoid(Theta2*tem2);
 vecy=zeros(num_labels,1);
 temy=y(i);
 vecy(temy)=1;
 vecy1=-vecy;
 vecy2=ones(size(vecy))-vecy;
 temlog=log(ones(size(tem3))-tem3);
 J=J+vecy1'*log(tem3)-vecy2'*temlog;
end
J=J/m;


%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
trig1=zeros(size(Theta1));
trig2=zeros(size(Theta2));
for j=1:m   
 a1=[1,X(j,:)]'; 
 z2=Theta1*a1;
 a2=[1;sigmoid(z2)];
 z3=Theta2*a2;
 a3=sigmoid(z3);
 vecy=zeros(num_labels,1);
 temy=y(j);
 vecy(temy)=1;
 sigma3=a3-vecy;
 sigma2tem=Theta2'*sigma3.*[1;sigmoidGradient(z2)];
 sigma2=sigma2tem(2:end);
 trig1=trig1+sigma2*a1';
 trig2=trig2+sigma3*a2';
end
Theta1_grad=trig1/m;
Theta2_grad=trig2/m;

 
 
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
temp1=sum(sum(Theta1(:,2:end).*Theta1(:,2:end)));
temp2=sum(sum(Theta2(:,2:end).*Theta2(:,2:end)));
J=J+lambda/2/m*(temp1+temp2);
Theta1_grad(:,2:end)=Theta1_grad(:,2:end)+lambda/m*Theta1(:,2:end);
Theta2_grad(:,2:end)=Theta2_grad(:,2:end)+lambda/m*Theta2(:,2:end);



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
