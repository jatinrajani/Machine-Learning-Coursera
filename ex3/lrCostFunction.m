function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
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
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%
	theta1=theta;
	gradmatrix=zeros(size(theta));	
	a=X*theta;
	hypo=sigmoid(a); %It wil give hypothesis matrix 
	c1=log(hypo);	 %calculating the cost when y=1
	c0=log(1-hypo);	 %calculating the cost when y=0
	r=1-y;
	cost=(-(c1'*y)-(c0'*r));
	theta1=theta(2:end);
	theta1=theta1.**2;  %squaring the theta matrix
	regterm=(sum(theta1,1)*lambda)/2; %calculating the regularization parameter
	cost=(cost+regterm)/m; 	%It wil give cost
	
	g=(hypo-y);
	grad1=X'*g;	%It wil give gradient matrix
	gradmatrix(1)=grad1(1)/m;
	f=lambda/m;
	gradmatrix(2:end)=grad1(2:end)/m+f*theta(2:end);
	
	grad=round(gradmatrix .* 10000) ./ 10000;


	J=cost;
	
	  	







% =============================================================

grad = grad(:);

end
