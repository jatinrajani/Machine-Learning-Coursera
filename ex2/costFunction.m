function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
% Note: grad should have the same dimensions as theta

	
	a=X*theta;
	hypo=sigmoid(a); %It wil give hypothesis matrix 
	c1=log(hypo);
	c0=log(1-hypo);
	r=1-y;
	cost=(-(c1'*y)-(c0'*r));
	cost=cost/m; 	%It wil give cost
	
	g=(hypo-y);
	grad1=X'*g;	%It wil give gradient matrix
	grad=grad1/m; 
	J=cost;



% =============================================================

end
