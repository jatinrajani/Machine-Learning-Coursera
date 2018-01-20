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
	
	gradmatrix=zeros(size(theta));
	a=X*theta;
	hypo=sigmoid(a); %It wil give hypothesis matrix 
	c1=log(hypo);	 %calculating the cost when y=1
	c0=log(1-hypo);	 %calculating the cost when y=0
	r=1-y;
	cost=(-(c1'*y)-(c0'*r));
	theta=theta(2:end);
	theta=theta.**2;  %squaring the theta matrix
	regterm=(sum(theta,1)*lambda)/2; %calculating the regularization parameter
	cost=(cost+regterm)/m; 	%It wil give cost
	
	
	g=(hypo-y);
	grad1=X'*g;	%It wil give gradient matrix
	gradmatrix(1)=grad1(1)/m;
	f=lambda/m;
	gradmatrix(2:end)=grad1(2:end)/m+f*theta;
	
	grad=round(gradmatrix .* 10000) ./ 10000;
	J=cost;
	





% =============================================================

end
