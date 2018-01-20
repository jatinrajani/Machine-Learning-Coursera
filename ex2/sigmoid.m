function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).
	exp1=zeros(size(z));
	sig=zeros(size(z));	
	z=-z;
	exp1=e.^z+1; % It will take the negative e value of each element
	sig=1./exp1; % It will take the sigmoid value of each element 
	g=sig;




% =============================================================

end
