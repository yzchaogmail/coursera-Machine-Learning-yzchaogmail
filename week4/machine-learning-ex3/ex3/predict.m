function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%
X = [ones(m,1) X];
Z2 = Theta1 * X';     % Z2(num_unit2,m) = theta(num_unit2,n+1) * X(m,n+1)' ;
A2 = sigmoid(Z2);     % A2(num_unit2,m)
A2 = A2';             % A2(m,num_unit2)

num_unit2 = size(A2,1);
A2 = [ones(num_unit2,1) A2]; % A2 = A0(2) + A2

Z3 = Theta2 * A2';    %Z3(num_labels * m) = theta2(num_labels * num_unit2) * A2(m,num_unit2)'
Z3 = sigmoid(Z3);
Z3 = Z3';             %Z3(m*num_labels);

[valueP,p] = max(Z3,[],2);

% =========================================================================

end
