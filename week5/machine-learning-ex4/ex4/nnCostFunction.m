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
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

for i=1:num_labels
  Yi = (y==i);

  if(1 == i)
    YV = Yi;
  else
    YV = [YV Yi];
  endif
end


%{
debug Randomly select 100 data points to display YV
sel = randperm(size(YV, 1));
sel = sel(1:100);
disp(YV(sel, :));

disp(size(YV));
%}

A1 = [ones(m,1) X];   %A1(m,n+1),Theta1(hidden_layer_size,n+1)

Z2 = A1 * Theta1';    %Z2(m,hidden_layer_size);
A2 = sigmoid(Z2);
A2 = [ones(m,1) A2];

Z3 = A2 * Theta2';
A3 = sigmoid(Z3);     %H(x) = A3

%{
%debug Randomly select 100 data points to display Hx/A3
sel = randperm(size(A3, 1));
sel = sel(1:100);
disp(A3(sel, :));
disp(size(A3));

Jtmp = 0;

for i=1:m
  for k=1:num_labels
    Jtmp += (-YV(m,k)) * log(A3(m,k)) - (1-YV(m,k)) * log(1-A3(m,k)) ;
  end
end
J = Jtmp/m;
disp(Jtmp);
disp(m);
disp(J);
%}
% -------------------------------------------------------------

Ytmp = -YV.* log(A3) - (1-YV).* log(1-A3);
J = sum(sum(Ytmp))/m;

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
