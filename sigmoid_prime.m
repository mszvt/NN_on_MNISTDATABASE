function a = sigmoid_prime(x)

b = zeros(length(x),1);

for i = 1:length(x)
    b(i) = (1./(1.0 + exp(-1.0*x(i))))*(1 - 1./(1.0 +exp(-1.0*x(i))));
end

 a = b;
 
end
