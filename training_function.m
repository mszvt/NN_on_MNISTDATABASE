%27 AUGUST 2021 Ver. 8.0
% AUTHOR CHIARA ZAVATTO 
%
%Develop, train and test a simple neural network to recognize single
%digits between 0 and 9 using Back-Propagation algorithm.

function training_function(nodes_1, nodes_2, eta, epochs, batch_size)

fprintf('TRAINING\n');

%Loads the training data

train_data = load('mnist_train.mat');

images = train_data.data_train.X_train;
labels = train_data.data_train.y_train;

%Divides each pixel value by 255 which results in inputs bounded between
%0 and 1.

images = images/255; 

number_of_samples = size(images,1);

images = images'; 

%Converts the labels in target vectors having all zero elements but 1
%corresponding to the correct digit between 0 and 9 (e.g. if the first
%label is 5 the corresponding target vector will have all 0 elements but a
%1 in the 6-th position).

targets = zeros(10,number_of_samples);

for i = 1:number_of_samples
    targets(labels(i)+1,i) = 1;
end

%Initialiazes the neural network. Simulates a shallow neural network
%consisting of 2 hidden layers. The number of hidden nodes (neurons) in each
%hidden layer can be arbitrarily set by the user at each running of the
%simulation code.

hidden_nodes = [nodes_1, nodes_2];

%Initializes the weights and the biases.
%Assuming that the neural netwowk is sigmoid activated, chooses to initialize 
%the weights using the Xavier (Glorot) initialization techinique. Regarding 
%the biases, chooses to initiliaze them to 0 according to the most common
%procedure

w12 = randn(hidden_nodes(1),784)*sqrt(2/(784+hidden_nodes(1))); 
w23 = randn(hidden_nodes(2),hidden_nodes(1))*sqrt(2/(hidden_nodes(1)+hidden_nodes(2)));
w34 = randn(10,hidden_nodes(2))*sqrt(2/(hidden_nodes(2)+10));

b12 = zeros(hidden_nodes(1),1);
b23 = zeros(hidden_nodes(2),1);
b34 = zeros(10,1);

%As optimization algorithm used for finding the weights and biases it's
%chosen Mini-Batch Gradient Descent: the training data set is split 
%into small batches that are used to evaluate
%the model errors and update its coefficients (weights and biases)


for k = 1:epochs
    
    batch = 1;
    
    for j = 1:number_of_samples/batch_size
        
        %Initializes errors and gradients 
        
        delta1 = zeros(hidden_nodes(1),1);
        delta2 = zeros(hidden_nodes(2),1);
        delta3 = zeros(10,1);
        
        total_delta1 = zeros(hidden_nodes(1),1);
        total_delta2 = zeros(hidden_nodes(2),1);
        total_delta3 = zeros(10,1);
        
        gradient1 = zeros(hidden_nodes(1),784);
        gradient2 = zeros(hidden_nodes(2),hidden_nodes(1));
        gradient3 = zeros(10,hidden_nodes(2));
        
        for i = batch:batch+batch_size-1
        
        %FEEDFORWARD
        
        out0 = images(:,i);      
        net1 = w12*out0 + b12;
        out1 = sigmoid(net1);
        net2 = w23*out1 + b23;
        out2 = sigmoid(net2);
        net3 = w34*out2 + b34;
        out3 = sigmoid(net3);  
        
        total_loss(:,k) = 0.5*(targets(:,i) - out3).^2;
       
        %BACKPROPAGATION
        
        delta3 = 2*(out3 - targets(:,i)).*sigmoid_prime(net3);
        delta2 = (w34'*delta3).*sigmoid_prime(net2);
        delta1 = (w23'*delta2).*sigmoid_prime(net1);
       
        total_delta3 = total_delta3 + delta3;
        total_delta2 = total_delta2 + delta2;
        total_delta1 = total_delta1 + delta1;
        
        gradient3 = gradient3 + delta3*out2';
        gradient2 = gradient2 + delta2*out1';
        gradient1 = gradient1 + delta1*out0';
        
        end
       
        %GRADIENT DESCENT
        
        w34 = w34 - eta/batch_size*gradient3;
        w23 = w23 - eta/batch_size*gradient2;
        w12 = w12 - eta/batch_size*gradient1;
        
        b34 = b34 - eta/batch_size*total_delta3;
        b23 = b23 - eta/batch_size*total_delta2;
        b12 = b12 - eta/batch_size*total_delta1;
        
        batch = batch + batch_size;
        
    end
   
        %Keeps track of the number of epochs
        
        fprintf('Epoch: %i\n',k);
        
        %Shuffles samples before each iteration
        
        permutations = randperm(number_of_samples);
        images = images(:,permutations);
        targets = targets(:,permutations);
end

%Saves updated weights and biases in .mat files to be used for the test of
%the neural network

save('train_w34.mat','w34');
save('train_w23.mat','w23');
save('train_w12.mat','w12');

save('train_b34.mat','b34');
save('train_b23.mat','b23');
save('train_b12.mat','b12');
save('total_loss.mat','total_loss');
end 
