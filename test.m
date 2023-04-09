%27 AUGUST 2021 Ver. 7.0
% AUTHOR CHIARA ZAVATTO 
%
%Develop, train and test a simple neural network to recognize single
%digits between 0 and 9 using Back-Propagation algorithm.

%TEST

%Loads test data

test_data = load('mnist_test.mat');

test_images = test_data.data_test.X_test;
test_labels = test_data.data_test.y_test;

%Divides each pixel value by 255 which results in inputs bounded between
%0 and 1.

test_images = test_images/255;

number_of_testsamples = size(test_images,1);

test_images = test_images';

%Converts the labels in target vectors having all zero elements but 1
%corresponding to the correct digit between 0 and 9 (e.g. if the first
%label is 5 the corresponding target vector will have all 0 elements but a
%1 in the 6-th position).

test_targets = zeros(10,number_of_testsamples);

for i = 1:number_of_testsamples
    test_targets(test_labels(i)+1,i) = 1;
end

%Loads updated model coefficients 

w_training34 = load('train_w34.mat');
w_training23 = load('train_w23.mat');
w_training12 = load('train_w12.mat');

b_training34 = load('train_b34.mat');
b_training23 = load('train_b23.mat');
b_training12 = load('train_b12.mat');

w_34 = w_training34.w34;
w_23 = w_training23.w23;
w_12 = w_training12.w12;

b_34 = b_training34.b34;
b_23 = b_training23.b23;
b_12 = b_training12.b12;


%Defines the output vector which will store the predictions of the model
%for further model performance evaluation

out = zeros(10,number_of_testsamples);

for j = 1:number_of_testsamples
    
    %FEEDFORWARD
    
    out_test0 = test_images(:,j);
    net_test1 = w_12*out_test0 + b_12;
    out_test1 = sigmoid(net_test1);
    net_test2 = w_23*out_test1 + b_23;
    out_test2 = sigmoid(net_test2);
    net_test3 = w_34*out_test2 + b_34;
    out_test3 = sigmoid(net_test3);
    
    %Takes as prediction the output value which is closer to 1 
    
    [maximum,index] = max(out_test3);
    
    for i = 1:10
        if i == index
           out(i,j) = 1;
        else
            out(i,j) = 0;
        end
    end
    
    total_testloss(:,j) = 0.5*(test_targets(:,j) - out(:,j)).^2;
    
end


%Computes and displays CONFUSION MATRIX and report on the performance of
%the model

confusion_matrix(test_targets, out, number_of_testsamples);