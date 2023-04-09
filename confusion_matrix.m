function confusion_matrix(A,B, number_of_samples)

%CONFUSION MATRIX 
%confusion_matrix(A,B) returns the confusion matrix which summarizes the
%performance of the machine learning algorithm, the accuracy, the
%misclassification rate, the microF1-score, the precision, the recall
%and the F1-score for each digit.
%Input parameters: 
%A is the target vector matrix 
%B is the output matrix 
%number_of_samples is the number of samples



%Defines the number of correct predictions 

counts = zeros(1,10); 

%Defines the number of wrong predictions for each digit

wrong_prediction_digit = zeros(10,10); 

%Counts the number of correct and wrong predictions

for l = 1:number_of_samples
    
    [targetmatrix,targetindex] = max(A(:,l));
    [outmatrix,outindex] = max(B(:,l));
    
    for i = 1:10
        if A(:,l) == B(:,l)
            
            if i == targetindex
            counts(i) = counts(i) + 1;
            end  
            
        else
            
            if i == outindex
               j = targetindex;
               wrong_prediction_digit(i,j) = wrong_prediction_digit(i,j) + 1;
            end 
            
         end
    end
end

%Confusion matrix 

confusion_matrix = zeros(10,10);

for j = 1:10
    for i = 1:10
        if i == j
            confusion_matrix(i,j) = counts(i);
        else 
            confusion_matrix(i,j) = wrong_prediction_digit(i,j);
        end       
    end    
end 

%Displays confusion matrix on command window

fprintf('                            CONFUSION MATRIX\n');
writematrix(confusion_matrix, 'confusion_matrix.txt','Delimiter','tab')
type 'confusion_matrix.txt'
fprintf('\n');

%Defines false positives and negatives in predicted output

false_positives = zeros(1,10); 
false_negatives = zeros(1,10);

for i = 1:10
    
    if i == 10
       false_positives(i) = sum(confusion_matrix(10,1:9));
       false_negatives(i) = sum(confusion_matrix(1:9,10));
    
    else 
    
    false_positives(i) = sum(confusion_matrix(i,i+1:10));
    false_negatives(i) = sum(confusion_matrix(i+1:10,i));
    
    end 
end

%Overall accuracy of the model

accuracy = sum(counts)/number_of_samples;

%Fraction of incorrect predictions

misclassification_rate = (1 - accuracy);

%Micro-F1 score is evaluated over the total correct 
%predictions, false positives and negatives of the model

microF1 = 2*sum(counts)/(2*sum(counts)+sum(false_positives)+sum(false_negatives));

fprintf('                            CLASSIFICATION REPORT\n\n');

param = table(accuracy,misclassification_rate,microF1,'VariableNames',{'ACCURACY','MISCLASSIFICATION RATE','Micro-F1'});
disp(param)
fprintf('\n\n');

%PRECISION for each digit aka the fraction of predicted labels 
%which were actually correctly predicted

Precision = zeros(1,10);

for i = 1:10
    Precision(i) = counts(i)/(counts(i)+false_positives(i));
end


%RECALL for each digit aka the fractions of predicted labels 
%which were correctly predicted by the model

Recall = zeros(1,10);

for i = 1:10
    Recall(i) = counts(i)/(counts(i)+false_negatives(i));
end


%F1-score for each digit, aka harmonic mean of precision and recall

F1score = zeros(1,10);

for i = 1:10
    F1score(i) = 2*counts(i)/(2*counts(i)+false_positives(i)+false_negatives(i));
end

Digit = 0:9;

results = table(Digit',Precision',Recall',F1score','VariableNames',{'Digit','PRECISION','RECALL','F1-score'});
disp(results)

end
