function [X_test, Y_test, X_train, Y_train] = split_data(X,y,test_size)
%this function splits data into training and test sets
% INPUT:    X - data
%           y - classes
%           test_size - % of data to be test data
% OUTPUT:   X_test - test data
%           Y_test - test classes
%           X_train - training data
%           Y_train - test data
%
%Written by Sophie A. Rogers, Corder Laboratory, University of Pennsylvania
    vec = randperm(length(X(:,1)));
    testIndMax = round(test_size*length(X(:,1)));
    testInd = vec(1:testIndMax);
    trainInd = vec(testIndMax+1:end);
    
    X_test = X(testInd,:);
    Y_test = y(testInd);
    X_train = X(trainInd,:);
    Y_train = y(trainInd);