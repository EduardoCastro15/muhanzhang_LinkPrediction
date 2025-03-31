function [auc, best_threshold, best_precision, best_recall, best_f1_score] = WLNM(train, test, K, ith_experiment)
    %  Usage: the main program for Weisfeiler-Lehman Neural Machine (WLNM)
    %  --Input--
    %  -train: a sparse matrix of training links (1: link, 0: otherwise)
    %  -test: a sparse matrix of testing links (1: link, 0: otherwise)
    %  -K: number of vertices in an enclosing subgraph
    %  -ith_experiment: exp index, for parallel computing
    %  --Output--
    %  -auc: the AUC score of WLNM
    %
    %  *author: Muhan Zhang, Washington University in St. Louis
    %%
    if nargin < 3
        K = 20;
    end
    if nargin < 4
        ith_experiment = 1;
    end
    
    htrain = triu(train, 1);  % half train adjacency matrix
    htest = triu(test, 1);
    
    % sample negative links for train and test sets
    [train_pos, train_neg, test_pos, test_neg] = sample_neg(htrain, htest, 2, 1, true);  % change the last argument to true to do link prediction on whole network
    
    % Convert graphs to feature vectors
    [train_data, train_label] = graph2vector(train_pos, train_neg, train, K);
    [test_data, test_label] = graph2vector(test_pos, test_neg, train, K);
    
    % train a feedforward neural network in MATLAB
    layers = [imageInputLayer([K*(K-1)/2 1 1], 'Normalization','none')
        fullyConnectedLayer(32)
        reluLayer
        fullyConnectedLayer(32)
        reluLayer
        fullyConnectedLayer(16)
        reluLayer
        fullyConnectedLayer(2)
        softmaxLayer
        classificationLayer];
    opts = trainingOptions('sgdm', 'InitialLearnRate', 0.1, 'MaxEpochs', 200, 'MiniBatchSize', 128, ...
        'LearnRateSchedule','piecewise', 'LearnRateDropFactor', 0.9, 'L2Regularization', 0, ...
        'ExecutionEnvironment', 'cpu');
    net = trainNetwork(reshape(train_data', K*(K-1)/2, 1, 1, size(train_data, 1)), categorical(train_label), layers, opts);
    [~, scores] = classify(net, reshape(test_data', K*(K-1)/2, 1, 1, size(test_data, 1)));
    scores(:, 1) = [];
    
    % calculate the AUC
    [~, ~, ~, auc] = perfcurve(test_label', scores', 1);
    
    % Set a range of threshold values to test for binary classification
    thresholds = 0.1:0.05:0.9;
    best_f1_score = 0;  % Initialize the best F1-score
    best_threshold = 0;  % Initialize the best threshold
    
    % Iterate over different threshold values to find the best one
    for t = thresholds
        binary_predictions = scores' > t;
        
        % Calculate true positives, false positives, false negatives
        true_positives = sum((binary_predictions == 1) & (test_label' == 1));
        false_positives = sum((binary_predictions == 1) & (test_label' == 0));
        false_negatives = sum((binary_predictions == 0) & (test_label' == 1));
    
        % Calculate Precision, Recall, and F1-Score
        if (true_positives + false_positives) > 0
            precision = true_positives / (true_positives + false_positives);
        else
            precision = 0;
        end
    
        if (true_positives + false_negatives) > 0
            recall = true_positives / (true_positives + false_negatives);
        else
            recall = 0;
        end
    
        if (precision + recall) > 0
            f1_score = 2 * (precision * recall) / (precision + recall);
        else
            f1_score = 0;
        end
    
        % Update the best threshold if the current F1-score is better
        if f1_score > best_f1_score
            best_f1_score = f1_score;
            best_threshold = t;
            best_precision = precision;
            best_recall = recall;
        end
    end
    
    % Display the best threshold and corresponding metrics
    fprintf('Best Threshold: %.2f, Precision: %.4f, Recall: %.4f, F1-Score: %.4f\n', best_threshold, best_precision, best_recall, best_f1_score);
    fprintf('AUC: %.4f\n', auc);
end
