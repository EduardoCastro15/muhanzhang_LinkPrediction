function auc = WLNM(train, test, K, ith_experiment)
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
end