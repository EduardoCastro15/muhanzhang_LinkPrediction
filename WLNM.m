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

%For later retrieve of the original (i, j) node pairs in test and match them to predictions
test_pairs = [test_pos; test_neg];

[train_data, train_label] = graph2vector(train_pos, train_neg, train, K);
[test_data, test_label] = graph2vector(test_pos, test_neg, train, K);

% train a model
model = 3;
switch model
    case 1  % logistic regression
        addpath('software/liblinear-2.1/matlab');  % need to install liblinear
        train_data = sparse(train_data);
        test_data = sparse(test_data);
        [~, optim_c] = evalc('liblinear_train(train_label, train_data, ''-s 0 -C -q'');');
        model = liblinear_train(train_label, train_data, sprintf('-s 0 -c %d -q', optim_c(1)));
        [~, acc, scores] = liblinear_predict(test_label, test_data, model, '-b 1 -q');
        acc
        l1 = find(model.Label == 1);
        scores = scores(:, l1);
    case 2 % train a feedforward neural network in Torch
        addpath('software/liblinear-2.1/matlab');  % need to install liblinear
        train_data = sparse(train_data);
        test_data = sparse(test_data);
        if exist('tempdata') ~= 7
            !mkdir tempdata
        end
        % libsvmwrite(sprintf('tempdata/traindata_%d', ith_experiment), train_label, train_data);
        % libsvmwrite(sprintf('tempdata/testdata_%d', ith_experiment), test_label, test_data);  % prepare data
        % Convert sparse matrix to full matrix before writing
        train_data_full = full(train_data);
        test_data_full = full(test_data);
        % Write to CSV files
        writematrix([train_label, train_data_full], sprintf('tempdata/traindata_%d.csv', ith_experiment));
        writematrix([test_label, test_data_full], sprintf('tempdata/testdata_%d.csv', ith_experiment));

        cmd = sprintf('th nDNN.lua -inputdim %d -ith_experiment %d', K * (K - 1) / 2, ith_experiment);
        [status, cmdout] = system(cmd, '-echo');  % Capture the status and output of the command
        if status ~= 0
            error('External command failed: %s', cmdout);
        end

        scores = load(sprintf('tempdata/test_log_scores_%d.asc', ith_experiment));
        delete(sprintf('tempdata/traindata_%d', ith_experiment));  % to delete temporal train and test data
        delete(sprintf('tempdata/testdata_%d', ith_experiment));
        delete(sprintf('tempdata/test_log_scores_%d.asc', ith_experiment));
    case 3 % train a feedforward neural network in MATLAB
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
    case 4 % train a neural network with sklearn
        addpath('software/liblinear-2.1/matlab');  % need to install liblinear
        train_data = sparse(train_data);
        test_data = sparse(test_data);
        if exist('tempdata') ~= 7
            !mkdir tempdata
        end
        libsvmwrite(sprintf('tempdata/traindata_%d', ith_experiment), train_label, train_data);
        libsvmwrite(sprintf('tempdata/testdata_%d', ith_experiment), test_label, test_data);  % prepare data
        cmd = sprintf('python3 nDNN.py %d %d', K * (K - 1) / 2, ith_experiment);
        system(cmd, '-echo');
        scores = load(sprintf('tempdata/test_log_scores_%d.asc', ith_experiment));
        delete(sprintf('tempdata/traindata_%d', ith_experiment));  % to delete temporal train and test data
        delete(sprintf('tempdata/testdata_%d', ith_experiment));
        delete(sprintf('tempdata/test_log_scores_%d.asc', ith_experiment));
end

% --- Step 4: Find the best threshold using F1 score ---
[~, ~, ~, auc] = perfcurve(test_label', scores', 1);

thresholds = 0.1:0.05:0.9;
best_f1_score = 0;
best_threshold = 0;

for t = thresholds
    binary_predictions = scores' > t;

    % Compute confusion components
    true_positives = sum((binary_predictions == 1) & (test_label' == 1));
    false_positives = sum((binary_predictions == 1) & (test_label' == 0));
    false_negatives = sum((binary_predictions == 0) & (test_label' == 1));

    % Precision
    if (true_positives + false_positives) > 0
        precision = true_positives / (true_positives + false_positives);
    else
        precision = 0;
    end

    % Recall
    if (true_positives + false_negatives) > 0
        recall = true_positives / (true_positives + false_negatives);
    else
        recall = 0;
    end

    % F1 Score
    if (precision + recall) > 0
        f1_score = 2 * (precision * recall) / (precision + recall);
    else
        f1_score = 0;
    end

    % Save best threshold
    if f1_score > best_f1_score
        best_f1_score = f1_score;
        best_threshold = t;
        best_precision = precision;
        best_recall = recall;
    end
end

% --- Step 5: Final binary predictions using best threshold ---
binary_predictions = scores' > best_threshold;

% Save node pairs used in test set (you must create this before calling graph2vector)
% test_pairs = [test_pos; test_neg];

predicted_links = test_pairs(binary_predictions == 1, :);  % links predicted as existing
true_links      = test_pairs(test_label == 1, :);          % actual positive examples

% Optional: confusion link sets
TP_links = intersect(predicted_links, true_links, 'rows');
FP_links = setdiff(predicted_links, true_links, 'rows');
FN_links = setdiff(true_links, predicted_links, 'rows');

% --- Step 6: Export results ---
exp_id = sprintf('exp_%d_K_%d', ith_experiment, K);  % unique experiment ID
results_dir = 'data/result/testing/';
if ~exist(results_dir, 'dir')
    mkdir(results_dir);
end

writematrix(predicted_links, fullfile(results_dir, ['predicted_links_' exp_id '.csv']));
writematrix(true_links,      fullfile(results_dir, ['true_links_' exp_id '.csv']));
writematrix(TP_links,        fullfile(results_dir, ['TP_links_' exp_id '.csv']));
writematrix(FP_links,        fullfile(results_dir, ['FP_links_' exp_id '.csv']));
writematrix(FN_links,        fullfile(results_dir, ['FN_links_' exp_id '.csv']));

% Display metrics
fprintf('Best Threshold: %.2f\n', best_threshold);
fprintf('Precision: %.4f, Recall: %.4f, F1-Score: %.4f\n', best_precision, best_recall, best_f1_score);
fprintf('AUC: %.4f\n', auc);

end