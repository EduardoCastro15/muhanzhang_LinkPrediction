function [train_pos, train_neg, test_pos, test_neg] = sample_neg(train, test, k, portion, evaluate_on_all_unseen, consumers, resources)
% Usage: to sample negative links for train and test datasets. When sampling negative train links, assume all testing
%       links are known and thus sample negative train links only from other unknown links. Set evaluate_on_all_unseen
%       to true to do link prediction on all links unseen during training.
% Input:
%   - train: half train positive adjacency matrix
%   - test: half test positive adjacency matrix
%   - k: how many times of negative links (w.r.t. pos links) to sample
%   - portion: if specified, only a portion of the sampled train and test links be returned
%   - evaluate_on_all_unseen: if true, will not randomly sample negative testing links, but regard all links unseen during training as neg testing links; train negative links are sampled in the original way
%   - consumers: indices of consumer nodes
%   - resources: indices of resource nodes
%
% Output:
%   - train_pos, train_neg: training positive and negative links
%   - test_pos, test_neg: testing positive and negative links
%%
    if nargin < 3
        k = 1;
    end

    if nargin < 4
        portion = 1;
    end

    if nargin < 5
        evaluate_on_all_unseen = false;
    end

    % Get all positive links
    [train_i, train_j] = find(train);
    train_pos = [train_i, train_j];
    train_size = length(train_i);

    [test_i, test_j] = find(test);
    test_pos = [test_i, test_j];
    test_size = length(test_i);

    % Combine train and test to find all edges
    if isempty(test)
        net = train;
    else
        net = train + test;
    end

    % Ensure positive train and test links do not overlap
    % assert(max(max(net)) == 1, 'Error: Positive train and test links overlap.');
    assert(all(net(:) <= 1), 'Error: Positive train and test links overlap.');

    % Get all potential negative links (non-existent links)
    [neg_i, neg_j] = find(triu(net == 0, 1));  % Upper triangular part for directed graphs
    neg_links = [neg_i, neg_j];

    % Filter negative links based on classification
    is_consumer_consumer = ismember(neg_i, consumers) & ismember(neg_j, consumers);
    is_resource_resource = ismember(neg_i, resources) & ismember(neg_j, resources);
    valid_indices = is_consumer_consumer | is_resource_resource;
    neg_links = neg_links(valid_indices, :);

    % Debugging: Check available negative links
    disp(['Debug: Negative links available: ', num2str(size(neg_links, 1))]);

    % Ensure enough negative links are available
    total_neg_links_needed = k * (train_size + test_size);
    if size(neg_links, 1) < total_neg_links_needed
        warning('Not enough negative links available. Reducing sample size.');
        k = size(neg_links, 1) / (train_size + test_size);
    end

    % Sample negative links for train and test
    perm = randperm(size(neg_links, 1));
    train_neg = neg_links(perm(1:ceil(k * train_size)), :);
    test_neg = neg_links(perm(ceil(k * train_size) + 1:ceil(k * (train_size + test_size))), :);

    % Sample a portion of links if specified
    if portion < 1
        train_pos = train_pos(1:ceil(size(train_pos, 1) * portion), :);
        train_neg = train_neg(1:ceil(size(train_neg, 1) * portion), :);
        test_pos = test_pos(1:ceil(size(test_pos, 1) * portion), :);
        test_neg = test_neg(1:ceil(size(test_neg, 1) * portion), :);
    elseif portion > 1
        train_pos = train_pos(1:portion, :);
        train_neg = train_neg(1:portion, :);
        test_pos = test_pos(1:portion, :);
        test_neg = test_neg(1:portion, :);
    end
end
