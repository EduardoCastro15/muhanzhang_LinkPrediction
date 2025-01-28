function [data, label] = graph2vector(pos, neg, A, K)
%  Usage: to convert links' enclosing subgraphs (both pos and neg) into real vectors
% Input:
%   - pos: indices of positive links
%   - neg: indices of negative links
%   - A: the observed graph's adjacency matrix from which to extract subgraph features
%   - K: the number of nodes in each link's subgraph
% Output:
%   - data: the constructed training data, each row is a link's vector representation
%   - label: a column vector of links' labels
%
%  *author: Muhan Zhang, Washington University in St. Louis

    all = [pos; neg];
    pos_size = size(pos, 1);
    neg_size = size(neg, 1);
    all_size = pos_size + neg_size;

    % Generate labels
    label = [ones(pos_size, 1); zeros(neg_size, 1)];

    % Allocate space for vectors
    d = K * (K - 1) / 2;  % Dimension of each vector
    data = zeros(all_size, d);

    % Progress display
    one_tenth = max(floor(all_size / 10), 1);
    disp('Subgraph Pattern Encoding Begins...');

    for i = 1:all_size
        ind = all(i, :);
        sample = subgraph2vector(ind, A, K);

        % Validate and ensure sample size matches `d`
        if length(sample) ~= d
            error(['Error: Sample size mismatch at index ', num2str(i), ...
                   '. Expected ', num2str(d), ' but got ', num2str(length(sample)), '.']);
        end

        data(i, :) = sample;

        % Display progress
        if mod(i, one_tenth) == 0
            disp(['Subgraph Pattern Encoding Progress ', num2str(floor(i / one_tenth) * 10), '%...']);
        end
    end
end


function sample = subgraph2vector(ind, A, K)
%  Usage: to extract the enclosing subgraph for a link Aij (i = ind(1), j = ind(2))
%         and impose a vertex ordering for the vertices of the enclosing subgraph using graph labeling
%         and construct an adjacency matrix and output the reshaped vector
% Input:
%   - ind: indices of the link
%   - A: the observed graph's adjacency matrix
%   - K: the number of nodes in the enclosing subgraph
% Output:
%   - sample: the reshaped vector representation of the enclosing subgraph
%
%  *author: Muhan Zhang, Washington University in St. Louis
%

    D = K * (K - 1) / 2;  % The length of the output vector

    % Extract a subgraph of K nodes
    links = [ind];
    dist = 0;
    fringe = [ind];
    nodes = [ind(1); ind(2)];

    while true
        dist = dist + 1;
        fringe = neighbors(fringe, A);  % Get directed neighbors
        fringe = setdiff(fringe, links, 'rows');  % Remove visited links

        if isempty(fringe) || size(nodes, 1) >= K
            subgraph = A(nodes, nodes);
            break;
        end

        new_nodes = setdiff(fringe(:), nodes, 'rows');
        nodes = [nodes; new_nodes];
        links = [links; fringe];
    end

    % Pad the subgraph to K x K if needed
    if size(subgraph, 1) < K
        subgraph = pad_subgraph(subgraph, K);
    end

    % Generate vertex ordering
    order = g_label(subgraph, 7);

    % Trim order and subgraph dimensions to match
    max_size = min(K, size(subgraph, 1));
    order = order(1:max_size);
    subgraph = subgraph(order, order);

    % Handle weighted subgraph and vectorize
    lweight_subgraph = A(nodes, nodes);
    if size(lweight_subgraph, 1) < K
        lweight_subgraph = pad_subgraph(lweight_subgraph, K);
    end
    lweight_subgraph = lweight_subgraph(order, order);

    % Vectorize the upper triangular part of the subgraph
    sample = lweight_subgraph(triu(true(size(subgraph)), 1));

    % Pad or trim `sample` to match expected dimension `D`
    if length(sample) < D
        sample = [sample; zeros(D - length(sample), 1)];
    elseif length(sample) > D
        sample = sample(1:D);
    end

    sample(1) = eps;  % Avoid empty sample
end


function padded_subgraph = pad_subgraph(subgraph, K)
    % Add padding rows and columns to the subgraph
    pad_size = K - size(subgraph, 1);
    padded_subgraph = [subgraph, zeros(size(subgraph, 1), pad_size)];
    padded_subgraph = [padded_subgraph; zeros(pad_size, K)];
end


function N = neighbors(fringe, A);
%  Usage: to find the neighbors of the nodes in the fringe
% Input:
%   - fringe: the nodes to find neighbors
%   - A: the observed graph's adjacency matrix
% Output:
%   - N: the neighbors of the nodes in the fringe
%

    N = [];
    for no = 1:size(fringe, 1)
        ind = fringe(no, :);
        i = ind(1);
        j = ind(2);

        % Outgoing and incoming neighbors
        [~, ij] = find(A(i, :));  % Outgoing edges from i
        [ji, ~] = find(A(:, j));  % Incoming edges to j

        N_out = [i * ones(length(ij), 1), ij'];
        N_in = [ji, j * ones(length(ji), 1)];

        N = unique([N_out; N_in], 'rows', 'stable');  % Preserve directionality
    end
end


function order = g_label(subgraph, p_mo)
%  Usage: impose a vertex order for a enclosing subgraph using graph labeling
% Input:
%   - subgraph: the enclosing subgraph
%   - p_mo: the method to impose vertex order
% Output:
%   - order: the imposed vertex order
%

    if nargin < 2
        p_mo = 7;  % default palette_wl
    end

    % Ensure directed graph object
    disp('Debug: Validating subgraph in g_label (graph2vector.m)...');
    if isequal(subgraph, subgraph')
        disp('Warning: Subgraph has become undirected before g_label.');
    else
        disp('Debug: Subgraph is directed before g_label.');
    end

    % Graph Representation and Distance Calculation
    G = digraph(subgraph);  % Create a directed graph object
    dist_to_1 = distances(G, 1);  % Compute shortest paths from node 1
    dist_to_2 = distances(G, 2);  % Compute shortest paths from node 2

    % Handling Unreachable Nodes
    K = size(subgraph, 1);  % local variable
    dist_to_1(isinf(dist_to_1)) = 2 * K;  % replace inf nodes (unreachable from 1 or 2) by an upperbound dist
    dist_to_2(isinf(dist_to_2)) = 2 * K;

    % Initial Vertex Coloring
    avg_dist = sqrt(dist_to_1 .* dist_to_2);  % use geometric mean as the average distance to the link
    [~, ~, avg_dist_colors] = unique(avg_dist);  % f mapping to initial colors

    % switch different graph labeling methods
    switch p_mo
        case 1
            % use classical wl, no initial colors
            classes = wl_string_lexico(subgraph);
            order = canon(full(subgraph), classes)';
        case 2
            % use wl_hashing, no initial colors
            classes = wl_hashing(subgraph);
            order = canon(full(subgraph), classes)';
        case 3
            % use classical wl, with initial colors
            classes = wl_string_lexico(subgraph, avg_dist_colors);
            order = canon(full(subgraph), classes)';
        case 4
            % use wl_hashing, with initial colors
            classes = wl_hashing(subgraph, avg_dist_colors);
            order = canon(full(subgraph), classes)';
        case 5
            % directly use nauty for canonical labeling
            order = canon(full(subgraph), ones(K, 1))';
        case 6
            % no graph labeling, directly use the predefined order
            order = [1: 1: K];
        case 7
            % palette_wl with initial colors, break ties by nauty
            classes = palette_wl(subgraph, avg_dist_colors);
            order = canon(full(subgraph), classes)';
        case 8
            % random labeling
            order = randperm(K);
    end

    % Trim order to match subgraph dimensions
    max_size = size(subgraph, 1);
    order = order(1:max_size);
end
