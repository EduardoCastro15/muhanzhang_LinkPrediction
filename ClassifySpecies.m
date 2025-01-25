function [consumers, resources] = ClassifySpecies(net, species_labels)
% Classifies nodes in the network into consumers and resources
%
% Input:
%   - net: adjacency matrix of the network
%   - species_labels: vector or cell array indicating species type for each node
%       (e.g., {'consumer', 'resource', 'consumer', ...} or [1, 2, 1, ...])
%
% Output:
%   - consumers: vector of node indices classified as consumers
%   - resources: vector of node indices classified as resources
%
% Usage:
%   [consumers, resources] = ClassifySpecies(net, species_labels);

    if isempty(species_labels) || length(species_labels) ~= size(net, 1)
        error('Species labels must match the number of nodes in the network.');
    end

    % Check for graph symmetry
    if issymmetric(net)
        warning('The adjacency matrix is undirected. Directed edges are expected for proper classification.');
    end
    
    % Classify species based on labels
    if isnumeric(species_labels)
        consumers = find(species_labels == 1 | species_labels == 3); % Numeric encoding
        resources = find(species_labels == 2 | species_labels == 3);
    elseif iscell(species_labels)
        consumers = find(strcmp(species_labels, 'consumer') | strcmp(species_labels, 'consumer-resource'));
        resources = find(strcmp(species_labels, 'resource') | strcmp(species_labels, 'consumer-resource'));
    else
        error('Species labels must be a numeric vector or a cell array of strings.');
    end

    % Debug: Check if groups are empty
    if isempty(consumers) || isempty(resources)
        warning('One or more species groups are empty. Verify the labels.');
    end

    % Debug: Check for isolated nodes
    isolated_nodes = find(sum(net, 1) == 0 & sum(net, 2)' == 0);
    if ~isempty(isolated_nodes)
        warning(['Isolated nodes detected: ', num2str(length(isolated_nodes)), ' nodes have no connections.']);
    end

end
