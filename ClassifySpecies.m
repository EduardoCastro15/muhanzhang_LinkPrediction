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
    
    % Identify consumers and resources based on labels
    consumers = find(strcmp(species_labels, 'consumer') | strcmp(species_labels, 'consumer-resource'));
    resources = find(strcmp(species_labels, 'resource') | strcmp(species_labels, 'consumer-resource'));

    if isempty(consumers) || isempty(resources)
        warning('One or more species groups are empty. Verify the labels.');
    end

end
