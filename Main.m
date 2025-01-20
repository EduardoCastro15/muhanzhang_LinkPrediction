%  Main Program. Partly adapted from the codes of
%  Lu 2011, Link prediction in complex networks: A survey.
%
%  *author: Muhan Zhang, Washington University in St. Louis
%
%rng(100);

%% Configuration
useParallel = true;         % Flag to enable or disable parallel pool
kRange = 5:15;              % Define the interval of K values to execute
numOfExperiment = 50;
ratioTrain = 0.9;

%% Load food web list from a CSV file or a predefined list
foodweb_list = readtable('data/foodwebs_mat/foodweb_metrics_small.csv');
foodweb_names = foodweb_list.Foodweb;

%% Set up logging
log_dir = 'data/result/';
terminal_log_dir = 'data/result/terminal_logs/';
if ~exist(log_dir, 'dir')
    mkdir(log_dir);
end

% Start parallel pool if the flag is enabled
if useParallel
    % Start parallel pool (initialize once for all datasets)
    if isempty(gcp('nocreate'))
        poolobj = parpool(feature('numcores'));
        % parpool('local', str2double(getenv('NSLOTS')));
    end
end

% Iterate over all food webs in the list
for f_idx = 1:length(foodweb_names)
    dataname = strvcat(foodweb_names{f_idx});  % Get the current food web name
    log_file = strcat(log_dir, dataname, '.txt');  % Create a log file for each food web

    % Check if log file already exists and contains data
    if isfile(log_file) && dir(log_file).bytes > 0
        disp(['Skipping ', dataname, ' as it already has a log file.']);
        continue;  % Skip to the next food web if the log file already exists and has content
    end

    fileID = fopen(log_file, 'a');  % Open the log file for appending text

    % Set up diary to log command window output to a file
    diary_file = fullfile(terminal_log_dir, strcat('terminal_log_', dataname, '.txt'));
    diary(diary_file);  % Start logging to diary
    
    % Add a header to the log file if it doesn't exist
    if ftell(fileID) == 0  % Check if file is empty
        fprintf(fileID, '|========================================================================================================================================================|\n');
        fprintf(fileID, '|    Iteration   |      AUC       |  Time Elapsed  |Encoded subgraph|   Train ratio  |    Threshold   |    Precision   |     Recall     |     F1-Score   |\n');
        fprintf(fileID, '|                |                |   (hh:mm:ss)   |      (K)       |       %%        |                |                |                |                |\n');
        fprintf(fileID, '|========================================================================================================================================================|\n');
    end
    
    %%Load data
    addpath(genpath('utils'));
    datapath = 'data/foodwebs_mat/';
    thisdatapath = fullfile(datapath, strcat(dataname, '.mat'));

    % Check if the .mat file exists
    if ~isfile(thisdatapath)
        disp(['File not found: ', thisdatapath]);
        fclose(fileID);
        diary off;
        continue;
    end
    
    % Load net, species, and classification
    load(thisdatapath, 'net', 'species', 'classification');

    % Classify species
    [consumers, resources] = ClassifySpecies(net, classification);
    
    disp(['Processing dataset: ', dataname]);
    disp(['Consumers: ', num2str(length(consumers)), ' | Resources: ', num2str(length(resources))]);

    % Loop over values of k
    for K = kRange
        disp(['Processing with k = ', num2str(K)]);

        % Pre-allocate cell array to store log entries for each experiment
        log_entries = cell(numOfExperiment, 1);

        if useParallel
            parfor ith_experiment = 1:numOfExperiment
                log_entries{ith_experiment} = processExperiment(ith_experiment, net, ratioTrain, K, consumers, resources);
            end
        else
            for ith_experiment = 1:numOfExperiment
                log_entries{ith_experiment} = processExperiment(ith_experiment, net, ratioTrain, K, consumers, resources);
            end
        end

        % Write accumulated log entries to file after the parfor loop
        for i = 1:numOfExperiment
            fprintf(fileID, '%s', log_entries{i});
        end
    end

    % Log summary results for the current dataset
    avg_auc = mean(cellfun(@(x) sscanf(x, '|%*d | %f', 1), log_entries));
    var_auc = var(cellfun(@(x) sscanf(x, '|%*d | %f', 1), log_entries));
    disp(['Average AUC for dataset ', dataname, ': ', num2str(avg_auc)]);
    disp(['Variance: ', num2str(var_auc)]);
    
    % Clean up memory
    fclose(fileID);
    diary off;
    clear net species classification consumers resources;  % Free memory
end
    
% Close parallel pool after all datasets
if useParallel && exist('poolobj', 'var')
    delete(poolobj);
end
disp(['Execution finished at: ', datestr(now)]);


%% Helper Function for Experiment Processing
function log_entry = processExperiment(ith_experiment, net, ratioTrain, K, consumers, resources)
    % Initialize temporary variables inside the loop
    tempauc = 0;
    iteration_start_time = tic;
    best_threshold = 0;
    best_precision = 0;
    best_recall = 0;
    best_f1_score = 0;

    if mod(ith_experiment, 10) == 0
        disp([num2str(ith_experiment), '%... ']);
    end

    % divide into train/test
    [train, test] = DivideNet(net, ratioTrain);
    train = sparse(train); test = sparse(test);
    train = spones(train + train'); test = spones(test + test');

    % WLNM Method
    disp('WLNM...');
    [tempauc, best_threshold, best_precision, best_recall, best_f1_score] = WLNM(train, test, K, ith_experiment, consumers, resources);

    % Measure time taken for this iteration
    iteration_time = toc(iteration_start_time);  % Time in seconds
    elapsed_time_str = datestr(seconds(iteration_time), 'HH:MM:SS');

    % Store log entry
    log_entry = sprintf('|           %4d |       %8.4f |       %6s |         %6d |        %6d%% |           %.2f |         %.4f |         %.4f |         %.4f |\n', ...
                        ith_experiment, tempauc, elapsed_time_str, K, ratioTrain * 100, best_threshold, best_precision, best_recall, best_f1_score);
end
