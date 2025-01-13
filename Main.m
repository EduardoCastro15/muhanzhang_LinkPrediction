%  Main Program. Partly adapted from the codes of
%  Lu 2011, Link prediction in complex networks: A survey.
%
%  *author: Muhan Zhang, Washington University in St. Louis

%rng(100);

%% Load food web list from a CSV file or a predefined list
foodweb_list = readtable('data/foodwebs_mat/foodweb_metrics_small.csv');
foodweb_names = foodweb_list.Foodweb;

%% Set up logging
log_dir = 'data/result/';
terminal_log_dir = 'data/result/terminal_logs/';
if ~exist(log_dir, 'dir')
    mkdir(log_dir);
end

% Start parallel pool (initialize once for all datasets)
if isempty(gcp('nocreate'))
    poolobj = parpool(feature('numcores'));
    % parpool('local', str2double(getenv('NSLOTS')));
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
    load(thisdatapath, 'net');  % Load only 'net' variable to save memory
    numOfExperiment = 50;
    ratioTrain = 0.9;
    % method = [1, 2, 3, 4, 5, 6];  % 1: WLNM,  2: common-neighbor-based,  3: path-based, 4: random walk  5: latent-feature-based,  6: stochastic block model
    method = [1];
    num_in_each_method = [1, 13, 6, 13, 1, 1];  % how many algorithms in each type of method
    num_of_methods = sum(num_in_each_method(method));  % the total number of algorithms
    
    disp(['Processing dataset: ', dataname]);

    % Loop over values of k
    for K = 5:15
        disp(['Processing with k = ', num2str(K)]);

        % Pre-allocate cell array to store log entries for each experiment
        log_entries = cell(numOfExperiment, 1);

        parfor ith_experiment = 1:numOfExperiment
            % Initialize temporary variables inside the parfor loop
            tempauc = 0;
            iteration_start_time = tic;
            
            % Initialize the metrics within the parfor loop
            best_threshold = 0;
            best_precision = 0;
            best_recall = 0;
            best_f1_score = 0;

            ith_experiment
            if mod(ith_experiment, 10) == 0
                tempcont = strcat(int2str(ith_experiment),'%... ');
                disp(tempcont);
            end

            % divide into train/test
            [train, test] = DivideNet(net,ratioTrain);
            train = sparse(train); test = sparse(test);
            train = spones(train + train'); test = spones(test + test');
            
            % Process methods
            ithAUCvector = zeros(1, num_of_methods); % Pre-allocate space for AUC vector
            Predictors = [];
            
            % run link prediction methods
            %% Weisfeiler-Lehman Neural Machine (WLNM)
            if ismember(1, method)
                disp('WLNM...');
                [tempauc, best_threshold, best_precision, best_recall, best_f1_score] = WLNM(train, test, K, ith_experiment);                  % WLNM
                Predictors = [Predictors '%WLNM	'];
                ithAUCvector(1) = tempauc;
            end
    
            %% Common Neighbor-based methods, 13 methods
            if ismember(2, method)
                disp('CN...');
                tempauc = CN(train, test);                  % Common Neighbor
                Predictors = [Predictors 'CN	'];      ithAUCvector = [ithAUCvector tempauc];
    
                disp('Salton...');
                tempauc = Salton(train, test);              % Salton Index
                Predictors = [Predictors 'Salton	'];  ithAUCvector = [ithAUCvector tempauc];
    
                disp('Jaccard...');
                tempauc = Jaccard(train, test);             % Jaccard Index
                Predictors = [Predictors 'Jaccard	'];  ithAUCvector = [ithAUCvector tempauc];
    
                disp('Sorenson...');
                tempauc = Sorenson(train, test);            % Sorenson Index
                Predictors = [Predictors 'Sorens	'];   ithAUCvector = [ithAUCvector tempauc];
    
                disp('HPI...');
                tempauc = HPI(train, test);                 % Hub Promoted Index
                Predictors = [Predictors 'HPI	'];       ithAUCvector = [ithAUCvector tempauc];
    
                disp('HDI...');
                tempauc = HDI(train, test);                 % Hub Depressed Index
                Predictors = [Predictors 'HDI	'];       ithAUCvector = [ithAUCvector tempauc];
    
                disp('LHN...');
                tempauc = LHN(train, test);                 % Leicht-Holme-Newman
                Predictors = [Predictors 'LHN	'];       ithAUCvector = [ithAUCvector tempauc];
    
                disp('AA...');
                tempauc = AA(train, test);                  % Adar-Adamic Index
                Predictors = [Predictors 'AA	'];       ithAUCvector = [ithAUCvector tempauc];
    
                disp('RA...');
                tempauc = RA(train, test);                  % Resourse Allocation
                Predictors = [Predictors 'RA	'];       ithAUCvector = [ithAUCvector tempauc];
    
                disp('PA...');
                tempauc = PA(train, test);                  % Preferential Attachment
                Predictors = [Predictors 'PA	'];       ithAUCvector = [ithAUCvector tempauc];
    
                disp('LNBCN...');
                tempauc = LNBCN(train, test);               % Local naive bayes method - Common Neighbor
                Predictors = [Predictors 'LNBCN	'];       ithAUCvector = [ithAUCvector tempauc];
    
                disp('LNBAA...');
                tempauc = LNBAA(train, test);               % Local naive bayes method - Adar-Adamic Index
                Predictors = [Predictors 'LNBAA	'];       ithAUCvector = [ithAUCvector tempauc];
    
                disp('LNBRA...');
                tempauc = LNBRA(train, test);               % Local naive bayes method - Resource Allocation
                Predictors = [Predictors 'LNBRA	'];       ithAUCvector = [ithAUCvector tempauc];
            end
        
            %% Path-based methods, 6 methods
            if ismember(3, method)
                disp('LocalPath...');
                tempauc = LocalPath(train, test, 0.0001);   % Local Path Index
                Predictors = [Predictors 'LocalP	'];   ithAUCvector = [ithAUCvector tempauc];
    
                disp('Katz 0.01...');
                tempauc = Katz(train, test, 0.01);          % Katz Index, beta=0.01
                Predictors = [Predictors 'Katz.01	'];   ithAUCvector = [ithAUCvector tempauc];
    
                disp('Katz 0.001...');
                tempauc = Katz(train, test, 0.001);         % Katz Index, beta=0.001
                Predictors = [Predictors '~.001	'];       ithAUCvector = [ithAUCvector tempauc];
    
                disp('LHNII 0.9...');
                tempauc = LHNII(train, test, 0.9);          % Leicht-Holme-Newman II
                Predictors = [Predictors 'LHNII.9	'];    ithAUCvector = [ithAUCvector tempauc];
    
                disp('LHNII 0.95...');
                tempauc = LHNII(train, test, 0.95);         % Leicht-Holme-Newman II
                Predictors = [Predictors '~.95	'];       ithAUCvector = [ithAUCvector tempauc];
    
                disp('LHNII 0.99...');
                tempauc = LHNII(train, test, 0.99);         % Leicht-Holme-Newman II
                Predictors = [Predictors '~.99	'];       ithAUCvector = [ithAUCvector tempauc];
            end
    
            %% Random walk-based Methods, 13 methods
            if ismember(4, method)
                disp('ACT...');
                tempauc = ACT(train, test);                 % Average commute time
                Predictors = [Predictors 'ACT	'];       ithAUCvector = [ithAUCvector tempauc];
    
                disp('CosPlus...');
                tempauc = CosPlus(train, test);             % Cos+ based on Laplacian matrix
                Predictors = [Predictors 'CosPlus	'];   ithAUCvector = [ithAUCvector tempauc];
    
                disp('RWR 0.85...');
                tempauc = RWR(train, test, 0.85);           % Random walk with restart (PageRank), d=0.85
                Predictors = [Predictors 'RWR.85	'];   ithAUCvector = [ithAUCvector tempauc];
    
                disp('RWR 0.95...');
                tempauc = RWR(train, test, 0.95);           % Random walk with restart, d=0.95
                Predictors = [Predictors '~.95	'];      ithAUCvector = [ithAUCvector tempauc];
    
                disp('SimRank 0.6...');
                tempauc = SimRank(train, test, 0.6);        % SimRank
                Predictors = [Predictors 'SimR	'];      ithAUCvector = [ithAUCvector tempauc];
    
                disp('LRW 3...');
                tempauc = LRW(train, test, 3, 0.85);        % Local random walk, step 3
                Predictors = [Predictors 'LRW_3	'];       ithAUCvector = [ithAUCvector tempauc];
    
                disp('LRW 4...');
                tempauc = LRW(train, test, 4, 0.85);        % Local random walk, step 4
                Predictors = [Predictors '~_4	'];       ithAUCvector = [ithAUCvector tempauc];
    
                disp('LRW 5...');
                tempauc = LRW(train, test, 5, 0.85);        % Local random walk, step 5
                Predictors = [Predictors '~_5	'];       ithAUCvector = [ithAUCvector tempauc];
    
                disp('SRW 3...');
                tempauc = SRW(train, test, 3, 0.85);        % Superposed random walk, step 3
                Predictors = [Predictors 'SRW_3	'];       ithAUCvector = [ithAUCvector tempauc];
    
                disp('SRW 4...');
                tempauc = SRW(train, test, 4, 0.85);        % Superposed random walk, step 4
                Predictors = [Predictors '~_4	'];       ithAUCvector = [ithAUCvector tempauc];
    
                disp('SRW 5...');
                tempauc = SRW(train, test, 5, 0.85);        % Superposed random walk, step 5
                Predictors = [Predictors '~_5	'];       ithAUCvector = [ithAUCvector tempauc];
    
                disp('MFI...');
                tempauc = MFI(train, test);                 % Matrix forest Index
                Predictors = [Predictors 'MFI	'];       ithAUCvector = [ithAUCvector tempauc];
    
                disp('TS...');
                tempauc = TSCN(train, test, 0.01);          % Transfer similarity - Common Neighbor
                Predictors = [Predictors 'TSCN	'];       ithAUCvector = [ithAUCvector tempauc];
            end
    
            %% latent feature models
            if ismember(5, method)
                disp('MF...');
                tempauc = MF(train, test, 5, ith_experiment);                 % matrix factorization
                Predictors = [Predictors 'MF	'];       ithAUCvector = [ithAUCvector tempauc];
            end
    
            %% latent feature models
            if ismember(6, method)
                disp('SBM...');
                tempauc = SBM(train, test, 12);                 % stochastic block models
                Predictors = [Predictors 'SBM	'];       ithAUCvector = [ithAUCvector tempauc];
            end
            
            %% Store AUC results for each experiment
            aucOfallPredictor(ith_experiment, :) = ithAUCvector;
            % PredictorsName = Predictors;
    
            % Measure time taken for this iteration
            iteration_time = toc(iteration_start_time);  % Time in seconds
            elapsed_time_str = datestr(seconds(iteration_time), 'HH:MM:SS');

            % Store log entry in cell array
            log_entries{ith_experiment} = sprintf('|           %4d |       %8.4f |       %6s |         %6d |        %6d%% |           %.2f |         %.4f |         %.4f |         %.4f |\n', ...
                    ith_experiment, tempauc, elapsed_time_str, K, ratioTrain * 100, best_threshold, best_precision, best_recall, best_f1_score);
        end

        % Write accumulated log entries to file after the parfor loop
        for i = 1:numOfExperiment
            fprintf(fileID, '%s', log_entries{i});
        end
    end

    % Log summary results for the current dataset
    avg_auc = mean(aucOfallPredictor, 1);
    var_auc = var(aucOfallPredictor, 0, 1);
    disp(['Average AUC for dataset ', dataname, ': ', num2str(avg_auc)]);
    disp(['Variance: ', num2str(var_auc)]);
    
    % Clean up memory
    fclose(fileID);
    diary off;
    clear net aucOfallPredictor; % Clear large variables after each dataset
end
    
% Close parallel pool after all datasets
if exist('poolobj', 'var')
    delete(poolobj);
end
disp(['Execution finished at: ', datestr(now)]);
