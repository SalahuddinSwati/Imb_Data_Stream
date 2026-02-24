clc;
%clear all;
close all;
clear all;

load('dataset1.mat');
load('dataset2.mat');

datasets={};
datasets_names={'dataset1','dataset2'};

summary_results = {};

for dt = 1:length(datasets)
    data = datasets{dt};
    dataset_name = datasets_names{dt};
    
    % Extract class labels (assuming last column is the class label)
    class_labels = data(:, end);
    
    % Count instances per class
    [unique_classes, ~, idx] = unique(class_labels);
    class_counts = accumarray(idx, 1);  % Count occurrences of each class
    total_instances = length(class_labels);
    
    % Compute class percentages (vectorized)
    class_percentages = (class_counts / total_instances) * 100;
    
    % Concatenate results in one go (no explicit loop)
    summary_results = [summary_results; repmat({dataset_name}, length(unique_classes), 1), ...
        num2cell(unique_classes), num2cell(class_counts), num2cell(class_percentages)];
end

% Convert to table
summary_table = cell2table(summary_results, 'VariableNames', {'Dataset', 'Class', 'Count', 'Percentage'});

% Define CSV filename
summary_csv_filename = '\Results\Class_Distribution_Summary.csv';

% Save to CSV
writetable(summary_table, summary_csv_filename);

disp(['? Class distribution summary saved at: ' summary_csv_filename]);


% Parameters
global uncertain_K;
chunk_size = 500;
win_size = 500;
global decy_rate;
 decy_rate=0.8;

best_k =1; %initial value for best_k
% Define label percentages
label_per_values = [5,10,20,98];
instanc_to_search_for_w = 2000;
global  k_values;
k_values = [1, 3, 5, 7,9];
% Initialize structure to store results across all label_per values
summary_results_combined = struct();
result_fre=500;

for lp_idx = 1:length(label_per_values)
    label_per=label_per_values(lp_idx);
    
        
    disp(['Processing for label_per = ', num2str(label_per)]);
    
    for dt = 1:length(datasets)
        data = datasets{dt};
        dataset_name = datasets_names{dt};
        disp(['Running for Dataset: ', dataset_name, ' with Label %: ', num2str(label_per)]);
        
        results_table = [];
        
        attributes = data(:, 1:end-1);
        class_labels = data(:, end);
        unique_classes = unique(class_labels);
        
       
        windows = [];
        used_indices = [];
        
        for i = 1:length(unique_classes)
            class = unique_classes(i);
            idx = find(class_labels(1:instanc_to_search_for_w) == class);
            idx = idx(1:min(length(idx), win_size));
            class_data = attributes(idx, :);
            class_labels_subset = class_labels(idx);
            weights = ones(size(class_data, 1), 1);
            label_flg = weights;
            ca_weights = zeros(size(class_data, 1), 1);
            windows = [windows; [class_data, class_labels_subset, label_flg, weights, ca_weights]];
            used_indices = [used_indices; idx];
        end
        
        remaining_indices = setdiff((1:size(data, 1))', used_indices);
        stream_data = data(remaining_indices, :);
        current_index = 1;
        total_instances = size(stream_data, 1);
        
        accumulated_true_labels = [];
        accumulated_predicted_labels = [];
        best_k_per_chunk = []; % Store one best_k per chunk
        chunk_processing_time_per_chunk = []; % Store one processing time per chunk
        
        while current_index <= total_instances
            end_index = min(current_index + chunk_size - 1, total_instances);
            chunk = stream_data(current_index:end_index, :);
%             
            if best_k == 1
                uncertain_K = 3;
            else
                uncertain_K = best_k * 2;
            end
            
            % Classification step
            tic;
            [predicted_labels, kNN_neighbor_indices, neighbor_classes, all_predicted_labels, ...
                nn1_indices_labels, c_nearest_labels] = classifyDataStream(chunk, windows, best_k, uncertain_K);
            best_k_per_chunk = [best_k_per_chunk;best_k];
            % Update Windows and best_k
            [windows, best_k] = updateWindows(...
                chunk, windows, predicted_labels, kNN_neighbor_indices, neighbor_classes, all_predicted_labels, ...
                nn1_indices_labels, c_nearest_labels, label_per, win_size,best_k);
            
%            
            % ? Store processing time for this chunk
            chunk_processing = toc;
            chunk_processing_time_per_chunk = [chunk_processing_time_per_chunk; chunk_processing];
            
           
           
            % Store true and predicted labels
            true_labels = chunk(:, end);
            accumulated_true_labels = [accumulated_true_labels; true_labels];
            accumulated_predicted_labels = [accumulated_predicted_labels; predicted_labels];
            
            current_index = end_index + 1;
            disp(current_index);
        end  % End of while loop
        
        % ? After processing all chunks, calculate performance metrics every 1000 instances
        disp('Computing results after the entire stream is processed...');
        num_instances = length(accumulated_true_labels);
        chunk_idx = 1; % Index for best_k_per_chunk
        for i = 1:result_fre:num_instances
            batch_end = min(i + (result_fre-1), num_instances);
            disp(['Processing accumulated instances from ' num2str(i) ' to ' num2str(batch_end)]);
            
            % Extract subset
            batch_true_labels = accumulated_true_labels(i:batch_end);
            batch_predicted_labels = accumulated_predicted_labels(i:batch_end);
            
            % ? Retrieve the correct `best_k` and `processing_time` for this batch
            if chunk_idx <= length(best_k_per_chunk)
                batch_best_k = best_k_per_chunk(chunk_idx); % Use stored best_k value
                batch_processing_time = chunk_processing_time_per_chunk(chunk_idx); % Use stored processing time
            else
                batch_best_k = best_k_per_chunk(end); % Use last stored best_k if index goes out of range
                batch_processing_time = chunk_processing_time_per_chunk(end); % Use last processing time
            end
            chunk_idx = chunk_idx + 1; % Move to next chunk index
            
            % Calculate performance
            [accuracy, gMean, f1Score, kappa] = calculatePerformanceMetrics(batch_predicted_labels, batch_true_labels);
            
            disp('Accuracy');
            disp(accuracy);
            
            
            % Store results 
            results_table = [results_table; batch_best_k, batch_processing_time, accuracy, gMean, f1Score, kappa];
        end
        
        
        if ~isempty(results_table)
            avg_best_k = mean(results_table(:,1));
            avg_processing_time = mean(results_table(:,2));
            avg_accuracy = mean(results_table(:,3));
            avg_gMean = mean(results_table(:,4));
            avg_f1Score = mean(results_table(:,5));
            
            kappa_values = results_table(:,6);
            kappa_values = kappa_values(~isnan(kappa_values));
            avg_kappa = mean(kappa_values, 'omitnan');
            
            % **Save individual dataset results**
            results_dir = fullfile('\Results\', num2str(label_per));
            if ~exist(results_dir, 'dir')
                mkdir(results_dir);
            end
            csv_filename = fullfile(results_dir, [dataset_name '.csv']);
            writetable(array2table(results_table, 'VariableNames', {'best_k', 'processing_time', 'accuracy', 'gMean', 'f1Score', 'kappa'}), csv_filename);
            
            % **Store results for final summary**
            str = num2str(label_per);  % Convert to string
%             str = strrep(str, '0', '');  % Remove '0'
%             str = strrep(str, '.', '');
              summary_results_combined.(dataset_name).(['decay_' str '_Acc']) = avg_accuracy;
            summary_results_combined.(dataset_name).(['decay_' str '_GMean']) = avg_gMean;
            summary_results_combined.(dataset_name).(['decay_' str '_F1Score']) = avg_f1Score;
            summary_results_combined.(dataset_name).(['decay_' str '_Kappa']) = avg_kappa;

        end
    end
end

% Convert combined results into a table only if results exist
if ~isempty(fieldnames(summary_results_combined))
    dataset_names_list = fieldnames(summary_results_combined);  % Get dataset names
    
    % Create an empty cell array to store data
    summary_data = {};
    
    % Get the correct column order
    label_per_values_str = arrayfun(@(x) num2str(x), label_per_values, 'UniformOutput', false);
    correct_column_order = {};
    
    for i = 1:length(label_per_values_str)
        lp = label_per_values_str{i};
        str = lp;  % Convert to string
%             str = strrep(str, '0', '');  % Remove '0'
%             str = strrep(str, '.', '');
        correct_column_order = [correct_column_order, ...
            ['decay_' str '_Acc'], ...
            ['decay_' str '_GMean'], ...
            ['decay_' str '_F1Score'], ...
            ['decay_' str '_Kappa']];
    end
    
    % Populate summary_data with dataset names and their respective results
    for i = 1:length(dataset_names_list)
        dataset_name = dataset_names_list{i};
        
        % Extract dataset results
        result_row = summary_results_combined.(dataset_name);
        
        % Ensure results are in the correct order
       result_values = cell(size(correct_column_order));

        for c = 1:length(correct_column_order)
            field = correct_column_order{c};
             if isfield(result_row, field)
                result_values{c} = result_row.(field);
             else
                result_values{c} = NaN;   % missing decay ? NaN
             end
        end
        
        % Store the dataset name along with its results
        summary_data = [summary_data; [{dataset_name}, result_values]];
    end
    
    % Convert to a table with proper column names
    column_names = [{'Dataset_Name'}, correct_column_order]; % Dataset name + metric names in correct order
    summary_combined_table = cell2table(summary_data, 'VariableNames', column_names);
    
    % Define final summary CSV file path
    final_summary_csv = fullfile('\Results\', 'Final_Summary_All_Decay.csv');
    
    % Save final combined summary to CSV
    writetable(summary_combined_table, final_summary_csv);
    disp(['? Final summary results saved at: ' final_summary_csv]);
    
else
    disp('?? No summary results were generated. Final summary file was not created.');
end
