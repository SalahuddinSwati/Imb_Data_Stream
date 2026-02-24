function [windows, best_k] = updateWindows(chunk, windows, predicted_labels, kNN_neighbor_indices, neighbor_classes, all_predicted_labels, nn1_indices_labels, c_nearest_labels, bugdt, w,best_kk)
   

global  k_values;

    [Uncertainty_selected_indices, certain_indices] = selectInstancesByUncertainty_Majority(chunk, c_nearest_labels, bugdt);
%     end
    % Extract class labels from nearest neighbor indices
    windows(:, end) = 1./windows(:, end-1);  
    class_labels = nn1_indices_labels(:, 2);
    
    % Get unique class labels present in the data
    unique_classes = unique(class_labels);
    
    % Compute class distribution
    [class_counts, ~] = histcounts(class_labels, [unique_classes; max(unique_classes)+1]);

    % Compute imbalance ratio (higher for underrepresented classes)
    imb_ratio = 1 ./ (class_counts ./ (sum(class_counts) / length(unique_classes)));

    % Initialize instance weights as 1
    weight_im = ones(length(class_labels), 1);

    % Create a class imbalance map (weights > 1 for minority classes)
    imb_ratio_class_map = imb_ratio .* (imb_ratio > 1);

    % Assign weights to instances based on class imbalance
    [~, class_indices] = ismember(class_labels, unique_classes);
    weight_im = imb_ratio_class_map(class_indices);
    weight_im = weight_im(:);


    for kk=1:best_kk
    windows(kNN_neighbor_indices(:, kk), end) = windows(kNN_neighbor_indices(:, kk), end) + weight_im;
    end
  
    CA_selected_indices = selectInstancesByCAWeight(chunk, kNN_neighbor_indices,nn1_indices_labels, windows, bugdt, Uncertainty_selected_indices,best_kk);

    % Reset the last column of windows (CA weight) after selection
    windows(:, end) = 0;

   
    num_to_select = round(bugdt / 100 * size(chunk, 1)); % Calculate total instances to select
    total_selected = length(Uncertainty_selected_indices) + length(CA_selected_indices); % Count selected instances

    % Adjust the remaining budget dynamically
    if total_selected < (2/3) * num_to_select
        remaining_budget = num_to_select - total_selected;
    else
        remaining_budget = round(num_to_select / 3);
    end

  
    Random_selected_indices = selectInstancesRandomly(chunk, remaining_budget, Uncertainty_selected_indices, CA_selected_indices);

    % Combine all selected indices
    selected_indices = [Uncertainty_selected_indices', CA_selected_indices, Random_selected_indices];

    % Ensure the final selected list does not exceed the number to select
    selected_indices = selected_indices(1:num_to_select);

   
    windows = updateWeightsAndCA(chunk, windows, selected_indices, predicted_labels, kNN_neighbor_indices, nn1_indices_labels);

    % Remove instances with very low weight (< 0.065) to prevent unnecessary storage
    windows(windows(:, end-1) < 0.065, :) = [];


    selected_data = chunk(selected_indices, 1:end-1);
    selected_labels = chunk(selected_indices, end);

    % Create a new matrix for selected instances with metadata:
    % Format: [features, label, flag (1 for selected), weight (1), CA weight (0)]
    new_rows_selected_indices = [selected_data, selected_labels, ones(length(selected_indices), 1), ones(length(selected_indices), 1), zeros(length(selected_indices), 1)];

    % Identify certain instances that were not selected via random/CA weight methods
    certain_final = setdiff(certain_indices, [Random_selected_indices, CA_selected_indices]);

    % Extract feature data and predicted labels for certain instances
    certain_data = chunk(certain_final, 1:end-1);
    certain_predicted_labels = predicted_labels(certain_final);

    % Format: [features, predicted label, flag (1 for certain), weight (1), CA weight (0)]
    new_rows_certain = [certain_data, certain_predicted_labels, ones(length(certain_final), 1), ones(length(certain_final), 1), zeros(length(certain_final), 1)];

    % Identify remaining instances that were not selected
    remaining_indices = setdiff(1:size(chunk, 1), [selected_indices, certain_final']);

    % Extract feature data and predicted labels for remaining instances
    remaining_data = chunk(remaining_indices, 1:end-1);
    remaining_predicted_labels = nn1_indices_labels(remaining_indices, 2);

    % Format: [features, predicted label, flag (0 for unselected), weight (1), CA weight (0)]
    new_rows_remaining = [remaining_data, remaining_predicted_labels, zeros(length(remaining_indices), 1), ones(length(remaining_indices), 1), zeros(length(remaining_indices), 1)];

    new_data = [new_rows_selected_indices; new_rows_certain; new_rows_remaining];

    % Append the data with merging logic to manage window limits
    windows = appendDataWithMerging(windows, new_data, w);

    

%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    selected_predicted_labels = structfun(@(x) x(selected_indices), all_predicted_labels, 'UniformOutput', false);
%
    % Step 2: Convert struct to a matrix (each column corresponds to a k-value)
    selected_predicted_matrix = horzcat(selected_predicted_labels.k_1, ...
                                        selected_predicted_labels.k_3, ...
                                        selected_predicted_labels.k_5, ...
                                        selected_predicted_labels.k_7);

  
    accuracy_scores = sum(selected_predicted_matrix == selected_labels, 1) ./ length(selected_labels);

    [~, best_k_idx] = max(accuracy_scores);
   
    best_k = k_values(best_k_idx);


end
