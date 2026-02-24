function windows = updateWeightsAndCA(chunk, windows, selected_indices, predicted_labels, kNN_neighbor_indices, nn1_indices_labels)
    % Step 1: Reduce the weight (end-1 column) by a factor of 0.8 for instances
    % that are not the nearest neighbor of the test instance
    global decy_rate;
    num_instances = size(windows, 1);
    
    % Get the indices of the nearest neighbors
    nearest_neighbor_indices = nn1_indices_labels(:, 1); % Assuming nn1_indices_labels is a matrix of nearest neighbor indices
    
    % Find instances that are not the nearest neighbors
    non_nn_indices = setdiff(1:num_instances, nearest_neighbor_indices);
    
    % Reduce the weight (end-1 column) of these non-nearest neighbors by a factor of 0.8
    windows(non_nn_indices, end-1) = windows(non_nn_indices, end-1) * decy_rate;

    % Extract the true labels of the selected instances
    true_labels = chunk(selected_indices, end);
    
    % Loop through each selected index
    for i = 1:length(selected_indices)
        selected_index = selected_indices(i);
        true_label = true_labels(i);
        
        % Get the nearest neighbors' indices for the selected instance
        nearest_neighbors = kNN_neighbor_indices(selected_index, :);
        
        % Identify the class labels of the nearest neighbors
        neighbor_classes = windows(nearest_neighbors, end-3); % end-3 is the class label column
        
        % Determine matching and non-matching neighbors
        matching_neighbors = neighbor_classes == true_label;
        non_matching_neighbors = ~matching_neighbors;
        
        % Reset the weight to 1 for matching neighbors
        % Extract the weights for the matching neighbors
        p_label=predicted_labels(selected_index);
        if true_label==p_label
            w_rest_1 = windows(nearest_neighbors(matching_neighbors), end-1) * 2;

            % Reset weights to 1 if they exceed 1
            w_rest_1(w_rest_1 > 1) = 1;

            % Update the `windows` array for all matching neighbors
            windows(nearest_neighbors(matching_neighbors), end-1) = w_rest_1;
        else
            % Reduce the weight by half for non-matching neighbors
            windows(nearest_neighbors(non_matching_neighbors), end-1) = ...
                windows(nearest_neighbors(non_matching_neighbors), end-1) * 0.5;

            % Update ca_weight (end column) for non-matching neighbors
           % windows(nearest_neighbors(non_matching_neighbors), end) = ...
               % 1/windows(nearest_neighbors(non_matching_neighbors), end-1) ;
        end
    %1/exp(windows(nearest_neighbors(non_matching_neighbors), end-1)) ;
    end
end
