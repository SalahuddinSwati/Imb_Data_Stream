function [predicted_labels, kNN_neighbor_indices, neighbor_classes, all_predicted_labels, nn1_indices_labels,c_nearest_labels] = ...
    classifyDataStream(chunk, windows, best_k,C)
    % k-values to consider for predictions
    global  k_values;
    
    % Initialize arrays to hold all training data and labels
  

    % Process the windows matrix
    % Assuming windows is of the form [attributes, labels, label_flag, weights]
    all_data = windows(:, 1:end-4);
    all_labels = windows(:, end-3);

    % Filter rows where label_flage is not zero
    valid_indices = windows(:, end-2) ~= 0;  % Label flag column

    % Extract valid data and labels
    valid_data = all_data(valid_indices, :);  % Exclude weights, labels, and label_flag
    valid_labels = all_labels(valid_indices);  % Label column is third from last

    % Append the original indices of the valid data
    original_indices = find(valid_indices);
    index_map = original_indices;

    % Extract attributes from the chunk (assuming the last column is not a feature)
    test_attributes = chunk(:, 1:end-1);

      K_all = max([k_values, best_k, C]);
    [knn_idx, ~] = knnsearch(valid_data, test_attributes, 'K', K_all);
    knn_labels = valid_labels(knn_idx);
    % Initialize a container for all predicted labels for different k values
    all_predicted_labels = struct();

    % Create and use k-NN classifiers for each k-value
     for k = k_values
        lbls = knn_labels(:, 1:k);

        % Majority vote
        y = mode(lbls, 2);

        % --------------------------------------------------------
        % TIE-BREAKING (distance-consistent)
        % If no strict majority ? fallback to nearest neighbor
        % --------------------------------------------------------
        for i = 1:size(lbls,1)
            counts = histc(lbls(i,:), unique(lbls(i,:)));
            if max(counts) == 1   % all labels unique ? tie
                y(i) = lbls(i,1); % closest neighbor wins
            end
        end

        all_predicted_labels.(sprintf('k_%d', k)) = y;
    end
    % Get the predicted labels for the best_k
    predicted_labels = all_predicted_labels.(sprintf('k_%d', best_k));

   neighbor_indices = knn_idx(:, 1:best_k);
    kNN_neighbor_indices = index_map(neighbor_indices);
    neighbor_classes = all_labels(kNN_neighbor_indices);

    % Find nearest neighbors using 1-NN (K=1) including all data in windows
    [nn1_indices, ~] = knnsearch(all_data, test_attributes, 'K', 1);
    nn1_labels = all_labels(nn1_indices);
    nn1_indices_labels = [nn1_indices, nn1_labels];
     % C-nearest neighbor labels (for uncertainty / consistency)
    % ------------------------------------------------------------
    c_nearest_labels = valid_labels(knn_idx(:, 1:C));
    
end
