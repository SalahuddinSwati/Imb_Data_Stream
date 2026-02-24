function selected_indices = selectInstancesByCAWeight(chunk, kNN_neighbor_indices,nn1_indices_labels, windows, bugdt, Uncertainty_selected_indices,best_kk)
    % Total number of instances in the chunk
    num_instances = size(chunk, 1);

    % Exclude the instances that were selected by the Uncertainty strategy
    valid_indices = setdiff(1:num_instances, Uncertainty_selected_indices);
    ca_weights = windows(:, end); 
    % Extract the nearest neighbor indices for valid instances
    nn_ca_weights=[];
 
    for kk=1:1%best_kk
    nn_indices = kNN_neighbor_indices(valid_indices, kk); 
    %nn_indices = nn1_indices_labels(valid_indices, 1);  % Extract the indices of nearest neighbors for valid instances
     nn_ca_weights=[nn_ca_weights,ca_weights(nn_indices)];
    
  
    end
    
    nn_ca_weights_avg = mean(nn_ca_weights,2);  % Using nn_indices to get the corresponding ca_weights

    % Sort the valid instances by ca_weight in descending order
    [~, sorted_indices] = sort(nn_ca_weights_avg, 'descend');
    sorted_valid_indices = valid_indices(sorted_indices);

    % Filter out the instances with ca_weight <= 0
    positive_indices = sorted_valid_indices(nn_ca_weights_avg > 0);  % Use nn_ca_weights directly for filtering

    % Select 1/3 of the budget (bugdt percentage of the total instances)
    num_to_select = round(bugdt / 100 * num_instances);
    num_select = round(num_to_select / 3);

    % Ensure that we select only from the filtered indices (with ca_weight > 0)
    selected_indices = positive_indices(1:min(num_select, length(positive_indices)));
end
