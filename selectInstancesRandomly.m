function selected_indices = selectInstancesRandomly(chunk, remaining_budget, Uncertainty_selected_indices, CA_selected_indices)
    % Total number of instances in the chunk
    num_instances = size(chunk, 1);
    
    % Combine the indices that have already been selected
    excluded_indices = union(Uncertainty_selected_indices, CA_selected_indices);
    
    % Get the indices of instances that were not selected
    valid_indices = setdiff(1:num_instances, excluded_indices);
    
    % Adjust the number to select in case it's greater than available
    num_to_select = min(remaining_budget, numel(valid_indices));
    
    % Randomly sample from the remaining valid indices
    selected_indices = randsample(valid_indices, num_to_select);
end
