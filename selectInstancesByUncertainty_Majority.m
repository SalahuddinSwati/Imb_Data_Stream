function [selected_indices, certain_indices] = selectInstancesByUncertainty_Majority(chunk, c_nearest_labels, bugdt)
    % Total number of instances in the chunk
    num_instances = size(chunk, 1);
    global uncertain_K;
    % Get the labels for the C-nearest neighbors for all instances
    labels_matrix = c_nearest_labels;

    % Find the unique classes in the nearest labels
%     unique_classes = unique(labels_matrix(:));
%     uncertain_K = numel(unique_classes);

    % Precompute the threshold for uncertain instances
    threshold = ceil(uncertain_K / 2) + 1;

    % Find the majority class for each instance (mode across th\efe neighbors)
    majority_classes = mode(labels_matrix, 2);  % Get the mode (majority class) for each instance

    % Count the occurrences of the majority class for each instance
    majority_counts = sum(labels_matrix == majority_classes, 2);  % Count occurrences of majority class

    % Identify uncertain instances (where majority count is below the threshold)
    uncertain_mask = majority_counts < threshold;

    % Get the indices of uncertain instances
    uncertain_indices = find(uncertain_mask);

    % Get the indices of certain instances (where majority count is >= threshold)
    threshold_cer = ceil((uncertain_K*2)/3);
    certain_mask = majority_counts > threshold_cer;
    certain_indices = find(certain_mask);

    % Calculate how many instances to select
    num_to_select = round(bugdt / 100 * num_instances);

    % Calculate how many instances to randomly select from uncertain instances
    num_to_random_select = round(num_to_select / 3);

    % Randomly select the minimum of 'num_to_random_select' and the number of uncertain instances available
    selected_indices = randsample(uncertain_indices, min(num_to_random_select, numel(uncertain_indices)));
end
