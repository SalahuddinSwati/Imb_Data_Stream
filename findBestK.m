function [best_k, accuracies] = findBestK(all_predicted_labels, true_labels)
    % Initialize an array to store accuracies for each k
    k_values = fieldnames(all_predicted_labels);
    num_k = length(k_values);
    accuracies = zeros(num_k, 1);
    
    % Calculate accuracy for each k value
    for i = 1:num_k
        k = k_values{i};
        predicted_labels = all_predicted_labels.(k);
        accuracies(i) = sum(predicted_labels == true_labels) / length(true_labels);
    end
    
    % Find the index of the highest accuracy
    [~, best_k_index] = max(accuracies);
    
    % Extract the corresponding k value from k_values
    % The field names are like 'k_1', 'k_3', etc., so we extract the numerical part
    best_k = str2double(k_values{best_k_index}(3:end));
end
