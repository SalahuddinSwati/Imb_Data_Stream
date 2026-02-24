function windows = appendDataWithMerging(windows, new_data, w)
    % Function to append data to windows while ensuring the class limits are not exceeded.
    % Merges excess instances if necessary by averaging feature data and updating flags, weights, and CA weights.
    %
    % Inputs:
    %   windows     - Current data in the windows (matrix)
    %   new_data    - New data to append to windows (matrix)
    %   w           - Class limit (maximum number of instances allowed per class)
    %
    % Output:
    %   windows     - Updated windows after appending and merging excess instances.

    % Extract class labels from new_data
    new_class_labels = new_data(:, end-3);  % Assuming class labels are in the 4th last column of new_data
    
    % Get unique classes in new_data
    unique_classes = unique(new_class_labels);

    for class_idx = 1:length(unique_classes)
        class_label = unique_classes(class_idx);

        % Extract class-specific instances in windows and new_data
        if ~isempty(windows) && any(windows(:, end-3) == class_label)
            % Extract class instances if windows is not empty and class exists
            class_instances_in_windows = windows(windows(:, end-3) == class_label, :);  
        else
            % Set to empty if no matching class exists in windows
            class_instances_in_windows = [];
        end
        
        class_instances_in_new_data = new_data(new_class_labels == class_label, :);  % Instances of current class in new data

        % Count the instances of the class in windows and new_data
        num_instances_in_windows = size(class_instances_in_windows, 1);
        num_instances_in_new_data = size(class_instances_in_new_data, 1);

        total_class_instances = num_instances_in_windows + num_instances_in_new_data;

        if total_class_instances > w
            % If the total instances exceed the limit, merge excess instances in windows

            excess_count = total_class_instances - w;  % Number of excess instances

            % 1. Get feature data for instances in `windows` of the current class
            features_in_windows = class_instances_in_windows(:, 1:end-4);  % Exclude the last 4 columns (label, flag, weight, CA weight)

            % 2. Calculate pairwise Euclidean distances between all instances in `windows` (same class)
            pairwise_distances = pdist2(features_in_windows, features_in_windows, 'euclidean');

            % 3. Set diagonal to inf (we don’t want to merge an instance with itself)
            pairwise_distances(1:size(pairwise_distances, 1) + 1:end) = inf;

            % 4. Find the nearest pairs based on minimum pairwise distance
            [sorted_distances, sorted_indices] = sort(pairwise_distances(:));  % Sort all pairwise distances
            [pair_indices_1, pair_indices_2] = ind2sub(size(pairwise_distances), sorted_indices);  % Get the index pairs of the nearest neighbors

            % 5. Remove duplicate pairs (i.e., instances that are already merged)
            merged_instances = false(size(features_in_windows, 1), 1);  % Track merged instances
            unique_pairs = [];  % Store unique pairs to merge

            % We now find the excess_count unique pairs
            for i = 1:length(pair_indices_1)
                idx1 = pair_indices_1(i);
                idx2 = pair_indices_2(i);

                % If neither of the instances has been merged before
                if ~merged_instances(idx1) && ~merged_instances(idx2)
                    unique_pairs = [unique_pairs; idx1, idx2];  % Store unique pair
                    merged_instances(idx1) = true;  % Mark instance 1 as merged
                    merged_instances(idx2) = true;  % Mark instance 2 as merged

                    % Stop if we've reached the required number of pairs to merge
                    if size(unique_pairs, 1) == excess_count
                        break;
                    end
                end
            end

            % 6. Merge the nearest pairs (vectorized)
            pair_indices_1 = unique_pairs(:, 1);
            pair_indices_2 = unique_pairs(:, 2);
            
            % Vectorized merging (we'll use array indexing and `mean` for merging)
            instance_1_data = class_instances_in_windows(pair_indices_1, 1:end-4);  % Feature data of instance 1
            instance_2_data = class_instances_in_windows(pair_indices_2, 1:end-4);  % Feature data of instance 2
            merged_instance = (instance_1_data + instance_2_data) / 2;  % Merge by averaging the features

            % Retain the class label, max flag, weight, and CA weight
            merged_class = class_instances_in_windows(pair_indices_1, end-3);  % Class label
            merged_flag = max(class_instances_in_windows(pair_indices_1, end-2), class_instances_in_windows(pair_indices_2, end-2));  % Max flag
            merged_weight = max(class_instances_in_windows(pair_indices_1, end-1), class_instances_in_windows(pair_indices_2, end-1));  % Max weight
            merged_ca_weight = max(class_instances_in_windows(pair_indices_1, end), class_instances_in_windows(pair_indices_2, end));  % Max CA weight

            % Create merged rows
            merged_rows = [merged_instance, merged_class, merged_flag, merged_weight, merged_ca_weight];

            % Track merged indices
            merged_indices = [pair_indices_1; pair_indices_2];  % Store the indices of the merged instances

            % 7. Remove the merged instances from class_instances_in_windows
            class_instances_in_windows(merged_indices, :) = [];  % Delete only the merged instances

            % 8. Append the merged instances to the windows
            class_instances_in_windows = [class_instances_in_windows; merged_rows];  % Append merged instances
        end

        % 9. Append the unmerged instances from new_data to class_instances_in_windows
        class_instances_in_windows = [class_instances_in_windows; class_instances_in_new_data];  % Add new data instances for this class

        % 10. Append the updated class_instances_in_windows to the final data (windows)
        if ~isempty(windows)
            windows(windows(:, end-3) == class_label, :) = [];  % Remove all class instances in windows (before appending new ones)
        end
        windows = [windows; class_instances_in_windows];  % Append class-specific instances after merging and adding new data
    end
end
