function [accuracy, gMean, f1Score, kappa] = calculatePerformanceMetrics(predicted_labels, true_labels)

    % Validate input sizes
    if length(predicted_labels) ~= length(true_labels)
        error('Predicted and true labels must be of the same length.');
    end

    % ---------------- Accuracy ----------------
    accuracy = sum(predicted_labels == true_labels) / length(true_labels) * 100;

    % ---------------- Classes in CURRENT chunk ----------------
    uniqueClasses = unique(true_labels);
    numClasses = length(uniqueClasses);

    % ---------------- Init ----------------
    sensitivity = zeros(numClasses, 1);
    f1_per_class = zeros(numClasses, 1);

    % ---------------- Per-class metrics ----------------
    for i = 1:numClasses
        class = uniqueClasses(i);

        tp = sum(predicted_labels == class & true_labels == class);
        fn = sum(predicted_labels ~= class & true_labels == class);
        fp = sum(predicted_labels == class & true_labels ~= class);

        % Recall (Sensitivity)
        if (tp + fn) > 0
            sensitivity(i) = tp / (tp + fn);
        else
            sensitivity(i) = 0;
        end

        % Precision
        if (tp + fp) > 0
            precision = tp / (tp + fp);
        else
            precision = 0;
        end

        % F1 (per class)
        if (precision + sensitivity(i)) > 0
            f1_per_class(i) = 2 * precision * sensitivity(i) / ...
                              (precision + sensitivity(i));
        else
            f1_per_class(i) = 0;
        end
    end

    % ---------------- G-Mean (?-smoothed, STREAM-SAFE) ----------------
    epsilon = 0.05;  % smoothing for unseen / missed classes
    recall_smoothed = max(sensitivity, epsilon);
    gMean = prod(recall_smoothed)^(1 / numClasses) * 100;

    % ---------------- Macro F1 ----------------
    f1Score = mean(f1_per_class) * 100;

    % ---------------- Cohen’s Kappa (chunk only) ----------------
    [cMatrix, ~] = confusionmat(true_labels, predicted_labels);
    totalSum = sum(cMatrix(:));

    if totalSum > 0
        sumExpected = sum(sum(cMatrix, 1) .* sum(cMatrix, 2)) / totalSum;
        sumActual = sum(diag(cMatrix));
        if (totalSum - sumExpected) > 0
            kappa = (sumActual - sumExpected) / ...
                    (totalSum - sumExpected) * 100;
        else
            kappa = 0;
        end
    else
        kappa = 0;
    end
end
