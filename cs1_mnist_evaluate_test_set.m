%% This code evaluates the test set.

% ** Important.  This script requires that:
% 1)'centroid_labels' be established in the workspace
% AND
% 2)'centroids' be established in the workspace
% AND
% 3)'test' be established in the workspace


% IMPORTANT!!:
% You should save 1) and 2) in a file named 'classifierdata.mat' as part of
% your submission.

predictions = zeros(200,1);
outliers = zeros(200,1);

% loop through the test set, figure out the predicted number
for i = 1:200

    testing_vector=test(i,1:784);

    % Extract the centroid that is closest to the test image
    [prediction_index, vec_distance]=assign_vector_to_centroid(testing_vector,centroids);

    predictions(i) = centroid_labels(prediction_index);
    outliers (i) = vec_distance;

end

%% DESIGN AND IMPLEMENT A STRATEGY TO SET THE outliers VECTOR
% outliers(i) should be set to 1 if the i^th entry is an outlier
% otherwise, outliers(i) should be 0
sorted_distances = sort(outliers, 'descend');
threshold = sorted_distances(11);
outliers = outliers >= threshold;

%% MAKE A STEM PLOT OF THE OUTLIER FLAG
figure;
stem(outliers);
xlabel("Test Image Number");
ylabel("Outlier Flag");
title("Detected Outliers");


%% The following plots the correct and incorrect predictions
% Make sure you understand how this plot is constructed
figure;
plot(correctlabels,'o');
hold on;
plot(predictions,'x');
title('Correct vs. Predicted Digits');
xlabel('Test Image Number');
ylabel('Digit');
legend('Correct Label', 'Prediction');

%% The following line provides the number of instances where and entry in correctlabel is
% equal to the corresponding entry in prediction
% However, remember that some of these are outliers
accuracy = 100 * sum(correctlabels==predictions) / length(correctlabels)

wrong = find(correctlabels ~= predictions);

most_mistaken_digit = mode(correctlabels(wrong));
predicted_for_that_digit = predictions( ...
    wrong(correctlabels(wrong) == most_mistaken_digit));

most_common_confusion = mode(predicted_for_that_digit);
fprintf('Most mistaken digit: %d\n', most_mistaken_digit);
fprintf('Most common predicted digit for it: %d\n', most_common_confusion);

incorrect=correctlabels~=predictions;
wrong_outliers = sum(incorrect & outliers);
wrong_notoutliers = sum(incorrect & ~outliers);
fprintf('Incorrect outlier predictions: %d\n', wrong_outliers);
fprintf('Incorrect non-outlier predictions: %d\n', wrong_notoutliers);

function [index, vec_distance] = assign_vector_to_centroid(data,centroids)
k = size(centroids, 1);
distances = zeros(k,1);

for j =1:k
    distances(j) = norm(data - centroids(j,1:784));
end

[vec_distance,index] = min(distances);

end
