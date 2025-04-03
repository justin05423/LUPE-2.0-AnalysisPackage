function projected_data = project_to_pc_space(new_data, PCs, mean_original)
    % PROJECT_TO_PC_SPACE Projects new data into an existing PC space
    % Inputs:
    %   new_data      - (m x n) matrix, where m is the number of new samples and n is the feature dimension
    %   PCs          - (n x k) matrix, where each column is a principal component (top k PCs)
    %   mean_original - (1 x n) vector, mean of the original dataset used for PCA (optional)
    % Outputs:
    %   projected_data - (m x k) matrix, new data projected onto the top k PCs

    if nargin < 3
        warning('No mean provided; assuming new_data is already centered.');
        mean_original = zeros(1, size(new_data, 2));
    end

    % Center the new data
    centered_data = new_data - mean_original;

    % Project the centered data onto the PC space
    projected_data = centered_data * PCs;
end