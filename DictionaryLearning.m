function D = DictionaryLearning(data, num_atoms, sparsity, iterations, init_method)
    % Improved Dictionary Learning using K-SVD with enhanced initialization and normalization
    % Input:
    % data - Input patches (each column is a patch)
    % num_atoms - Number of dictionary atoms
    % sparsity - Sparsity level for CoSaMP
    % iterations - Number of K-SVD iterations
    % init_method - Dictionary initialization method ('random' or 'data')
    % Output:
    % D - Learned dictionary
    
    % Default parameters
    if nargin < 5
        init_method = 'data';
    end
    if nargin < 4
        iterations = 30;
    end
    if nargin < 3
        sparsity = 10;
    end
    if nargin < 2
        num_atoms = 256;
    end
    
    % Input validation
    [patch_size, num_patches] = size(data);
    if num_atoms > num_patches
        error('Number of atoms cannot exceed number of patches');
    end
    
    % Normalize input data
    data_mean = mean(data);
    data_std = std(data) + eps;
    data_normalized = (data - data_mean) ./ data_std;
    
    % Initialize dictionary
    switch lower(init_method)
        case 'random'
            D = randn(patch_size, num_atoms);
        case 'data'
            % Initialize with random selection of data patches
            idx = randperm(num_patches, num_atoms);
            D = data_normalized(:, idx);
        otherwise
            error('Invalid initialization method. Use ''random'' or ''data''');
    end
    
    % Normalize dictionary atoms
    D = D ./ sqrt(sum(D.^2) + eps);
    
    % K-SVD parameters
    params.data = data_normalized;
    params.Tdata = sparsity;
    params.dictsize = num_atoms;
    params.iternum = iterations;
    params.memusage = 'high';
    params.initdict = D;
    params.errorFlag = 1;
    params.preserveDCAtom = 0;
    
    % Run K-SVD
    try
        [D, X] = ksvd(params);
        D = D ./ sqrt(sum(D.^2) + eps);
        reconstructed = D * X;
        mse = mean(mean((data_normalized - reconstructed).^2));
        fprintf('Dictionary Learning completed. Final MSE: %.4f\n', mse);
    catch e
        fprintf('Error in K-SVD: %s\n', e.message);
        D = [];
    end
end