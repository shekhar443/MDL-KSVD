function X = cosamp(D, Y, sparsity)
    % CoSaMP sparse coding
    [n, m] = size(Y);
    k = sparsity;
    X = zeros(size(D, 2), m);
    for i = 1:m
        y = Y(:, i);
        residual = y;
        support = [];
        for iter = 1:k
            % Find 2k largest correlations
            corr = abs(D' * residual);
            [~, idx] = sort(corr, 'descend');
            support = union(support, idx(1:2*k));
            
            % Least squares on support
            D_s = D(:, support);
            x_s = pinv(D_s) * y;
            
            % Prune to k largest coefficients
            [~, idx] = sort(abs(x_s), 'descend');
            support = support(idx(1:k)); % Update support to k indices
            
            % Solve least squares again on pruned support
            D_s = D(:, support);
            x_s = pinv(D_s) * y; % x_s now matches size of pruned support
            
            % Update residual
            residual = y - D_s * x_s; % Corrected multiplication
            
            % Store coefficients
            X_temp = zeros(size(D, 2), 1);
            X_temp(support) = x_s;
            X(:, i) = X_temp;
        end
    end
end