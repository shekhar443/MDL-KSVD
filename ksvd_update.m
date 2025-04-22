function D = ksvd_update(D, Y, X)
    % k-SVD dictionary update
    [n, k] = size(D);
    for j = 1:k
        % Find signals using atom j
        idx = find(X(j, :) ~= 0);
        if isempty(idx)
            continue;
        end
        
        % Compute error without atom j
        E = Y(:, idx) - D * X(:, idx) + D(:, j) * X(j, idx);
        
        % SVD to update atom j
        [U, S, V] = svd(E, 'econ');
        D(:, j) = U(:, 1);
        X(j, idx) = S(1, 1) * V(:, 1)';
    end
    % Normalize dictionary
    D = D ./ sqrt(sum(D.^2));
end