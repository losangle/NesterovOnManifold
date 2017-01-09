function main
    % Generate random problem data.
    n = 500;
%     A = randn(n);
%     A = .5*(A+A.');
%    [U,S,V] = svd(A);
%    test(A,n);
%      test(S,n);
%      disp(S)
    %m = 100;
    %A = diag([1./(1:m), zeros(1, n-m)]);
    A = diag(1./(1:n));
    %disp(A(1:10,1:10));
    %V = randn(n);
%     A = V * A * V';
    %A = hilb(n);
    disp(cond(A))
    test(A,n)
    
    function test(A,n)
        % Create the problem structure.
        manifold = spherefactory(n);
        problem.M = manifold;

        % Define the problem cost function and its Euclidean gradient.
        problem.cost  = @(x) -x'*(A*x);
        problem.egrad = @(x) -2*A*x;
        egrad = @(x) - 2*A*x;

        % Numerically check gradient consistency (optional).

        %[result, xcost, info, options] = trustregions(problem);
        %disp(xcost);
        %checkgradient(problem);

        % Solve.
        % Initialization
        beta = 3 * norm(eig(A),Inf);
        x(1,:) = manifold.rand();
        y(1,:) = x(1,:);
        lambda(1) = 0;
        lambda(2) = (1+sqrt(1+4*lambda(1)^2))/2;
        gamma(1) = 0;
        iter(1) = 1;
        T = 30; %Number of iteration

        % Nesterov Acceleration
        grad_ac(1) = norm(egrad(x(1,:)'),2);
        for k = 2:T
            iter(k) = k;
            lambda(k+1) = (1+sqrt(1+3.5*lambda(k)^2))/2; %Notice the difference in the coefficient 4-> 3.5
            gamma(k) = (1-lambda(k))/lambda(k+1);
            x_s = x(k-1,:)';
            df = egrad(x_s);
            gradient = - (1/beta) * getGradient(problem, x_s);
            y(k,:) = manifold.exp(x_s, gradient);
            x(k,:) = manifold.exp(y(k,:), gamma(k) * manifold.log(y(k,:),y(k-1,:)));
            grad_ac(k) = norm(getGradient(problem, x_s));
        end
        disp(norm(x(k,:)*A*x(k,:)'))

        %Original Gradient Descent
        grad_gc(1) = norm(egrad(x(1,:)'),2);
        for k = 2:T
            iter(k) = k;
            %lambda(k+1) = (1+sqrt(1+4*lambda(k)^2))/2;
            %gamma(k) = (1-lambda(k))/lambda(k+1);
            x_s = x(k-1,:)';
            df = egrad(x_s);
            %gradient = - (1/beta) * (df-(x_s'*df)*x_s);
            gradient = - (1/beta) * getGradient(problem, x_s); 
            x(k,:) = manifold.exp(x_s, gradient);
            %x(k,:) = manifold.exp(y(k,:), gamma(k) * manifold.log(y(k,:),y(k-1,:)));
            grad_gc(k) = norm(getGradient(problem, x_s));
        end
        disp(norm(x(k,:)*A*x(k,:)'))


        % Display some statistics.
        figure;
        semilogy(iter,grad_gc, '.-');
        title('Gradient Descent on Manifold')
        figure 
        semilogy(iter,grad_ac, '.-');
        title('Attempt Neesterov AGD on Manifold')
        xlabel('Iteration number');
        ylabel('Norm of the gradient of f');
    end
end