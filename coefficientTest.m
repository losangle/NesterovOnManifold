function coefficientTest
    % Generate random problem data.
    n = 500;
    A = randn(n);
    %A = .5*(A+A.');
    b = rand(n,1);
    disp(cond(A))
    test(A,b,n)

    function c = cost(A,x,b)
        c = sum(costvec(A,x,b));
    end

    function C = costvec(A,x,b)
        C = A*x - b;
        C = C.*C;
        C = C + 0.1;
        C = sqrt(C);
    end

    function G = gradHobbes(A,x,b)
        N = length(b);
        g = costvec(A,x,b);
        g = (1./g);
        C = A*x - b;
        for i = 1:N
            D = C.*A(:,i);
            D = D.*g;
            G(i,1) = sum(D);
        end
    end
    
    function test(A,b,n)
        % Create the problem structure.
        manifold = spherefactory(n);
        problem.M = manifold;

        % Define the problem cost function and its Euclidean gradient.
        problem.cost  = @(x) cost(A,x,b);
        problem.egrad = @(x) gradHobbes(A,x,b);

        % Numerically check gradient consistency (optional).
        % Solve.
        % Initialization
        beta = 600 * norm(eig(A),Inf);
        T = 1000; %Number of iteration
        x(1,:) = manifold.rand();
        y(1,:) = x(1,:);
        lambda = zeros(T,1);
        iter = zeros(T,1);
        lambda(1,1) = 0;
        lambda(2,1) = (1+sqrt(1+4*lambda(1)^2))/2;
        gamma = 0;
        iter(1) = 1;
        figure;

        for l = 1:25
            coe = 0.2*l;
            % Nesterov Acceleration
            grad_ac(1) = norm(gradHobbes(A,x(1,:)',b),2);
            for k = 2:T
                iter(k,1) = k;
                lambda(k+1,1) = (1+sqrt(1+coe*lambda(k,1)^2))/2; %Notice change in coefficient
                gamma = (1-lambda(k,1))/lambda(k+1,1);
                x_s = x(k-1,:)';
                gradient = - (1/beta) * gradHobbes(A,x_s,b);
                y(k,:) = manifold.exp(x_s, gradient);
                x(k,:) = manifold.exp(y(k,:), gamma * manifold.log(y(k,:),y(k-1,:)));
                grad_ac(k) = norm(gradHobbes(A,x_s,b));
                if mod(k,500) == 0
                    fprintf('At point %d', k);
                end
            end
        subplot(5,5,l)
        str = sprintf('AGD %f',coe);
        semilogy(iter,grad_ac, '.-');
        title(str);
        end
        
        %Original Gradient Descent 
        grad_gc(1) = norm(gradHobbes(A,x(1,:)',b));
        for k = 2:T
            x_s = x(k-1,:)';
            gradient = - (1/beta) * gradHobbes(A,x_s,b); 
            x(k,:) = manifold.exp(x_s, gradient);
            grad_gc(k) = norm(getGradient(problem, x_s));
        end
        disp(cost(A, x(k,:)',b))

        % Display some statistics.

        figure 
        semilogy(iter,grad_gc, '.-');
        ('Attempt Neesterov GD on Manifold')
        xlabel('Iteration number');
        ylabel('Norm of the gradient of f');
    end
end