function absSmooth
    n = 500;
    T = 1200000;
    beta = 2000;
    datas = 2000;
    %generate(n);
    %[A,b] = read(n);
    %test(A,b,n,T,beta);
    %continueTest(A,b,n,T,beta);
    display(datas);
    
    
    
    function [RandMat,RandVec] = read(n)
        RandMat = dlmread('matrix.txt');
        RandVec = dlmread('b.txt');
    end

    function generate(n)
        % Generate random problem data.
        RandMat = randn(n);
        RandMat = .5*(RandMat+RandMat.');
        RandVec = rand(n,1);
        fid=fopen('matrix.txt','wt');
        for rows=1:size(RandMat,1)
           fprintf(fid, '%f ', RandMat(rows,:));
           fprintf(fid, '\n');
        end
        fclose(fid);
        fid=fopen('b.txt','wt');
        for rows=1:size(RandVec,1)
           fprintf(fid, '%f ', RandVec(rows,:));
           fprintf(fid, '\n');
        end
        fclose(fid);
    end
    
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
    
    function continueTest(A,b,n,T,beta)
        % Create the problem structure.
        manifold = spherefactory(n);
        problem.M = manifold;

        % Define the problem cost function and its Euclidean gradient.
        problem.cost  = @(x) cost(A,x,b);
        problem.egrad = @(x) gradHobbes(A,x,b);

        % Solve.
        % Initialization
        CurrentSit = dlmread('current.txt');
        x = CurrentSit(1,:)';
        yOld = CurrentSit(2,:)';
        yNew = CurrentSit(3,:)';
        CurrentSit = dlmread('currentLambda.txt');
        lambdaOld = CurrentSit(1,1);
        lambdaNew = CurrentSit(2,1);
        iter = CurrentSit(3,1);
        fiter = fopen('iter.txt','a');
        fnorm = fopen('normac.txt','a');
        
        % Nesterov Acceleration
        fprintf(fnorm, '%f \n', norm(gradHobbes(A,x,b)));
        for k = 1:T
            iter = iter + 1;
            fprintf(fiter, '%d \n', iter);
            
            lambdaOld = lambdaNew;
            lambdaNew = (1+sqrt(1+4*lambdaOld^2))/2; 
            gamma = (1-lambdaOld)/lambdaNew;
            
            yOld = yNew;
            gradient = - (1/beta) * gradHobbes(A,x,b);
            yNew = manifold.exp(x, gradient);
            x = manifold.exp(yNew, gamma * manifold.log(yNew,yOld));
            
            fprintf(fnorm, '%f \n', norm(gradHobbes(A,x,b)));
            if mod(k,500) == 0
                fprintf('At point %d; ', k);
            end
        end
        fclose(fiter);
        fclose(fnorm);
        fcurrent = fopen('current.txt','w');
        fprintf(fcurrent, '%f ', x);
        fprintf(fcurrent, '\n');
        fprintf(fcurrent, '%f ', yOld);
        fprintf(fcurrent, '\n');
        fprintf(fcurrent, '%f ', yNew);
        fprintf(fcurrent, '\n');
        fclose(fcurrent);
        fcurrentLambda = fopen('currentLambda.txt','w');
        fprintf(fcurrentLambda, '%f \n', lambdaOld);
        fprintf(fcurrentLambda, '%f \n', lambdaNew);
        fprintf(fcurrentLambda, '%f \n', iter);
        fclose(fcurrentLambda);
        
        disp(cost(A, x, b))
        
        %Original Gradient Descent 
        current_gc = dlmread('current_gc_x.txt');
        x = current_gc(:,1);
        fnorm = fopen('normgc.txt','a');
        for k = 1:T
            gradient = - (1/beta) * gradHobbes(A,x,b); 
            x = manifold.exp(x, gradient);
            fprintf(fnorm, '%f \n', norm(gradHobbes(A,x,b)));
        end
        fclose(fnorm);
        fcurrent_gc = fopen('current_gc_x.txt','w');
        fprintf(fcurrent_gc, '%f \n', x);
        fclose(fcurrent_gc);
        disp(cost(A, x, b))
    end


    function test(A,b,n,T,beta)
        % Create the problem structure.
        manifold = spherefactory(n);
        problem.M = manifold;

        % Define the problem cost function and its Euclidean gradient.
        problem.cost  = @(x) cost(A,x,b);
        problem.egrad = @(x) gradHobbes(A,x,b);

        % Solve.
        % Initialization
        x = manifold.rand();
        starting = x;
        yOld = x;
        yNew = yOld;
        lambdaOld = 0;
        lambdaNew = (1+sqrt(1+4*lambdaOld^2))/2;
        gamma = 0;
        iter = 1;
        fiter = fopen('iter.txt','w');
        fprintf(fiter, '%d \n', iter);
        fnorm = fopen('normac.txt','w');
        
        % Nesterov Acceleration
        fprintf(fnorm, '%f \n', norm(gradHobbes(A,x,b)));
        for k = 2:T
            iter = iter + 1;
            fprintf(fiter, '%d \n', iter);
            
            lambdaOld = lambdaNew;
            lambdaNew = (1+sqrt(1+4*lambdaOld^2))/2; 
            gamma = (1-lambdaOld)/lambdaNew;
            
            yOld = yNew;
            gradient = - (1/beta) * gradHobbes(A,x,b);
            yNew = manifold.exp(x, gradient);
            x = manifold.exp(yNew, gamma * manifold.log(yNew,yOld));
            
            fprintf(fnorm, '%f \n', norm(gradHobbes(A,x,b)));
            if mod(k,500) == 0
                fprintf('At point %d; ', k);
            end
        end
        fclose(fiter);
        fclose(fnorm);
        fcurrent = fopen('current.txt','w');
        fprintf(fcurrent, '%f ', x);
        fprintf(fcurrent, '\n');
        fprintf(fcurrent, '%f ', yOld);
        fprintf(fcurrent, '\n');
        fprintf(fcurrent, '%f ', yNew);
        fprintf(fcurrent, '\n');
        fclose(fcurrent);
        fcurrentLambda = fopen('currentLambda.txt','w');
        fprintf(fcurrentLambda, '%f \n', lambdaOld);
        fprintf(fcurrentLambda, '%f \n', lambdaNew);
        fprintf(fcurrentLambda, '%f \n', iter);
        fclose(fcurrentLambda);
        
        disp(cost(A, x, b))
        
        %Original Gradient Descent 
        x = starting;
        fnorm = fopen('normgc.txt','w');
        fprintf(fnorm, '%f \n', norm(gradHobbes(A,x,b)));
        
        for k = 2:T
            gradient = - (1/beta) * gradHobbes(A,x,b); 
            x = manifold.exp(x, gradient);
            fprintf(fnorm, '%f \n', norm(gradHobbes(A,x,b)));
        end
        fclose(fnorm);
        fcurrent_gc = fopen('current_gc_x.txt','w');
        fprintf(fcurrent_gc, '%f \n', x);
        fclose(fcurrent_gc);
        disp(cost(A, x,b))
    end

    % Display some statistics
    function display(k)
        grad_gc = dlmread('normgc.txt');
        grad_ac = dlmread('normac.txt');
        iter = dlmread('iter.txt');
        figure;
        semilogy(iter(1:k),grad_gc(1:k), '.-');
        %loglog(iter(1:k),grad_gc(1:k), '.-');
        title('Gradient Descent on Manifold')
        figure 
        semilogy(iter(1:k),grad_ac(1:k), '.-');
        %loglog(iter(1:k),grad_ac(1:k), '.-');
        title('Attempt Neesterov AGD on Manifold')
        xlabel('Iteration number');
        ylabel('Norm of the gradient of f');
    end
end