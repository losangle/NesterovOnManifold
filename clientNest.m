function clientNest

    %Nonsmooth Rosenbeog Doesn't work at all
%     dim = 2;
%     w = 2;
%     manifold = euclideanfactory(dim);
%     cost = @(x) (1-x(2))^2+w*abs(x(2)-x(1)^2);
%     grad = @(x) [sign(x(2)-x(1)^2)*w*(-2*x(1)); sign(x(2)-x(1)^2)*w-2*(1-x(2))];

    %Raleigh Quotient
    n = 10;
    manifold = spherefactory(n);
    A = randn(n);
    A = .5*(A+A.');
    cost = @(x) -x'*(A*x);
    grad = @(x) -2*A*x;
    
    
    problem.M = manifold;
    problem.cost  = cost;
    problem.egrad = grad;
    
    
    xCur = problem.M.rand();
    options = [];
    [finalX, stats, xk, yk] = nesterov(problem, xCur, options);
    displaystats(stats);
    disp(finalX)
    
    
    function displaystats(stats)
        finalcost = stats(end).cost;
        for numcost = 1 : length([stats.cost])
            stats(numcost).cost = stats(numcost).cost - finalcost;
        end
        
        
        figure;
        
        
        subplot(2,2,1)
        semilogy([stats.gradnorm], '.-');
        xlabel('Iter');
        ylabel('GradNorms');
        
        titletest = sprintf('Time: %f', stats.time);
        title(titletest);
        
        subplot(2,2,3)
        semilogy([stats.stepsize], '.-');
        xlabel('Iter');
        ylabel('stepsizes');
        
        subplot(2,2,4)
        semilogy([stats.cost], '.-');
        xlabel('Iter');
        ylabel('costs');
    end

end
    