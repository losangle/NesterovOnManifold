function clientNest
    n = 10;
    manifold = spherefactory(n);
    A = randn(n);
    A = .5*(A+A.');
    
    problem.M = manifold;
    problem.cost  = @(x) -x'*(A*x);
    problem.egrad = @(x) -2*A*x;
    
    xCur = problem.M.rand();
    options = [];
    [finalX, stats, xk, yk] = nesterov(problem, xCur, options);
    displaystats(stats);
    
    
    function displaystats(stats)
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
    