function clientNest

    %Nonsmooth Rosenbeog Doesn't work at all
%     dim = 2;
%     w = 2;
%     manifold = euclideanfactory(dim);
%     cost = @(x) (1-x(2))^2+w*abs(x(2)-x(1)^2);
%     grad = @(x) [sign(x(2)-x(1)^2)*w*(-2*x(1)); sign(x(2)-x(1)^2)*w-2*(1-x(2))];

%     Raleigh Quotient
    n = 1000;
    manifold = spherefactory(n);
    A = randn(n);
    A = .5*(A+A.');
    cost = @(x) -x'*(A*x);
    grad = @(x) -2*A*x;

%     Rosenbrog
%     dim = 2;
%     cost = @(x) (1-x(1))^2+5*(x(2)-x(1)^2)^2;
%     grad = @(x) [-2*(1-x(1))+10*(x(2)-x(1)^2)*(-2*x(1));10*(x(2)-x(1)^2)];
%     manifold =  euclideanfactory(dim);
    
    
    problem.M = manifold;
    problem.cost  = cost;
    problem.egrad = grad;
    
    
    xCur = problem.M.rand();
    options = [];
    [finalX, stats, xk, yk] = nesterov(problem, xCur, options);
    displaystats(stats);
    disp(finalX)
    
%     options.linesearch = @(problem, x, desc_dir, cost, gradnormsquare, ...
%                              options, storedb, key) lsdumb(problem, x, desc_dir, cost, gradnormsquare, ...
%                              options, storedb, key);
    
    [finalX, cost, info, options] = steepestdescent(problem, xCur, options)
    displaystats(info);    
    
    function  [stepsize, newx, newkey, lsstats] = lsdumb( ...
                             problem, x, desc_dir, cost, gradnormsquare, ...
                             options, storedb, key)
        stepsize = 1;
        newx = problem.M.retr(x, desc_dir, 1);
        newkey = storedb.getNewKey();
        lsstats = [];
    end
    
    function displaystats(stats)
        finalcost = stats(end).cost;
        for numcost = 1 : length([stats.cost])
            stats(numcost).cost = stats(numcost).cost - finalcost;
        end
        
        figure;
        subplot(2,2,1)
        loglog([stats.gradnorm], '.-');
        xlabel('Iter');
        ylabel('GradNorms');
        
        titletest = sprintf('Time: %f', stats(end).time);
        title(titletest);
        
        subplot(2,2,3)
        loglog([stats.stepsize], '.-');
        xlabel('Iter');
        ylabel('stepsizes');
        
        subplot(2,2,4)
        loglog([stats.cost], '.-');
        xlabel('Iter');
        ylabel('costs');
    end

end
    