function [finalX, info, xk, yk] = nesterovtest(problem, xCur, options)
    localdefaults.maxiter = 2000;
    localdefaults.tolgradnorm =  1e-6;
    localdefaults.alpha = 0.02; % < 1/L
    localdefaults.minstepsize = 1e-10;
    localdefaults.linesearch = @linesearch;
    
    % Merge global and local defaults, then merge w/ user options, if any.
    localdefaults = mergeOptions(getGlobalDefaults(), localdefaults);
    if ~exist('options', 'var') || isempty(options)
        options = struct();
    end
    options = mergeOptions(localdefaults, options);
    
    M = problem.M;
    
    if ~exist('xCur','var')|| isempty(xCur)
        xCur = M.rand(); 
    end
    
    
    xk = cell(1, options.maxiter);
    yk = cell(1, options.maxiter);
    
    
    timetic = tic();
    
    iter = 0;
    xk{1} = xCur;
    yk{1} = xCur;
    stepsize = 1;
    storedb = StoreDB(options.storedepth);
    key = storedb.getNewKey();
    xCurCost = getCost(problem, xCur);
    xCurGradient = getGradient(problem, xCur);
    xCurGradNorm = M.norm(xCur, xCurGradient);
    
    stats = savestats();
    info(1) = stats;
    info(min(10000, options.maxiter+1)).iter = [];
    
    
    while(1)
        curIter = iter + 1;
        % Run standard stopping criterion checks
        [stop, reason] = stoppingcriterion(problem, xCur, options, ...
            info, iter+1);
        
        % If none triggered, run specific stopping criterion check
        if ~stop && stats.stepsize < options.minstepsize
            stop = true;
            reason = sprintf(['Last stepsize smaller than minimum '  ...
                'allowed; options.minstepsize = %g.'], ...
                options.minstepsize);
        end
        
        if stop
            if options.verbosity >= 1
                fprintf([reason '\n']);
            end
            break;
        end
        
                % Pick the descent direction as minus the gradient
        desc_dir = problem.M.lincomb(xCur, -1, xCurGradient);
        
        % Execute the line search
        [stepsize, newx, newkey, lsstats] = options.linesearch( ...
                             problem, xCur, desc_dir, xCurCost, -xCurGradNorm^2, ...
                             options, storedb, key);
        
        yNext = M.exp(xCur, desc_dir, min(options.alpha, stepsize/xCurGradNorm));
        xNext = M.exp(yNext, M.log(yNext, yk{curIter}), -(curIter-1)/(curIter+2));
        yk{curIter + 1} = yNext;
        xk{curIter + 1} = xNext;
        
        key = storedb.getNewKey();
        stepsize = M.dist(xNext, xCur);
        xCur = xNext;
        xCurGradient = getGradient(problem, xCur, storedb, key);
        xCurCost = getCost(problem, xCur);
        xCurGradNorm = M.norm(xCur, xCurGradient);
        
        iter = iter +1;
        stats = savestats();
        info(iter+1) = stats; 
    end
    
    info = info(1:iter+1);
    xk = xk(1, 1:iter+1);
    yk = yk(1, 1:iter+1);
    
    finalX = xCur;
    

    function stats = savestats()
        stats.iter = iter;
        stats.cost = xCurCost;
        stats.gradnorm = xCurGradNorm;
        if iter == 0
            stats.stepsize = NaN;
            stats.time = toc(timetic);
        else
            stats.stepsize = stepsize;
            stats.time = info(iter).time + toc(timetic);
        end
        stats.linesearch = [];
        stats = applyStatsfun(problem, xCur, storedb, key, options, stats);
    end
end

