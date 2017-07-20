function [finalX, info, xk, yk] = nesterov(problem, xCur, options)
    localdefaults.maxiter = 1000;
    localdefaults.tolgradnorm =  1e-6;
    localdefaults.alpha = 0.02; % < 1/L
    localdefaults.minstepsize = 1e-10;
    
    % Merge global and local defaults, then merge w/ user options, if any.
    localdefaults = mergeOptions(getGlobalDefaults(), localdefaults);
    if ~exist('options', 'var') || isempty(options)
        options = struct();
    end
    options = mergeOptions(localdefaults, options);
    
    xk = cell(1, options.maxiter);
    yk = cell(1, options.maxiter);
    M = problem.M;
    
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
        
        yNext = M.exp(xCur, xCurGradient, -options.alpha);
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
% 
% function [xCur, xCurCost, info, options] = bfgs_Smooth_release_version(problem, xCur, options)
%     localdefaults.maxiter = 1000;
%     localdefaults.tolgradnorm =  1e-6;
%     localdefaults.alpha = 1; % < 1/L
%     localdefaults.minstepsize = 1e-10;
%     
%     % Merge global and local defaults, then merge w/ user options, if any.
%     localdefaults = mergeOptions(getGlobalDefaults(), localdefaults);
%     if ~exist('options', 'var') || isempty(options)
%         options = struct();
%     end
%     options = mergeOptions(localdefaults, options);
%     
%     xk = cell(1, options.maxiter);
%     yk = cell(1, options.maxiter);
%     info = cell(1, options.maxiter);
%     M = problem.M;
%     
%     timetic = tic();
%     
%     iter = 0;
%     xk{1} = xCur;
%     yk{1} = xCur;
%     stepsize = 1;
%     storedb = StoreDB(options.storedepth);
%     key = storedb.getNewKey();
%     xCurCost = getCost(problem, xCur);
%     xCurGradient = getGradient(problem, xCur);
%     xCurGradNorm = M.norm(xCur, xCurGradient);
%     
%     stats = savestats();
%     info(1) = stats;
%     info(min(10000, options.maxiter+1)).iter = [];
%     
% 
% 
%     % Create a random starting point if no starting point
%     % is provided.
%     if ~exist('xCur','var')|| isempty(xCur)
%         xCur = M.rand(); 
%     end
%     
%     timetic = tic();
%     
%     % Create a store database and get a key for the current x
%     storedb = StoreDB(options.storedepth);
%     key = storedb.getNewKey();
%     
%     % __________Initialization of variables______________
%     % number of current element in memory
%     k = 0;  
%     %number of total iteration in BFGS
%     iter = 0; 
%     % saves vector that represents x_{k}'s projection on 
%     % x_{k+1}'s tangent space. And transport it to most
%     % current point's tangent space after every iteration.
%     sHistory = cell(1, options.memory);
%     % saves gradient of x_{k} by transporting it to  
%     % x_{k+1}'s tangent space. And transport it to most
%     % current point's tangent space after every iteration.
%     yHistory = cell(1, options.memory);
%     % saves inner(sk,yk)
%     rhoHistory = cell(1, options.memory);
%     % scaling of direction given by getDirection for acceptable step
%     alpha = 1; 
%     % scaling of initial matrix, BB.
%     scaleFactor = 1;
%     % norm of the step
%     stepsize = 1;
%     accepted = 1;
%     xCurGradient = getGradient(problem, xCur, storedb, key);
%     xCurGradNorm = M.norm(xCur, xCurGradient);
%     xCurCost = getCost(problem, xCur);
%     lsstats = [];
%     ultimatum = 0;
%     
%     % Save stats in a struct array info, and preallocate.
%     stats = savestats();
%     info(1) = stats;
%     info(min(10000, options.maxiter+1)).iter = [];
%     
%     if options.verbosity >= 2
%     fprintf(' iter\t               cost val\t                 grad. norm\t        alpha \n');
%     end
%     
%     while (1)
% %------------------------ROUTINE----------------------------
% 
%         % Display iteration information
%         if options.verbosity >= 2
%         %_______Print Information and stop information________
%         fprintf('%5d\t%+.16e\t%.8e\t %.4e\n', iter, xCurCost, xCurGradNorm, alpha);
%         end
%         
%         % Start timing this iteration
%         timetic = tic();
%         
%         % Run standard stopping criterion checks
%         [stop, reason] = stoppingcriterion(problem, xCur, options, ...
%             info, iter+1);
%         
%         % If none triggered, run specific stopping criterion check
%         if ~stop 
%             if stats.stepsize < options.minstepsize
%                     fprintf(['stepsize is too small, restart the bfgs procedure' ...
%                         'with the current point\n']);
%                 else
%                     stop = true;
%                     reason = sprintf(['Last stepsize smaller than minimum '  ...
%                         'allowed; options.minstepsize = %g.'], ...
%                         options.minstepsize);
%                 end
%             else
%                 ultimatum = 0;
%             end
%         end  
%         
%         if stop
%             if options.verbosity >= 1
%                 fprintf([reason '\n']);
%             end
%             break;
%         end
% 
%         
%         
%         
%         iter = iter + 1;
%         xCur = xNext;
%         xCurGradient = getGradient(problem, xCur);
%         xCurGradNorm = M.norm(xCur, xNextGradient);
%         xCurCost = xNextCost;
%         
%         % Make sure we don't use too much memory for the store database
%         storedb.purge();
%         
%         key = newkey;
%         
%         % Log statistics for freshly executed iteration
%         stats = savestats();
%         info(iter+1) = stats; 
%         
%     end
% 
%     
%     info = info(1:iter+1);
% 
%     if options.verbosity >= 1
%         fprintf('Total time is %f [s] (excludes statsfun)\n', ...
%                 info(end).time);
%     end
% 
%     % Routine in charge of collecting the current iteration stats
%     function stats = savestats()
%         stats.iter = iter;
%         stats.cost = xCurCost;
%         stats.gradnorm = xCurGradNorm;
%         if iter == 0
%             stats.stepsize = NaN;
%             stats.time = toc(timetic);
%         else
%             stats.stepsize = stepsize;
%             stats.time = info(iter).time + toc(timetic);
%         end
%         stats.linesearch = lsstats;
%         stats = applyStatsfun(problem, xCur, storedb, key, options, stats);
%     end
% 
% end
