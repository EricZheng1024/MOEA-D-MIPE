classdef MOEADMIPE_V2 < ALGORITHM
% <multi/many> <real>
% MOEA/D with model-based ideal point estimation
% New features: Clip; Stable normalization
% delta --- 1 --- The probability of selecting candidates from neighbors
% type --- 2 --- The type of aggregation function
% Tm --- 0.1 ---  The size of neighborhood for mating
% Tr --- 0.1 --- The size of neighborhood for replacement
% nr --- 0.1 --- The maximum number of replacement
% modeling_fre --- 0.02 --- the frequency of modeling 
% spread_rbf --- 1000 --- the spread speed of RBF model

%------------------------------- Reference --------------------------------
% "Decomposition-Based Multi-Objective Evolutionary Algorithm with
% Model-Based Ideal Point Estimation"
% Remarks: Further study shows that the model can not estimate the ideal
% point well, and the effectiveness of the algorithm basically depends on
% the remedy scheme.
%--------------------------------------------------------------------------

    methods
        function main(Algorithm,Problem)
            %% Parameter setting
            [delta, type, Tm, Tr, nr, modeling_fre, spread_rbf] = Algorithm.ParameterSet(1, 2, 0.1, 0.1, 0.1, 0.02, 1000);
            modeling_end = 0.5;
            epsilon_ini = 0.5;
            epsilon_end = 0.001;

            n_interval = 20;

            % As "spread_rbf" increases (its default value in MATLAB is 1),
            % the fitted model becomes smoother, albeit at the expense of
            % reduced accuracy.


            %% Initialization
            % Generate the weight vectors
            [W, Problem.N] = UniformPoint(Problem.N, Problem.M);
            Tm = ceil(Problem.N * Tm);
            Tr = ceil(Problem.N * Tr);
            nr = ceil(Problem.N * nr);

            % Detect the neighbours of each solution
            B = pdist2(W, W);
            [~,B] = sort(B, 2);
            Bm = B(:, 1:Tm);
            Br = B(:, 1:Tr);
            
            % Generate random population
            Population = Problem.Initialization();
            p_train_raw = (Population.decs)';
            t_train_raw = (Population.objs)';
            z_min = min(Population.objs, [], 1);
            z_max = max(Population.objs, [], 1);
            z_e = z_min;


            %% Optimization
            while Algorithm.NotTerminated(Population)
                % For each solution
                z_max = 0.95*z_max + (1-0.95)*max(Population.objs, [], 1);
                for i = 1 : Problem.N
                    % Choose the parents
                    if rand < delta
                        P = Bm(i, randperm(Tm));
                        R = Br(i, randperm(Tr));
                    else
                        P = randperm(Problem.N);
                        R = randperm(Problem.N);
                    end

                    % Generate an offspring
                    Offspring = OperatorDE(Population(P(1)),Population(P(2)),Population(P(3)),{0.9,0.5,1,50});
                    z_min = min(z_min, Offspring.obj);
                    p_train_raw = [p_train_raw, (Offspring.dec)'];
                    t_train_raw = [t_train_raw, (Offspring.obj)'];

                    % Determine the reference point
                    if Problem.FE < modeling_end * Problem.maxFE && mod(Problem.FE/Problem.maxFE, modeling_fre) == 0
                        [p_train, t_train] = grid_sampling(p_train_raw, t_train_raw, n_interval);
                        if size(p_train,1) > n_interval^Problem.M  % Clip
                            p_train_raw = p_train;
                            t_train_raw = t_train;
                        end
                        z_pred = zeros(size(z_min));
                        for j = 1: Problem.M
                            z_pred(j) = predict_ide(p_train, t_train(j,:), spread_rbf, Problem);
                        end
                        z_lb = z_min - (z_max-z_min) .* (epsilon_ini-epsilon_end);
                        z_ub = z_min - (z_max-z_min) .* (epsilon_ini+(epsilon_end-epsilon_ini)*(Problem.FE-1)/(Problem.maxFE-1));
                        z_e = remedy(z_pred, z_lb, z_ub);
                    else
                        z_e = min(z_e, z_min);
                    end

                    % Update the neighbors
                    g_old = calSubpFitness(type, normalize_2(Population(R).objs,z_min,z_max), normalize_2(z_e,z_min,z_max), W(R, :));
                    g_new = calSubpFitness(type, normalize_2(Offspring.obj,z_min,z_max), normalize_2(z_e,z_min,z_max), W(R, :));
                    Population(R(find(g_old>=g_new, nr))) = Offspring;
                end
            end
        end
    end
end



%%
function res = predict_ide(p, t, rbf_spread, Problem)

    warning('off')
    net = newrbe(p, t, rbf_spread);
    warning('on')
    model = @(x) sim(net, x);
    [~, I] = min(t);
    options = optimoptions('fmincon','Display','off');
    [~, res] = fmincon(model, p(:,I), [] , [], [], [], Problem.lower, Problem.upper, [], options);
end

    
function z_e = remedy(z_pred, z_lb, z_ub)

    z_e = zeros(size(z_lb));
    for i = 1 : size(z_ub, 2)
        if z_pred(i) > z_ub(i)
            z_e(i) = z_ub(i);
        elseif z_pred(i) < z_lb(i)
            z_e(i) = z_lb(i);
        else
            z_e(i) = z_pred(i);
        end
    end
end


function [p, t] = grid_sampling(p, t, n_interval)

    [d,n] = size(t);
    
    grids = zeros(d,n_interval);
    xx = linspace(0,1,n_interval);
    for i = 1 : d
        % grids(i,:) = max(t(i,:))*xx.^2;
        lb = min(t(i,:));
        ub = max(t(i,:));
        grids(i,:) = lb + (ub-lb)*xx.^2;
    end
    
    index_selected = zeros(d,n);
    for j = 1 : n
        for i = 1 : d
            [~, index_selected(i,j)] = min(abs(grids(i,:) - t(i,j)));
            if t(i,j) - grids(i,index_selected(i,j)) < 0
                index_selected(i,j) = index_selected(i,j) - 1;
            end
        end
    end
    
    [~, ia] = unique(index_selected','row','stable');
    p = p';  t = t';
    p = (p(ia,:))';  t = (t(ia,:))';
end


%%
function objs_n = normalize_2(objs, lb, ub)
    objs_n = (objs - lb) ./ (ub - lb);
end


function g = calSubpFitness(type, objs, z, W)
% Calculate the function values of the scalarization method

    switch type
        case 1
            % weight sum approach
            g = sum(objs ./ W, 2);
        case 2
            % Tchebycheff approach
            g = max(abs(objs-z) ./ W, [], 2);
    end
end

