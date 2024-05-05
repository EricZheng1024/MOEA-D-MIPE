classdef BiasedMOP_GECCO2023 < PROBLEM
% <multi/many> <real>
% Biased MOP
% Continuous PF: dc_alpha=1, dc_A=dc_beta=0
% Unimodality: c_t_poly=1, fre=0
% Mapping between codes parameters and paper parameters:
%   A-E,c_t_poly,fre -> vector a
%   F -> q
%   center -> p
%   dc_A,dc_beta -> vector c
%   dc_alpha is always 1
% A --- 4 --- 
% B --- 0 --- 
% C --- 2 --- 
% D --- 0 --- 
% E --- 0 --- 
% F --- 1 --- Scalar or vector; i-th entry is for i-th objective; e.g. [0.5;2;0.8] for 3-objective case.
% center ---  --- The easiest center to converge. Note: on the unit simplex; column vector. e.g. [0.8;0.2], and we can find 0.8+0.2=1.
% dc_A --- 5 --- e.g. larger, more disconnected regions
% dc_alpha --- 1 --- e.g. larger than 1, convex; note that the shape of PF also can be control by F
% dc_beta --- 1 --- e.g. larger, disconnected regions is more dense in some parts and more sparse in others
% c_t_poly --- 4 --- e.g. larger, more local optima that are far from global optimum will disappear
% fre --- 8*pi --- e.g. larger, more local optima in the distance function

%------------------------------- Reference --------------------------------
% "Decomposition-Based Multi-Objective Evolutionary Algorithm with
% Model-Based Ideal Point Estimation" (GECCO2023)
% Related papers:
% 1. "A Generator for Multiobjective Test Problems with
% Difficult-to-Approximate Pareto Front Boundaries"
% 2. "A Review of Multiobjective Test Problems and a Scalable Test Problem
% Toolkit"
%--------------------------------------------------------------------------
% 
% Author: Ruihao Zheng
% Last modified: 11/01/2023

    properties(Access = private)
        A_;
        B_;
        C_;
        D_;
        E_;
        F_;
        center_;
        dc_A_;
        dc_alpha_;
        dc_beta_;
        c_t_poly_;
        fre_;
    end
    
    methods
        %% Default settings of the problem
        function Setting(obj)
            if isempty(obj.M); obj.M = 2; end
            if isempty(obj.D); obj.D = 29+obj.M; end
            [obj.A_,obj.B_,obj.C_,obj.D_,obj.E_,obj.F_,obj.center_,obj.dc_A_,obj.dc_alpha_,obj.dc_beta_,obj.c_t_poly_,obj.fre_] = ...
                obj.ParameterSet(4, 0, 2, 0, 0, 1, 1/obj.M*ones(obj.M,1), 5, 1, 1, 4, 8*pi);
            tmp = [repmat([0,1],obj.M-1,1); -1*ones(obj.D-obj.M+1,1),ones(obj.D-obj.M+1,1)];
            obj.lower    = tmp(:,1)';
            obj.upper    = tmp(:,2)';
            obj.encoding = 'real';
        end
        %% Calculate objective values
        function PopObj = CalObj(obj,PopDec)
            PopObj = zeros(size(PopDec, 1), obj.M);
            for i = 1 : size(PopDec, 1)
                PopObj(i,:) = evaluate(obj, PopDec(i,:)');
            end
        end
        %% Generate points on the Pareto front
        function R = GetOptimum(obj,N)
            R = obj.GetFeasibleLowerBound(N);
            R(NDSort(R,1)~=1,:) = [];
        end
        %% Generate the image of Pareto front
        function R = GetPF(obj)
            % *Some parameter settings require more sampled points, such as very large dc_A
            switch obj.M
                case 2
                    R = obj.GetFeasibleLowerBound(1000);
                    R(NDSort(R,1)~=1) = nan;
                case 3
                    N_sqrt = 40;
                    tmp = obj.GetFeasibleLowerBound(N_sqrt^2);  % *Uniformly sampled PS does not correspond to uniformly sampled PF
                    tmp(NDSort(tmp,1)~=1) = nan;
                    R = cell(1,3);
                    for i = 0 : (N_sqrt^2-1)
                        R{1}(mod(i,N_sqrt)+1,floor(i/N_sqrt)+1) = tmp(i+1,1);
                        R{2}(mod(i,N_sqrt)+1,floor(i/N_sqrt)+1) = tmp(i+1,2);
                        R{3}(mod(i,N_sqrt)+1,floor(i/N_sqrt)+1) = tmp(i+1,3);
                    end
                otherwise
                    R = [];
            end
        end
        %% Generate points on the lower boundary of feasible region
        function R = GetFeasibleLowerBound(obj,N)
            x_grid = UniformPoint(N,obj.M-1,'grid')';
            R = zeros(N,obj.M);
            dc_A = obj.dc_A_;
            dc_alpha = obj.dc_alpha_;
            dc_beta = obj.dc_beta_;
            for i = 1 : size(x_grid,2)
                x = x_grid(:,i);
                X = zeros(obj.M,1);
                X(1) = (1-x(1));
                X(obj.M) = prod(x(1:obj.M-1,1));
                for j = 2 : obj.M-1
                    X(j) = prod(x(1:j-1,1))*(1-x(j));
                end
                R(i,:) = ([1-x(1)^dc_alpha*cos(dc_A*x(1)^dc_beta*pi)^2; X(2:end)].^obj.F_)';
            end
        end
        
%         function DrawObj(obj,Population)
%         %Overwrite DrawObj
% 
%             ax = Draw(Population.objs,{'\it f\rm_1','\it f\rm_2','\it f\rm_3'});
%             if ~isempty(obj.PF)
%                 if obj.M == 2
%                     plot(ax,obj.PF(:,1),obj.PF(:,2),'-k','LineWidth',1);
%                 elseif obj.M == 3
%                     plot3(ax,obj.PF(:,1),obj.PF(:,2),obj.PF(:,3),'ok','LineWidth',1);  % -k -> ok
%                 end
%             elseif size(obj.optimum,1) > 1 && obj.M < 4
%                 if obj.M == 2
%                     plot(ax,obj.optimum(:,1),obj.optimum(:,2),'.k');
%                 elseif obj.M == 3
%                     plot3(ax,obj.optimum(:,1),obj.optimum(:,2),obj.optimum(:,3),'.k');
%                 end
%             end
%         end
    end
end

function y = evaluate(obj, x)
    n = size(x,1);
    M=obj.M;
    X=zeros(M,1);
    g=zeros(M,1);
    J=cell(M,1);
    Jsize=zeros(M,1);
    co_r=ones(M,M)/(M-1);
    R=zeros(M,1);
    %%
    A=obj.A_;
    B=obj.B_;
    C=obj.C_;
    D=obj.D_;
    E=obj.E_;
    F=obj.F_;
    center=obj.center_;
    dc_A = obj.dc_A_;
    dc_alpha = obj.dc_alpha_;
    dc_beta = obj.dc_beta_;
    c_t_poly=obj.c_t_poly_;
    fre=obj.fre_;
    %%
    X(1)=(1-x(1));
    X(M)=prod(x(1:M-1,1));
    for i=2:M-1
        X(i)=prod(x(1:i-1,1))*(1-x(i));
    end
    %%
    cor=-M*diag(diag(co_r))+co_r;
    % r=sqrt((M-1)/M)*max(cor*(X-1/M));
    r=sqrt((M-1)/M)*max(cor*(X-center));
    % R(1:M-1)=1/(M-1)-1/M;
    % R(M)=-1/M;
    % R_long=sqrt(sum(R.^2));
    R_long = zeros(1,M);
    tmp = eye(M);
    for i = 1 : M
        R_long(i)=sqrt((M-1)/M)*max(cor*(tmp(:,i)-center));
    end
    R_long = max(R_long);
    h=(r/R_long);
    theta=h.^(M-1);
    Y_bias=A*(sin(0.5*pi*theta)^D)+1;
    X_bias=0.9*(sin(0.5*pi*theta)^B);
    %%
    t=x(M:n)-X_bias*cos(E*pi*repmat(h,[1,n-M+1])+0.5*pi*(n+2)*(M:n)/n)';
    J{1}=M:M:n;
    Jsize(1)=length(J{1});
    % g(1)=Y_bias/Jsize(1)*sum(abs(t(J{1}-M+1)).^C);
    g(1)=Y_bias/Jsize(1)*sum(c_t_poly*abs(t(J{1}-M+1)).^C - cos(fre*t(J{1}-M+1)) + 1);  % adopt g^2 in Appendix A of "A Generator ..."
    J{M}=2*M-1:M:n;
    Jsize(M)=length(J{M});
    % g(M)=Y_bias/Jsize(M)*sum(abs(t(J{M}-M+1)).^C);
    g(M)=Y_bias/Jsize(M)*sum(c_t_poly*abs(t(J{M}-M+1)).^C - cos(fre*t(J{M}-M+1)) + 1);
    for j=2:M-1
        J{j}=M+j-1:M:n;
        Jsize(j)=length(J{j});
        % g(j)=Y_bias/Jsize(j)*sum(abs(t(J{j}-M+1)).^C);
        g(j)=Y_bias/Jsize(j)*sum(c_t_poly*abs(t(J{j}-M+1)).^C - cos(fre*t(J{j}-M+1)) + 1);
    end
    % y=X.^F + g;
    y=[1-x(1)^dc_alpha*cos(dc_A*x(1)^dc_beta*pi)^2; X(2:end)].^F + g;  % ref: WFG2
end
