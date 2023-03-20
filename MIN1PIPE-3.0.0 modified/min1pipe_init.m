function min1pipe_init
% parse path, and install cvx if not
%   Jinghao Lu, 11/10/2017

    %%% prepare main folder %%%
    pathname = mfilename('fullpath');
    mns = mfilename;
    lname = length(mns);
    pathtop1 = pathname(1: end - lname);
    
    %%% check if on path %%%
    pathCell = regexp(path, pathsep, 'split');
    if ispc  % Windows is not case-sensitive
        onPath = any(strcmpi(pathtop1(1: end - 1), pathCell)); %%% get rid of filesep %%%
    else
        onPath = any(strcmp(pathtop1(1: end - 1), pathCell));
    end
    
    %%% set path and setup cvx if not on path %%%
    cvx_dir = [pathtop1, 'utilities'];
    pathcvx = [cvx_dir, filesep, 'cvx', filesep, 'cvx_setup.m'];
    if ~onPath
        pathall = genpath(pathtop1);
        addpath(pathall)
        if ~exist([cvx_dir, filesep, 'cvx'], 'dir')
            if ispc
                cvxl = 'http://web.cvxr.com/cvx/cvx-w64.zip';
            elseif isunix
                cvxl = 'http://web.cvxr.com/cvx/cvx-a64.zip';
            elseif ismac
                cvxl = 'http://web.cvxr.com/cvx/cvx-maci64.zip';
            end
            disp('Downloading CVX');
            unzip(cvxl, cvx_dir);
        end
    end
    if ~exist(fullfile(fileparts(prefdir), 'cvx_prefs.mat'), 'file')
        run(pathcvx);
    end
end






