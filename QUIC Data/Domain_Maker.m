%% Domain_Maker.m
% Nipun Gunawardena
% Reads four files from QUIC output and creates matrix of all the
% obstacles and concentrations. Also reads experiment parameters and saves
% all data in a form usable by the PSO code. See README for files required
% to run this script. This script should be run before the PSO code

clear all, clc, close all



%% Read Building Data
folderPath = './OKC/';


%% Read domain parameters
paramPath = strcat(folderPath, 'QU_simparams.inp');
paramStr = fileread(paramPath);
paramSplit = strsplit(paramStr, '\n');

% Get cells in domain
xCells = quicParamParse(paramSplit{2});   % Number of cells
yCells = quicParamParse(paramSplit{3});   % Number of cells
zCells = quicParamParse(paramSplit{4});   % Number of cells

% Get grid spacing
dx = quicParamParse(paramSplit{5});  % Meters
dy = quicParamParse(paramSplit{6});  % Meters
dz = quicParamParse(paramSplit{8});  % Meters


%% Read concentration parameters
paramPath = strcat(folderPath, 'Qp_params.inp');
paramStr = fileread(paramPath);
paramSplit = strsplit(paramStr, '\n');

duration = quicParamParse(paramSplit{23});  % Seconds
avgTime = quicParamParse(paramSplit{24});    % Seconds
numPeriods = duration/avgTime;


%% Read source parameters
paramPath = strcat(folderPath, 'Qp_source.inp');
paramStr = fileread(paramPath);
paramSplit = strsplit(paramStr, '\n');

sourceLoc = [quicParamParse(paramSplit{13}), quicParamParse(paramSplit{14}), quicParamParse(paramSplit{15})];   % Source Location


%% Filter Building Data
filePath = strcat(folderPath, 'celltype.mat');
qc = load(filePath);
domain = qc.celltype{1};
domain(domain ~= 0) = NaN;
domain = domain(:,:,2:end);

% Account for OKC data
if strcmp(folderPath, './OKC/')
    domain = domain(:,:,1:60);
end


%% Create Plotting Coordinates
[X, Y, Z] = ndgrid(1:size(domain,1), 1:size(domain,2), 1:size(domain,3)); 
X = X*dx;
Y = Y*dy;
Z = Z*dz;


%% Read Concentration Data for Simple Plotting
filePath = strcat(folderPath, 'concentration.mat');
qc = load(filePath);
C = qc.concentration;

for i = 1:length(C)
    ct = C{i};
    ct(ct == 0) = NaN;
    C{i} = ct;
end


%% Check Building vs. Concentration
for i = 1:length(C)
    if ~(all(size(domain) == size(C{i})))
        error('Domain size and concentration size are not equal!');
    end
end


%% Plot for visualization
% dIdx = ~isnan(domain);
% plotC = C{1}(:);    % !! May break here depending on number of averaging time periods present in simulation
% meanC = nanmean(plotC);
% stdC = nanstd(plotC);
% lowCIdx = (plotC <= meanC + 2*stdC);    % Low concentrations are plotted different from high concentrations
% hiCIdx = (plotC > meanC + 2*stdC);
% 
% figure('units','normalized','outerposition',[0 0 1 1]);
% hold on
% grid on
% 
% buildingAlpha = 1.0;
% lowCAlpha = 0.1;
% hiCAlpha = 0.6;
% pointsize = 7;
% 
% colormap hot
% plot3(X(dIdx), Y(dIdx), Z(dIdx), 'g.', 'MarkerSize', 75);
% scatter3(X(lowCIdx), Y(lowCIdx), Z(lowCIdx), pointsize, plotC(lowCIdx), 'filled', 'MarkerFaceAlpha',lowCAlpha, 'MarkerEdgeAlpha',lowCAlpha);
% scatter3(X(hiCIdx), Y(hiCIdx), Z(hiCIdx), pointsize, plotC(hiCIdx), 'filled', 'MarkerFaceAlpha',hiCAlpha, 'MarkerEdgeAlpha',hiCAlpha);
% scatter3(sourceLoc(2), sourceLoc(1), sourceLoc(3), 300, 'filled', 'bp');
% axis equal
% xlim([min(X(:)) max(X(:))]);
% ylim([min(Y(:)) max(Y(:))]);
% 
% xlabel('x (m)');
% ylabel('y (m)');
% zlabel('z (m)');
% view(2);



%% Save Data
save(strcat(folderPath, 'Data.mat'), 'domain', 'C', 'xCells', 'yCells', 'zCells', 'dx', 'dy', 'dz', 'X', 'Y', 'Z', 'duration', 'avgTime', 'numPeriods', 'sourceLoc');