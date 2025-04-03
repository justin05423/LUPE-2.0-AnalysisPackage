%This code will produce the Markov-K-Means behavioral state model.

% Models were trained on data in LUPE_statemodel_sourcedata.zip
%
% To generate models de novo, run sections in order. To reproduce Figs. 1,
% S7-9 load data of interest and stateModelClassifier.m, and skip section 3
% 


%SECTION 1: Load and downsample data
%SECTION 2: Generate sliding window transition matrices for different window lengths and choose k with silhouette and elbow method
%SECTION 3: Generate transition matrices for desired window length and cluster for chosen k
%SECTION 4: Classify states in each animal and validate over conditions
%SECTION 5: Export state data for each animal in group
%SECTION 6: Generate and calculate pain scale.

%Sophie A. Rogers, Corder Lab, University of Pennsylvania, March 24, 2025
%% Section 1: Load and downsample data

%TO RUN THIS AND REPRODUCE STATE MODEL FROM FIG 1, S7-8, UNZIP & LOAD FOLDER
%"LUPE_statemodel_sourcedata.zip". TO REPRODUCE DOSE RESPONSE FIGS LOAD
%THOSE FOLDERS INDIVIDUALLY AFTER LOADING StateModelClassifier.mat

%OUTPUTS:
% 1. behDS: matrix of behavior labels from recordings of all desired mice
% 2. anOrder: filenames the in order they were analyzed

myDir = uigetdir; %gets directory. 

% to generate state model in manuscript, use the 0mg/kg morphine data from 
% capsaicin & formalin experiments and the 3 weeks post-SNI YFP data from 
% the SNI experiment

myFiles = dir(fullfile(myDir,'*.csv')); %gets all csv files in struct
dtB = 60;

%change this to your desired sampling rate and real recording length:
dt = 20;
recLength = 30; %minutes
behDS = zeros(dt*dtB*recLength,1);

for k = 1:length(myFiles)
    
    
    baseFileName = myFiles(k).name;
    if length(baseFileName)<50
        continue
    end
    
    fullFileName = fullfile(myDir, baseFileName);
    data = readtable(fullFileName);
    data = table2array(data);
    
    behav = data(1:60*dtB*recLength,2);
    
    %UNCOMMENT THE BELOW IF RUNNING FOR WEEK 4 HM4DI-SNI MICE BECAUSE ONE
    %VIDEO WAS TOO SHORT (this zero-pads that data)
    %           if k==21
    %             behav = [data(:,2); zeros(30*60*60-size(data,1),1)];
    %           else
    %               behav = data(1:30*60*60,2);
    %           end
    
    
    %flip labels for left and right paw for cap experiments since those
    %injuries were in right paw instead of left
    if ~isempty(strfind(baseFileName,'Cap'))
          behav(behav==4) = 6;
          behav(behav==5) = 4;
          behav(behav==6) = 5;
    elseif ~isempty(strfind(myDir,'Cap'))
          behav(behav==4) = 6;
          behav(behav==5) = 4;
          behav(behav==6) = 5;
    end
 
    %record the order of animal files processed
    anOrder{k} = baseFileName;
    
    %downsample behavior
    rateRatio = dt/dtB;
    bd = zeros(round(length(behav(:,1))*rateRatio),1);
    for n=1:length(behDS)-1
        bd(n,1) = mode(behav((n-1)/rateRatio+1:n/rateRatio,1));
        if ~isinteger(bd(n,1))
            floor(bd(n,1)); %make sure outputs are categorical.
            %round down so it doesnt add a behavior category
        end
    end
    
    behDS(:,k) = bd;
    disp(strcat('Animal: ',num2str(k)))
end


%% Section 2: Generate sliding window transition matrices for different window lengths and choose k with silhouette and elbow method

%OUTPUTS:
% 1. figure: silhouette score for all values of k in all time window sizes of
% interest
% 2. figure: same for elbow method

nSecs = [10 30 60 100]; %window length (s)
toSlide = [5 10 10 10]; %seconds to slide windows by
nBeh = 6; %number of behaviors
dt = 20; %sampling rate
K_range = [1:20]; %k's to try


for n=1:length(nSecs)
    winSize = dt*nSecs(n); 
    winSlide = dt*toSlide(n); 

    wins = [1:winSlide:dt*recLength*60]; %window boundaries
    wins(wins+winSize+1>dt*recLength*60) = []; %cut windows that extend beyond recording end
    nWins = length(wins);

    clear transUnfolded
    transUnfolded = zeros(nWins,nBeh^2,size(behDS,2)); %initialize matrix to store unfolded transition matrices
    T=zeros(6,6);

    %loop through animals
    for a=1:size(behDS,2)
        tic
        data = behDS(:,a)+1; %get rid of zeros

        for t = 1:nWins
            %take a window of data
            slice = data(1+winSlide*(t-1):winSlide*(t-1)+winSize,:);

            %for each behavior, find the fraction of times it switches to the
            %other behavior
            for n=1:nBeh
                for l = 1:nBeh
                    idx = find(slice==n);
                    idx(idx == winSize) = [];
                    T(n,l) = mean(slice(idx+1)==l);
                end
            end
            %unfold
            transUnfolded(t,:,a) = reshape(T,1,nBeh^2);
        end
        toc
    end
    %get rid of nans
    transUnfolded(isnan(transUnfolded)) = 0;
    
    %concatenate unfolded transition matrices
    data = [];
    for a=1:size(behDS,2)
        data = [data; transUnfolded(:,:,a)];
    end
    
    %over a range of k
    K_range = 1:20;

    sumd_values = zeros(length(nSecs), 1);
    silhouette_values = zeros(length(K_range), 1);
    % Loop over each value of k
    for k_idx = 1:length(K_range)
        k = K_range(k_idx);
        tic
        for r = 1:100
            disp([k_idx r])
            % Run K-means clustering
            [idx, ~, sumd] = kmeans(data, k);

            % Store the sum of within-cluster distances (for elbow method)
            sumd_values(k_idx, r, n) = sum(sumd);

            % Compute silhouette values
            silhouette_scores = silhouette(data, idx);

            % Store the average silhouette score for this run
            silhouette_values(k_idx, r, n) = mean(silhouette_scores);
        end
        toc
    end

    % Elbow Method Calculation
    % Average the sumd values across runs for each k
    avg_within_cluster_distance = mean(sumd_values, 2);
    figure
    % Plot K vs. Sum of Squared Distances (Inertia)
    subplot(2,4,n)
    errorbar(K_range, mean(silhouette_values(:,:,n),2), std(silhouette_values(:,:,n),[],2)./sqrt(99), '-o');
    xlabel('Number of Clusters (K)');
    ylabel('Sum of Squared Distances');
    title(strcat('Silhouette Method for Optimal K: ',num2str(nSecs),'s bins'))

    subplot(2,4,n+4)
    errorbar(K_range, mean(sumd_values(:,:,n),2), std(sumd_values(:,:,n),[],2)./sqrt(99), '-o');
    xlabel('Number of Clusters (K)');
    ylabel('Sum of Squared Distances');
    title(strcat('Elbow Method for Optimal K: ',num2str(nSecs),'s bins'))

    nClusts(n,1) = find(mean(silhouette_values(:,:,n),2)==max(mean(silhouette_values(:,:,s),2)));
end

%% Section 3: Generate transition matrices for desired window length and cluster for chosen k

%OUTPUTS:
% 1. transUnfolded: timeseries of overlapping transition matrices for each
% animal
% 2. c: model centroids
% 3. figure: heatmap of transformed model centroids

nSecs = 30; %window length (s)
toSlide = 10; %seconds to slide windows by
nBeh = 6; %number of behaviors
dt = 20; %sampling rate

winSize = dt*nSecs; 
winSlide = dt*toSlide; 

wins = [1:winSlide:dt*recLength*60]; %window boundaries
wins(wins+winSize+1>dt*recLength*60) = []; %cut windows that extend beyond recording end
nWins = length(wins);

clear transUnfolded
transUnfolded = zeros(nWins,nBeh^2,size(behDS,2)); %initialize matrix to store unfolded transition matrices
T=zeros(6,6);

%loop through animals
for a=1:size(behDS,2)

    data = behDS(:,a)+1; %get rid of zeros

    for t = 1:nWins
        %take a window of data
        slice = data(1+winSlide*(t-1):winSlide*(t-1)+winSize,:);

        %for each behavior, find the fraction of times it switches to the
        %other behavior
        for n=1:nBeh
            for l = 1:nBeh
                idx = find(slice==n);
                idx(idx == winSize) = [];
                T(n,l) = mean(slice(idx+1)==l);
            end
        end
        %unfold
        transUnfolded(t,:,a) = reshape(T,1,nBeh^2);
    end

end
%get rid of nans
transUnfolded(isnan(transUnfolded)) = 0;

%concatenate unfolded transition matrices
data = [];
for a=1:size(behDS,2)
    data = [data; transUnfolded(:,:,a)];
end

k = 6;
[I,c] = kmeans(data,k);

%visualize state centroids. LOAD stateModelClassifier.m TO REPRODUCE STATE
%MODEL FOR MANUSCRIPT
behaviors = {'Still','Walk','Rear','Groom','Left Lick','Right Lick'};
figure
for m=1:k
    subplot(2,round(k/2),m)
    data=reshape(squeeze(c(m,:,:)),k,k);
    data = log10(1000000.*data); %log transform scaled data to visualize states
    data(isnan(data)) = 0;
    data(isinf(data)) = 0;
    heatmap(data,'XData',behaviors,'YData',behaviors)
    colormap(palette('scheme',4))
    title(strcat('State ',num2str(m)))
end
%% Section 4: Classify states in each animal and validate over conditions

% OUTPUTS:
% 1. ditMean: mean Euclidean distance of all real animals' data from the
% nearest model centroid, model fit metric
% 2. match: fraction of state assigments that match the full model when
% 3. removing different pieces of information

nSecs = 30; %window length (s)
toSlide = 10; %seconds to slide windows by
nBeh = 6; %number of behaviors
dt = 20; %sampling rate

winSize = dt*nSecs; 
winSlide = dt*toSlide; 

wins = [1:winSlide:dt*recLength*60]; %window boundaries
wins(wins+winSize+1>dt*recLength*60) = []; %cut windows that extend beyond recording end
nWins = length(wins);


% load stateModelClassifier.m %SET THIS TO YOUR FILE PATH

clear behState
clear ds
phase = 0; %set to phase 1 or 2 for formalin. when phase = 0, whole session is analyzed
winsP1 = [1:winSlide:dt*10*60]; %window boundaries
winsP1(winsP1+winSize+1>dt*10*60) = []; %cut windows that extend beyond recording end
nWinsP1 = length(winsP1);
if phase==1
    times=[1:nWinsP1];
elseif phase==2
    times=[nWinsP1+1:size(transUnfolded,1)];
else
    times=[1:size(transUnfolded,1)];
end


nVals = 100; %number of iterations for permutation test

%initialize outcomes
ditMeans=zeros(length(anOrder),9,nVals);

%indices of unfolded transition matrices to use when removing each behavior
subsetStill = [[1:7] 13 19 25 31];
subsetWalk = [2 [7:12] 14 20 26 32];
subsetRear = [3 9 [13:18] 21 27 33];
subsetGroom = [4 10 16 [19:24] 28 34];
subsetLick = [5 11 17 23 [25:30] 35];
subsetRight = [6 12 18 24 30 [31:36]];

subsets = {subsetStill, subsetWalk, subsetRear, subsetGroom, subsetLick, subsetRight};

subset2 = [[2:6] [[1] 3:6] [[1 2] 4:6] [[1:3] [5 6]] [[1:4] 6] [1:5]];


for l=1:9 %for each condition (full model, without self transitions, without stillness, without walking, without rearing, without grooming, without left lick, without right lick, and shuffle)
    if l<9
        nVals = 1;
    else
        nVals = 100; %only run 100x for shuffle b/c this algorithm is otherwise deterministic
    end
    
    for v=1:nVals %for each validation
        
        for a=1:size(behDS,2) %for each animal
            
            dists=[];
            data = transUnfolded(times,:,a);
            centroids = c; %centroids are the loaded behavioral state model
            
            if l==1 %if l = 1, use all information in model centroids
                data = data;
                centroids = c;
            elseif l==2 %if l=2, use only self --> other transitions
                data = data(:,subset2);
                centroids = centroids(:,subset2);
            elseif l==9 %if l=9, shuffle
                centroids = centroids(:,randperm(nBeh^2));
            else
                data(:,subsets{l-2}) = []; %if 2<l<9, use one of the subsets removing one behavior
                centroids(:,subsets{l-2}) = [];
            end
            
            % Compute euclidean distance of each transition matrix in the animal's time series to each centroid
            distances = pdist2(data,centroids); 
            [d, idx] = min(distances, [], 2); % Find the nearest centroid
            dists = [dists; d];
            
            
            ds(:,:,a) = distances;
            
            %resample state classifications to 20Hz
            scaledIdx = zeros(length(toSlide*dt*length(idx)),1);
            
            for t=1:length(idx)
                scaledIdx(toSlide*dt*(t-1)+1:toSlide*dt*(t+1)) = idx(t);
                
            end
            
            %store behavior state time series for each animal and condition
            behState(:,a,l) = scaledIdx;
            
            %store model fit (how big is the distance between real data to
            %nearest model centroid at any given time
            ditMeans(a,l) = mean(dists);
            
            d1 = behState(:,a,1);
            d2 = behState(:,a,l);
            
            %store similarity metric between each condition and full model
            match(a,l,v) = mean(d1==d2);
        end
    end
end
%% Section 5: Calculate and export outcomes

% OUTPUTS:
% 1. fracTime: fraction occupancy of each animal in each state
% 2. nBouts: number of bouts of each state in each anmal
% 3. boutDur: mean bout duration in seconds of each state in animal

fracTime = zeros(length(anOrder),6);
nBouts = zeros(length(anOrder),6);
boutDur = zeros(length(anOrder),6);

for a=1:size(behDS,2)
    tic
        for s=1:6
            for l=1
                if l==1
                state = behState(:,a,1);
                else
                    state = behState{a,2};
                end
                state(state~=s) = 0;
                state(state==s) = 1;

                fracTime(a,s,l) = mean(state);
                
                bouts = strfind(state',[0 1]);
                
                nBouts(a,s,l) = length(bouts);
                
                
                off = strfind(state',[1 0]);
                
                if length(off)>1 & length(bouts)>1
                    bouts(bouts>off(end)) = [];
                    off(off<bouts(1)) = [];
                    

                    boutDur(a,s,l) = mean(off-bouts)./20;
                elseif isempty(bouts)
                    boutDur(a,s,l) = 0;
                else
                    if max(bouts)>max(off)
                        set1 = off-bouts(bouts<off);
                        boutDur(a,s,l) = mean([set1 (length(state)-bouts(end))])./20;
                    else
                        boutDur(a,s,l) = mean((off-bouts))./20;
                    end
                end
            end
        end
    toc
end


animalNames = cell(size(behDS,2),1);
for a=1:length(animalNames)
    animalNames{a} = anOrder{a}(1:50);
end


dat = array2table(ditMeans','VariableNames', animalNames, 'RowNames',{'Model','No self-transition','No still', 'No walk','No rear','No groom','No left lick','No right lick','Shuffled'});
fileName = 'modelFit.csv';
fullFilePath = fullfile(myDir, fileName);
writetable(dat,fullFilePath)


dat = array2table(fracTime(:,:,1)','VariableNames', animalNames, 'RowNames',{'State 1','State 2','State 3','State 4','State 5','State 6'});
fileName = 'fractionOccupancy.csv';
fullFilePath = fullfile(myDir, fileName);
writetable(dat,fullFilePath)

dat = array2table(nBouts(:,:,1)','VariableNames', animalNames, 'RowNames',{'State 1','State 2','State 3','State 4','State 5','State 6'});
fileName = 'stateBouts.csv';
fullFilePath = fullfile(myDir, fileName);
writetable(dat,fullFilePath)

dat = array2table(boutDur(:,:,1)','VariableNames', animalNames, 'RowNames',{'State 1','State 2','State 3','State 4','State 5','State 6'});
fileName = 'stateDuration.csv';
fullFilePath = fullfile(myDir, fileName);
writetable(dat,fullFilePath)

%% Section 6: Pain Scale

%OUTPUTS:
% 1. painScale: 2nd dimension scores of projection of whichever data you are 
% analyzing into PC space formed by state occupancies of injured and uninjured 
% animals
% 2. generalizedBehaviorScale: 1st dimension scores of projection of whichever data you are 
% analyzing into PC space formed by state occupancies of injured and uninjured 
% animals
% 3. figure: scatter plot of PC scores and projections

%these data are the fraction time spent in each of six states (rows 1-6 are
%states 1-6) by all of the training data mice + uninjured mice
data=[0.797752808988764	0.943820224719101	0.943820224719101	0.915730337078652	0.685393258426966	0.921348314606742	0.887640449438202	0.837078651685393	0.707865168539326	0.865168539325843	0.910112359550562	0.921348314606742	0.550561797752809	0.359550561797753	0.808988764044944	0.331460674157303	0.820224719101124	0.831460674157303	0.865168539325843	0.904494382022472	0.567415730337079	0.719101123595506	0.719101123595506	0.735955056179775	0.325842696629214	0.629213483146067	0.606741573033708	0.561797752808989	0.207865168539326	0.421348314606742	0.814606741573034	0.758426966292135	0.612359550561798	0.185393258426966	0.443820224719101	0.426966292134831	0.584269662921348	0.685393258426966	0.634831460674157	0.679775280898876	0.870786516853933	0.533707865168539	0.792134831460674	0.780898876404494	0.230337078651685	0.747191011235955	0.887640449438202	0.617977528089888	0.47752808988764	0.792134831460674	0.764044943820225	0.426966292134831	0.337078651685393	0.410112359550562	0.230337078651685	0.764044943820225	0.589887640449438	0.52247191011236	0.702247191011236	0.185393258426966	0.275280898876405	0.258426966292135	0.365168539325843	0.151685393258427	0.230337078651685	0.410112359550562	0.539325842696629	0.651685393258427	0.455056179775281	0.533707865168539	0.556179775280899	0.241573033707865	0.308988764044944	0.52247191011236	0.202247191011236	0.398876404494382	0.466292134831461	0.550561797752809	0.196629213483146	
0.140449438202247	0.0449438202247191	0	0.0617977528089888	0.106741573033708	0.050561797752809	0.0280898876404494	0.0224719101123595	0.00561797752808989	0.0617977528089888	0.050561797752809	0.050561797752809	0.314606741573034	0.567415730337079	0.157303370786517	0.623595505617977	0.129213483146067	0.112359550561798	0.0280898876404494	0.0617977528089888	0.0449438202247191	0.0112359550561798	0.0786516853932584	0.0224719101123595	0.174157303370787	0.247191011235955	0.174157303370787	0.179775280898876	0.320224719101124	0.162921348314607	0.0617977528089888	0.117977528089888	0.219101123595506	0.573033707865168	0.370786516853933	0.308988764044944	0.134831460674157	0.168539325842697	0.123595505617978	0.101123595505618	0.00561797752808989	0.00561797752808989	0.0112359550561798	0.00561797752808989	0.410112359550562	0.0955056179775281	0.0168539325842697	0.168539325842697	0.353932584269663	0.0337078651685393	0.106741573033708	0.337078651685393	0.393258426966292	0.286516853932584	0.49438202247191	0.168539325842697	0.191011235955056	0.0337078651685393	0.0561797752808989	0.724719101123595	0.129213483146067	0.230337078651685	0.258426966292135	0	0.589887640449438	0.202247191011236	0.202247191011236	0.0674157303370786	0.224719101123595	0.303370786516854	0.207865168539326	0	0.438202247191011	0.247191011235955	0.219101123595506	0.393258426966292	0.292134831460674	0.162921348314607	0.308988764044944	
0	0	0	0.00561797752808989	0.0168539325842697	0	0	0	0.00561797752808989	0	0	0	0.00561797752808989	0.0224719101123595	0	0.0224719101123595	0	0	0	0	0.00561797752808989	0.0112359550561798	0.00561797752808989	0.0112359550561798	0.134831460674157	0.0112359550561798	0.050561797752809	0.0617977528089888	0.140449438202247	0.134831460674157	0	0.0168539325842697	0.0168539325842697	0.0674157303370786	0.0337078651685393	0.162921348314607	0.0280898876404494	0.00561797752808989	0.0168539325842697	0.0112359550561798	0	0	0.0561797752808989	0.0168539325842697	0.174157303370787	0.0280898876404494	0.0168539325842697	0.0786516853932584	0.0224719101123595	0	0.0280898876404494	0.00561797752808989	0.129213483146067	0.00561797752808989	0.168539325842697	0.0112359550561798	0.00561797752808989	0	0.0112359550561798	0.0393258426966292	0.0674157303370786	0.0842696629213483	0.258426966292135	0.0393258426966292	0.140449438202247	0.112359550561798	0.0561797752808989	0.00561797752808989	0.0674157303370786	0.0393258426966292	0.0730337078651685	0.0674157303370786	0.134831460674157	0.151685393258427	0.370786516853933	0.0898876404494382	0.050561797752809	0.0280898876404494	0.219101123595506	
0.0393258426966292	0	0.0168539325842697	0.0168539325842697	0.179775280898876	0.0168539325842697	0.050561797752809	0.0617977528089888	0.0617977528089888	0.050561797752809	0	0	0.117977528089888	0.0337078651685393	0.0168539325842697	0.0168539325842697	0.0224719101123595	0	0	0.0168539325842697	0.353932584269663	0.241573033707865	0.191011235955056	0.219101123595506	0.337078651685393	0.112359550561798	0.162921348314607	0.140449438202247	0.308988764044944	0.241573033707865	0.117977528089888	0.0898876404494382	0.129213483146067	0.151685393258427	0.123595505617978	0.0842696629213483	0.241573033707865	0.0449438202247191	0.207865168539326	0.185393258426966	0.0786516853932584	0.460674157303371	0.123595505617978	0.191011235955056	0.134831460674157	0.129213483146067	0.0730337078651685	0.0842696629213483	0.129213483146067	0.174157303370787	0.0842696629213483	0.213483146067416	0.129213483146067	0.286516853932584	0.0617977528089888	0.050561797752809	0.146067415730337	0.421348314606742	0.219101123595506	0	0.49438202247191	0.314606741573034	0.0955056179775281	0.808988764044944	0.0393258426966292	0.269662921348315	0.196629213483146	0.252808988764045	0.219101123595506	0.0617977528089888	0.151685393258427	0.668539325842697	0.106741573033708	0.0786516853932584	0.140449438202247	0.112359550561798	0.101123595505618	0.219101123595506	0.157303370786517	
0.0224719101123595	0.0112359550561798	0.0393258426966292	0	0.00561797752808989	0.0112359550561798	0.0337078651685393	0.0786516853932584	0.219101123595506	0.0224719101123595	0.0393258426966292	0.0280898876404494	0	0.0168539325842697	0.0168539325842697	0	0.0280898876404494	0.0561797752808989	0.106741573033708	0	0.0280898876404494	0.0168539325842697	0.00561797752808989	0.0112359550561798	0.0280898876404494	0	0.00561797752808989	0.050561797752809	0	0.0337078651685393	0.00561797752808989	0.0168539325842697	0.0224719101123595	0.0112359550561798	0.00561797752808989	0.00561797752808989	0	0.0786516853932584	0.0168539325842697	0.0112359550561798	0.0337078651685393	0	0.0168539325842697	0.00561797752808989	0.050561797752809	0	0	0.050561797752809	0.0168539325842697	0	0.0112359550561798	0.00561797752808989	0.0112359550561798	0	0	0.00561797752808989	0.0674157303370786	0.0168539325842697	0.0112359550561798	0.050561797752809	0.0224719101123595	0.0561797752808989	0	0	0	0.00561797752808989	0.00561797752808989	0.0224719101123595	0	0.00561797752808989	0	0.0224719101123595	0	0	0	0	0.0898876404494382	0.0224719101123595	0	
0	0	0	0	0.00561797752808989	0	0	0	0	0	0	0	0.0112359550561798	0	0	0.00561797752808989	0	0	0	0.0168539325842697	0	0	0	0	0	0	0	0.00561797752808989	0.0224719101123595	0.00561797752808989	0	0	0	0.0112359550561798	0.0224719101123595	0.0112359550561798	0.0112359550561798	0.0168539325842697	0	0.0112359550561798	0.0112359550561798	0	0	0	0	0	0.00561797752808989	0	0	0	0.00561797752808989	0.0112359550561798	0	0.0112359550561798	0.0449438202247191	0	0	0.00561797752808989	0	0	0.0112359550561798	0.0561797752808989	0.0224719101123595	0	0	0	0	0	0.0337078651685393	0.0561797752808989	0.0112359550561798	0	0.0112359550561798	0	0.0674157303370786	0.00561797752808989	0	0.0168539325842697	0.117977528089888	];
																																																																															
																																																																															
																																																																															
																																																																																																																																																						
data = data';

[co,sc, ~, ~, ex, mu] = pca(data);
scale = project_to_pc_space(fracTime(:,:,1),co,mu);

painScale = scale(:,2);
generalBehaviorScale = scale(:,1);

figure
subplot(121)
scatter(sc(1:20,1), sc(1:20,2),'k')
hold on
scatter(sc(21:40,1), sc(21:40,2),'c')
hold on
scatter(sc(41:60,1), sc(41:60,2),'m')
hold on
scatter(sc(61:79,1), sc(61:79,2),'r')
legend({'Uninjured','Capsaicin','Formalin','SNI'})
title('PCA of state occupancy')
xlabel('PC1: Generalized behavior scale')
ylabel('PC2: Pain behavior scale')
subplot(122)
scatter(sc(1:20,1), sc(1:20,2),'c')
hold on
scatter(sc(21:39,1), sc(21:39,2),'m')
hold on
scatter(sc(40:58,1), sc(40:58,2),'r')
legend({'Capsaicin','Formalin','SNI'})
title('Projections')
xlabel('PC1: Generalized behavior scale')
ylabel('PC2: Pain behavior scale')