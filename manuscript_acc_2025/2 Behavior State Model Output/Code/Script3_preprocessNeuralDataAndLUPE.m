% This code was used to preprocess all raw calcium and behavior label
% files. this is run with the capsaicin source data due to smaller
% experiment size. preprocessed data for SNI and uninjured mice are
% provided to skip straight to file 4. 

% Section 1: load data
% Section 2: clean calcium data, reorganize and downsample behavior data
% Section 3: collect peri-behavioral time histograms. Reproduce Fig. 2k

%Sophie A. Rogers, Corder Lab, University of Pennsylvania, March 23, 2025
%% Section 1: load data
%behavior
tic
oswell.animals(1).sessions(1).behavior = table2array(readtable('m76_bl.csv'));
oswell.animals(1).sessions(2).behavior = table2array(readtable('m76_cap.csv'));
oswell.animals(1).sessions(3).behavior = table2array(readtable('m76_mor.csv'));
oswell.animals(1).sessions(4).behavior = table2array(readtable('m76_morcap.csv'));
toc
tic
oswell.animals(2).sessions(1).behavior = table2array(readtable('m81_bl.csv'));
oswell.animals(2).sessions(2).behavior = table2array(readtable('m81_cap.csv'));
oswell.animals(2).sessions(3).behavior = table2array(readtable('m81_mor.csv'));
oswell.animals(2).sessions(4).behavior = table2array(readtable('m81_morcap.csv'));
toc
tic
oswell.animals(3).sessions(1).behavior = table2array(readtable('m84_bl.csv'));
oswell.animals(3).sessions(2).behavior = table2array(readtable('m84_cap.csv'));
oswell.animals(3).sessions(3).behavior = table2array(readtable('m84_mor.csv'));
oswell.animals(3).sessions(4).behavior = table2array(readtable('m84_morcap.csv'));
toc
tic
oswell.animals(4).sessions(1).behavior = table2array(readtable('m88_bl.csv'));
oswell.animals(4).sessions(2).behavior = table2array(readtable('m88_cap.csv'));
oswell.animals(4).sessions(3).behavior = table2array(readtable('m88_mor.csv'));
oswell.animals(4).sessions(4).behavior = table2array(readtable('m88_morcap.csv'));
toc
%temporal alignment
oswell.animals(1).sessions(1).offset = 30;
oswell.animals(1).sessions(2).offset = 40;
oswell.animals(1).sessions(3).offset = 30;
oswell.animals(1).sessions(4).offset = 30;

oswell.animals(2).sessions(1).offset = 30;
oswell.animals(2).sessions(2).offset = 30;
oswell.animals(2).sessions(3).offset = 30;
oswell.animals(2).sessions(4).offset = 30;

oswell.animals(3).sessions(1).offset = 30;
oswell.animals(3).sessions(2).offset = 30;
oswell.animals(3).sessions(3).offset = 30;
oswell.animals(3).sessions(4).offset = 30;

oswell.animals(4).sessions(1).offset = 30;
oswell.animals(4).sessions(2).offset = 30;
oswell.animals(4).sessions(3).offset = 30;
oswell.animals(4).sessions(4).offset = 30;

%neural activity
tic
oswell.animals(1).sessions(1).calcium = table2array(readtable('MM76_baseline.csv'));
oswell.animals(1).sessions(2).calcium = table2array(readtable('MM76_capsaicin.csv'));
oswell.animals(1).sessions(3).calcium = table2array(readtable('MM76_morphine.csv'));
oswell.animals(1).sessions(4).calcium = table2array(readtable('MM76_morphinecapsaicin.csv'));
toc
tic
oswell.animals(2).sessions(1).calcium = table2array(readtable('MM81_baseline.csv'));
oswell.animals(2).sessions(2).calcium = table2array(readtable('MM81_capsaicin.csv'));
oswell.animals(2).sessions(3).calcium = table2array(readtable('MM81_morphine.csv'));
oswell.animals(2).sessions(4).calcium = table2array(readtable('MM81_morphinecapsaicin.csv'));
toc
tic
oswell.animals(3).sessions(1).calcium = table2array(readtable('MM84_baseline.csv'));
oswell.animals(3).sessions(2).calcium = table2array(readtable('MM84_capsaicin.csv'));
oswell.animals(3).sessions(3).calcium = table2array(readtable('MM84_morphine.csv'));
oswell.animals(3).sessions(4).calcium = table2array(readtable('MM84_morphinecapsaicin.csv'));
toc
tic
oswell.animals(4).sessions(1).calcium = table2array(readtable('MM88_baseline.csv'));
oswell.animals(4).sessions(2).calcium = table2array(readtable('MM88_capsaicin.csv'));
oswell.animals(4).sessions(3).calcium = table2array(readtable('MM88_morphine.csv'));
oswell.animals(4).sessions(4).calcium = table2array(readtable('MM88_morphinecapsaicin.csv'));
toc
tic
oswell.animals(5).sessions(1).calcium = table2array(readtable('M1_baseline_deconvolved.csv'));
oswell.animals(5).sessions(2).calcium = table2array(readtable('M1_capsaicin_decovolvedtraces.csv'));
oswell.animals(5).sessions(3).calcium = table2array(readtable('M1_baseline_deconvolved.csv'));
oswell.animals(5).sessions(4).calcium = table2array(readtable('M1_morphinecapsaicin_deconvolvedtraces.csv'));
toc


tic
oswell.animals(5).sessions(1).behavior = table2array(readtable('M1_baseline_file0.csv'));
oswell.animals(5).sessions(2).behavior = table2array(readtable('M1_cap_file0.csv'));
oswell.animals(5).sessions(3).behavior = table2array(readtable('M1_morphine_file0.csv'));
oswell.animals(5).sessions(4).behavior = table2array(readtable('M1_morphine-capsaicin_file0.csv'));
toc

%temporal alignment
oswell.animals(5).sessions(1).offset = 30;
oswell.animals(5).sessions(2).offset = 82;
oswell.animals(5).sessions(3).offset = 30;
oswell.animals(5).sessions(4).offset = 30;

oswell.animals(1).sessions(1).props = readtable('MM76_baseline-props.csv');
oswell.animals(1).sessions(2).props = readtable('MM76_capsaicin-props.csv');
oswell.animals(1).sessions(3).props = readtable('MM76_morphine-props.csv');
oswell.animals(1).sessions(4).props = readtable('MM76_morphinecapsaicin-props.csv');

oswell.animals(1).sessions(1).props = readtable('MM76_baseline-props.csv');
oswell.animals(1).sessions(2).props = readtable('MM76_capsaicin-props.csv');
oswell.animals(1).sessions(3).props = readtable('MM76_morphine-props.csv');
oswell.animals(1).sessions(4).props = readtable('MM76_morphinecapsaicin-props.csv');

oswell.animals(2).sessions(1).props = readtable('MM81_baseline-props.csv');
oswell.animals(2).sessions(2).props = readtable('MM81_capsaicin-props.csv');
oswell.animals(2).sessions(3).props = readtable('MM81_morphine-props.csv');
oswell.animals(2).sessions(4).props = readtable('MM81_morphinecapsaicin-props.csv');

oswell.animals(3).sessions(1).props = readtable('MM84_baseline-props.csv');
oswell.animals(3).sessions(2).props = readtable('MM84_capsaicin-props.csv');
oswell.animals(3).sessions(3).props = readtable('MM84_morphine-props.csv');
oswell.animals(3).sessions(4).props = readtable('MM84_morphinecapsaicin-props.csv');

oswell.animals(4).sessions(1).props = readtable('MM88_baseline-props.csv');
oswell.animals(4).sessions(2).props = readtable('MM88_capsaicin-props.csv');
oswell.animals(4).sessions(3).props = readtable('MM88_morphine-props.csv');
oswell.animals(4).sessions(4).props = readtable('MM88_morphinecapsaicin-props.csv');

oswell.animals(5).sessions(1).props = readtable('M1_baseline_deconvolved-props.csv');
oswell.animals(5).sessions(2).props = readtable('M1_capsaicin_decovolvedtraces-props.csv');
oswell.animals(5).sessions(3).props = readtable('M1_baseline_deconvolved-props.csv');
oswell.animals(5).sessions(4).props = readtable('M1_morphinecapsaicin_deconvolvedtraces-props.csv');


nAnimals = 5;
nSesh = 4; %nSessions
dt=20; %neural data sampling rate
dtB = 60; %behavior sampling rate


behCols  = {'r',[1.0000    0.4980    0.3137],'y','g','c','b'};
sessCols = {'k','r','b',[102 51 153]./255};
pcCols = {'r',[1.0000    0.4980    0.3137],'g','b'};

behaviors = {'Still','Walking','Rearing','Grooming','Left lick','Right lick'};
sessions = {'Baseline', 'Capsaicin', 'Morphine','Capsaicin+Morphine'};

%% Section 2: clean calcium data, reorganize and downsample behavior data

activities = cell(nSesh,nAnimals);
for a=1:nAnimals
    
    behMatTot = [];
    for m = 1:nSesh

    %downsample behavioral data to 20Hz. 
        %since its categorical, interpolation wasn't appropriate so i took the mode
        %across 3 frames. sloppy, im sure theres a function but whatever
        rateRatio = dt/dtB;
        behDS = zeros(round(length(oswell.animals(a).sessions(m).behavior(:,2))*rateRatio),1);
        for n=1:length(behDS)-1
            behDS(n,1) = mode(oswell.animals(a).sessions(m).behavior((n-1)/rateRatio+1:n/rateRatio,2));
            if ~isinteger(behDS(n,1))
                floor(behDS(n,1)); %make sure outputs are categorical. 
                                    %round down so it doesnt add a behavior category
            end
        end


        %plot original and downsampled behavior to ensure accuracy
        % figure
        % histogram(oswell.animals(1).sessions(m).behavior(:,2))
        % hold on
        % histogram(behDS)

        %create a binary matrix version of OG behavior data, with each behavior 
        %variable in columns and times in rows. 
        behs = [0:5];
        behMat = zeros(length(behDS),length(behs));
        for n=1:length(unique(behDS))
            behMat(find(behDS==behs(n)),n) = 1;
        end

        %align neural data by taking out times before recording offset
        data =oswell.animals(a).sessions(m).calcium(oswell.animals(a).sessions(m).offset*dt:end,2:end);
        
        %remove broken frames/missing data
        data(isnan(data)) = 0;

        %z-score neural activity
        data = zscore(data);
        
        
        if length(data)<length(behMat)
            behMat = behMat(1:length(data),:);
            behDS = behDS(1:length(data),:);
        else
            data = data(1:length(behMat),:);
        end
        
        %store aligned neural activities
        activities{m,a} = data;
        
        %store aligned behavior matrix
        behMats{m,1,a} = behMat;

        %store downsampled data
        behMats{m,2,a} = behDS;

        %concatenate behavior mats across sessions for PCA
        behMatTot = [behMatTot; behMats{m,1,a}];

        %store length of each session
        lens(m,a) = length(behMats{m,1,a});

        
    end
end

%% Section 3: collect peri-behavioral time histograms. Reproduce Fig. 2k

%OUTPUTS:
% 1. psthStore: cell of PBTHs of every cell +/-2s around behavior onset
% zscored to baseline (-2-0s prior to onset) for each animal, session,
% and behavior
% 2. psthBehStore: PBTHs of behavior -2 - +2s around onset for each animal,
% session, and behavior

w=40; %number of frames to take
psthStore=cell(nAnimals,nSesh,nBeh);
psthBehStore=cell(nAnimals,nSesh,nBeh);

for a=1:nAnimals
    tic
    disp(strcat('Animal ', num2str(a)))
    for m=1:nSesh
        bData = behMats{m,1,a};
        aDat = activities{m,a};
        %get behavior and activities
        
        for b=1:nBeh
            %identify bout starts
            
            bDat = bData(:,b);
            
            bouts = strfind(bDat',[0, 1]);%);
            
            %cut bouts that are too close to the ends for our time window
            bouts(bouts<w) = [];
            bouts(bouts>length(bDat)-w) = [];
            
            %initialize neural and behavioral psths (bouts x time x
            %neurons)
            psthMat = zeros(length(bouts),w*2,size(aDat,2));
            psthBeh = zeros(length(bouts),w*2);
            
            for p=1:length(bouts)
                
                %collect activity of all neurons throughout bout window
                psthMat(p,:,:) = aDat(bouts(p)-w+1:bouts(p)+w,:);
                %same for behavior
                psthBeh(p,:) = bDat(bouts(p)-w+1:bouts(p)+w,1);
                
            end
            
            psthBehStore{a,m,b} = psthBeh;
            
            
            psthStore{a,m,b}= psthMat;
            
        end
        
    end
    toc
end
%%
figure
for m=1:nSesh
    for n=1:nBeh
        subplot(nSesh,nBeh,nBeh*(m-1)+n)
        data = [];
        for a=1:nAnimals
            data = [data; psthBehStore{a,m,b}];
        end
        data = data';
        win = [1:size(data,1)];
        plot(nanmean(data(win,:),2),'color',sessCols{m},'Linewidth',1)
        hold on
        y_upper = nanmean(data(win,:),2)+nanstd(data(win,:),[],2)./(sqrt(size(data(win,:),2)-1));
        y_lower = nanmean(data(win,:),2)-nanstd(data(win,:),[],2)./(sqrt(size(data(win,:),2)-1));
        fill([1:size(data(win,:), 1), fliplr(1:size(data(win,:), 1))], [y_upper; flipud(y_lower)], sessCols{m}, 'FaceAlpha', 0.2, 'EdgeColor', 'none');
        hold on
        if m==1
            title(behaviors{n})
        end
        if n==1
            ylabel(sessions{m})
        end
        xticks(linspace(0,80,5))
        xticklabels(linspace(-40,40,5))
    end
end