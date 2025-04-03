% This code analyzes neural responses to sensory stimuli and will reproduce
% Fig. S11. It can also be used to reproduce Fig. S12 

%SECTION 1: load data
%SECTION 2: calculate and plot PSTHs
%SECTION 3: identify significantly enhanced or inhibited cells
%SECTION 4: make heatmaps of neurons.
%SECTION 5: calculate stimulus overlaps.
%SECTION 6: decode stimuli.

%Sophie A. Rogers, Corder Lab, University of Pennsylvania, March 25, 2025

%% Section 1: load data 
%(can be adapted with painPanel data for Fig. S11 but for now Fig. S12)

load painvalence.mat %REPLACE WITH FILE PATH

nAnimals=size(painvalence.animals,2);
nStim=length(painvalence.modeTypes);
dt=20;
%% Section 2: calculate and plot PSTHs.

%OUTCOMES:
% 1. psths: cell containing baselined psths (z-score) for every neuron 
% around each stimulus
% 2. figure: psths for all cells and stim-responsive cells for each 
% stimulus

for a=1:nAnimals
    nCells(a,1) = size(table2array(painvalence.animals(a).calcium(:,2:end)),2);
end

%first, only run this code with subset 1. once you have completed section
%3, uncomment the below run for each subset of neurons (up
subsets = [1:nStim*2+1];

%seconds before and after stim to take
window = 60;
l=0;

figure
for d=subsets
    
    %define cell subset of interest. if d is even, look at activated
    %neurons (dir). stimulus index (pop) is d/2 rounded down.
    if d>1
        if mod(d,2) == 0
            dir = 1;
            dir_string = 'Up';
        else
            dir = 2;
            dir_string = 'Down';
        end
        pop = floor(d/2);
    end
    
    for m=1:nStim
        tic
        collected = [];
        l=0;
        for a=1:nAnimals
            disp([a m d])
            if d==1
                cellset = 1:nCells(a);
            else
                cellset = responsive{a,pop,dir};
            end
           
            %count neurons
            l=l+1;
            data = table2array(painvalence.animals(a).calcium(:,2:end));
            nCells(a,1) = size(data,2);
            
            
            %for pain panel, cut broken recordings
            if ismember(a,[12 13]) & m==5
                continue
            end
            
            
            %number of trials for that stimulus
            nStims=length(painvalence.animals(a).modes(m).trials);
            
            if d==1
                data2=data;
                
                %generate psths 60s before stim to 60s after stim
                psth = zeros(window*2,nCells(a,1),nStims);
                stimTimes = painvalence.animals(a).modes(m).trials.*dt;
                for t=1:nStims
                    psth(:,:,t) = data2(stimTimes(t)-window:stimTimes(t)+window-1,:);
                end
                dat = psth;
                
                %z-score to baseline
                dat = (dat-mean(dat(1:window,:,:)))./std(dat(1:window,:,:));
                
                psths{a,m} = dat;
            else
                dat = psths{a,m}(:,cellset,:);
            end
            
            %flatten crazy numbers that
            dat(dat>30) = 30;
            dat(dat<-30) = -30;
            dat = mean(dat,3);
            
            
            
            collected = [collected dat];
            
        end
        
        subplot(max(subsets),nStim,nStim*(d-1)+m)
        dat=collected;
        plot(mean(dat,2),'r','Linewidth',2)
        hold on

        y_upper = mean(dat,2)+std(dat,[],2)./sqrt(nCells(a)-1);
        y_lower = mean(dat,2)-std(dat,[],2)./sqrt(nCells(a)-1);
        fill([1:size(dat, 1), fliplr(1:size(dat, 1))], [y_upper; flipud(y_lower)], 'r', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
        xline(60)
        yline(0)
        xticks(linspace(1,120,7))
        xticklabels([-3 -2 -1 0 1 2 3])
        if d==max(subsets)
            xlabel('Seconds from stim')
        end
        
        if d==1
            title(painvalence.modeTypes{m})
        end
        
        if m==1
            if d==1
                ylabel('All cells')
            else
                ylabel(strcat(painvalence.modeTypes{floor(d/2)},'-',dir_string,' cells'))
            end
        end
        ylim([-2 10])
        
            
        %collect pooled statistics
        aucMs(m,d,1:3) = [mean(collected(61:80,:),'all') std(mean(collected(61:80,:))) size(collected,2)];
        aucMs(m,d,4:6) = [mean(collected(81:100,:),'all') std(mean(collected(81:100,:))) size(collected,2)];
        aucMs(m,d,7:9) = [mean(collected(101:120,:),'all') std(mean(collected(101:120,:))) size(collected,2)];
        toc
        
    end
end
%% Section 3: identify significantly enhanced or inhibited cells

%OUTCOMES:
% 1. responsive: cell containing all significantly activated and suppressed
% neurons (permutation test, alpha<0.01) for each stimulus type and animal
% 2. fracCells: fraction of neurons in each animal activated or suppressed
% to each stimulus
% 3. figure: bar plot of fractions of activated/suppressed neurons


for a=1:nAnimals
    tic
    for m=1:nStim
        
        %for pain panel, cut broken recordings
        if ismember(a,[12 13]) & m==5
            continue
        end
        
        data = psths{a,m};
        
        bls = data(1:window,:,:);
        dels = data(window+21:2*window,:,:);
        
        %pool all baseline periods and evoked response periods for trials
        %of a given stimulus
        bl=[];
        del = [];
        for t = 1:size(bls,3)
            bl = [bl; bls(:,:,t)];
            del = [del; dels(:,:,t)];
        end
        
        %run permutation test (alpha = 0.01)
        changedCells = permTest([bl; del], [1:size(bl,1)], [size(bl,1)+1:size(bl,1)+size(del,1)],100)
       
        %save activated (1) and inhibited (2) cells
        responsive{a,m,1} = changedCells{1};
        responsive{a,m,2} = changedCells{2};
        
        %cell fraction
        fracCells(a,1,m) = length(responsive{a,m,1})/nCells(a,1);
        fracCells(a,2,m) = length(responsive{a,m,2})/nCells(a,1);
    end
    toc
end

titles = painvalence.modeTypes;
figure

for m=1:nStim
    subplot(1,nStim,m)
    barWithError(fracCells(:,:,m))
    ylim([0 .6])
    xticklabels({'Up','Down'})
    ylabel('Fraction of cells')
    title(titles{m})
end

%% Section 4: make heatmaps of neurons.

%OUTCOMES:
% figure: heatmaps of psths for all cells and stim-responsive cells for 
% each stimulus, sorted

for d=1:3
    for s=1:nStim
        figure
        for m=1:nStim
            collected=[];
            for a=1:nAnimals
                if ismember(a,[12 13]) & m==5
                    continue
                end
                if d==1
                    cellset=1:nCells(a);
                    ylimit = [-1 3];
                    
                elseif d==2
                    cellset = responsive{a,s,1};
                    ylimit = [-2 10];
                    dir_string = 'Up';
                else
                    cellset = responsive{a,s,2};
                    ylimit = [-2 1];
                    dir_string = 'Down';
                end
                
                collected = [collected median(psths{a,m}(:,cellset,:),3)];
            end
            %      collected(collected>30)=30;
            %      collected(collected<-30)=-30;
            
            [B,I]  = sort(mean(collected(61:120,:)));
            Is{a,m,d,s}=I;
            subplot(1,nStim,m)
            dat=collected;
            heatmap(dat(:,Is{a,s,d,s})','Colorlimits',[-10 10])
            colormap(palette('scheme',6));
            grid off
            
            if m==1
                if d>1
                    ylabel(strcat('z-score - ',painvalence.modeTypes{floor(d/2)},'-',dir_string,' cells'))
                else
                    ylabel('z-score all cells')
                end
            end
            
            xlabel('Time from onset (s)')
            
            title(painvalence.modeTypes{m})
            
        end
        
    end
end

%% Section 5: calculate stimulus overlaps.

%OUTCOMES:
% figure: fraction overlap between activated/inhibited neurons for each
% stim type, averaged over animals

for a=1:nAnimals
    for d=1:2
        for c=1:2
            for m=1:3
                for s=1:3
                    overlaps(a,m,s,d,c) = sum(ismember(responsive{a,m,d},responsive{a,s,c}))./length(unique([responsive{a,m} responsive{a,s}])); %mean(ismember(ups{a,m},ups{a,s}));%
                end
            end
        end
    end
end

titles = {'Activated Overlap', 'Activted x Inhibited Overlap', 'Inhibited x Activted Overlap', 'Inhibited Overlap'};
figure
for d=1:2
    for c=1:2
        subplot(2,2,2*(d-1)+c)
        heatmap(squeeze(mean(overlaps(:,:,:,d,c))),'colorlimits',[0 1],'XData',painvalence.modeTypes,'YData',painvalence.modeTypes)
        colormap(palette('scheme',4))
        title(titles{2*(d-1)+c})
    end
end

%% Section 6: decode stimuli

%OUTCOMES
% 1. accuracy: normalized confusion matrices for stimulus decoding for each
% animal, crossvalidation, and real vs. shuffled
% 2. aurocsDisc: auROCs of decoders
% 3. figure: normalized confusion matrices averaged over animals

nAnimals=4;
for a=1:nAnimals
    for s=1:2 %shuffle condition
        tic
        
        
        
        if ismember(a,[12 13]) %exclude if running painPanel due to broken recordings during hot water
            continue
        end
        
        toDecode =[];
        classes=[];
        %concatenate data over stimuli
        for m=1:3
            d1 = psths{a,m}(:,:,:);
            s1 = zeros(40*size(d1,3),size(d1,2));
            for t=1:size(d1,3)
                s1(40*(t-1)+1:40*t,:) = d1(81:120,:,t);
            end
            toDecode = [toDecode; s1];
            classes = [classes; m.*ones(size(s1,1),1)];
        end
        
        %PCA normalized data and take PCs to 80% var explained
        [co,sc,~,~,ex] = pca(sqrt(toDecode-min(toDecode)));
        f=cumsum(ex);
        f=find(f>80);
        f=f(1);
        
        for v=1:1000 %crossvals
            if s==2 %shuffle if perm test
                shuff=randperm(length(classes));
                classes=classes(shuff);
            end
            
            %train on 80% test on 20
            [X,Y,x,y] = split_data(sc(:,1:f),classes,.2); %split_data is a custom function, described below that randomly selects test_size % of your data to test on and 1 - test_size to train on
            
            model = fitcdiscr(x,y); %fit discr fits a linear model to your training data & classes
            [predictedLabels, scores]  = predict(model, X); %predict applies your model to your test data to generate class predictions
            
            %calculate confusion mat and aurocs
            accuracy(a,v,:,:,s) = confusionmat(Y,predictedLabels)./sum(confusionmat(Y,predictedLabels));
            
            [~, ~, ~, AUROC] = perfcurve(Y, scores(:,2), true);
            aurocsDisc(a,s,v) = AUROC;
            
        end
        %count true labels in your test set to normalize confusion
        %matrix

        
        toc
    end
end

titles = {'Confusion matrix real','Confusion matrix shuffled'}
figure
for s=1:2
    subplot(1,2,s)
    heatmap(squeeze(nanmean(nanmean(accuracy(:,:,:,:,s),1),2)),'colorlimits',[0 1],'XData',painvalence.modeTypes,'YData',painvalence.modeTypes)
    colormap(palette('scheme',4))
end
