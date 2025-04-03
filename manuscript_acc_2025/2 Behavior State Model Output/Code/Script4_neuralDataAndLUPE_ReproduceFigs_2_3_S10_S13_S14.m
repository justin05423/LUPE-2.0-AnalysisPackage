
%This code will produce all source data and figures for neural data in
%Figs. 2, 3, S10, S13, and S14. One experimental group (Capsaicin, SNI, or
%Uninjured) can be run at a time.

% Section 1: load data
% Section 2: generate lick-probabilities for each animal
% Section 3: identify behavior probability-encoding principal components
% Section 4: identify positive- and negative- behavior encoding cells. Reproduce Fig. Fig. 2k, S13k,l.
% Section 5: collect firing rates of behavior-encoding cells. Reproduce Fig. 2g, i, j; Fig. 3k.
% Section 6: collect behavior-evoked activity and selectivity of behavior-encoding cells. Reproduce Fig. 2l,m; 3g,h,i,j; S10f,l; S13i,j; S14
% Section 7: visualize behavior-evoked activity (heatmaps). Reproduce Fig. S10e
% Section 8: Fisher decoder of behaviors. Reproduce Fig. 2e, S10g,h, S13a-d
% Section 9: Fisher decoder of states. Reproduce Fig. 2e, S10g,h, S13a-d
% Section 10: representative image in Fig. 2c

%Sophie A. Rogers, Corder Lab, University of Pennsylvania, March 23, 2025

%% SECTION 1: load data
%for Figs. 2 and S10, set the experiment to 'Capsaicin'. for Figs. 3 and
%S11, set the experiment to 'SNI' or 'Uninjured' to load session parameters.

experiment = 'Capsaicin';

if experiment == 'Capsaicin'
    %number of animals, sessions, behaviors, and groups
    nAnimals = 5;
    nSesh = 4;
    nBeh = 5;
    nGroups = 1;
    
    
    %group members
    groups = {[1:nAnimals]};
    
    %color schemes
    gCols = {'k'};
    behCols  = {'r',[1.0000    0.4980    0.3137],'y','g','c','b'};
    sessCols = {'k','r','b',[102 51 153]./255};
    
    behaviors = {'Still','Walking','Rearing','Grooming','Left lick','Right lick'};
    sessions = {'Baseline', 'Capsaicin', 'Morphine','Capsaicin+Morphine'};
    
    %data
    load activitiesCap.mat %REPLACE WITH FILE PATH TO FILE WITH PREPROCESSED CALCIUM DATA (output of Script3)
    load behaviorCap.mat %REPLACE WITH FILE PATH TO FILE WITH PREPROCESSED LUPE OUTPUT BEHAVIOR DATA (output of Script3)
    load stateCap.mat %REPLACE WITH FILE PATH TO THIS FILE WITH STATE DATA (output of Script1)
    load psthStoreCap.mat %REPLACE WITH FILE PATH TO FILE WITH PREPROCESSED CALCIUM DATA (output of Script3)
    
elseif experiment == 'SNI'
    
    %number of animals, sessions, and behaviors
    nAnimals = 9;
    nSesh = 6;
    nBeh = 5;
    
    %color schemes
    gCols = {'r','k'};
    behCols  = {'r',[1.0000    0.4980    0.3137],'y','g','c','b'};
    sessCols = {'k',[202 39 34]./255, 'r',[255,69,0]./255,[255,165,0]./255,'b'};
    
    behaviors = {'Still','Walking','Rearing','Grooming','Left lick','Right lick'};
    sessions = {'Baseline', '1 hour', '1 day','3 days','1week','2weeks','3weeks','Morphine'};
    
    %data
    load activities_SNIscope.mat %REPLACE WITH FILE PATH TO FILE WITH PREPROCESSED CALCIUM DATA (output of Script3)
    load behMats_SNIscope.mat %REPLACE WITH FILE PATH TO FILE WITH PREPROCESSED LUPE OUTPUT BEHAVIOR DATA (output of Script3)
    load states_SNIscope.mat %REPLACE WITH FILE PATH TO THIS FILE WITH STATE DATA (output of Script1)
    load psthStoreSNI.mat %REPLACE WITH FILE PATH TO FILE WITH PREPROCESSED CALCIUM DATA (output of Script3)
    %sessions of interest and animals of interest
    sOI = [1 3 5 6 7 8];
    sni = [1 2 3 4 14 15 16 17 18];

    activities = activities(sOI,sni);
    behMats = behMats(sOI,:,sni);
    behState = behState(sOI,sni);
    
elseif experiment == 'Uninjured'
    
    %number of animals, sessions, and behaviors
    nAnimals = 9;
    nSesh = 6;
    nBeh = 5;
    
    %color schemes
    gCols = {'r','k'};
    behCols  = {'r',[1.0000    0.4980    0.3137],'y','g','c','b'};
    sessCols = {'k',[202 39 34]./255, 'r',[255,69,0]./255,[255,165,0]./255,'b'};
    
    behaviors = {'Still','Walking','Rearing','Grooming','Left lick','Right lick'};
    sessions = {'Baseline', '1 hour', '1 day','3 days','1week','2weeks','3weeks','Morphine'};
    
    %data
    load activities_SNIscope.mat %REPLACE WITH FILE PATH TO FILE WITH PREPROCESSED CALCIUM DATA (output of Script3)
    load behMats_SNIscope.mat %REPLACE WITH FILE PATH TO FILE WITH PREPROCESSED LUPE OUTPUT BEHAVIOR DATA (output of Script3)
    load states_SNIscope.mat %REPLACE WITH FILE PATH TO THIS FILE WITH STATE DATA (output of Script1)
    load psthStoreCtrl.mat %REPLACE WITH FILE PATH TO FILE WITH PREPROCESSED CALCIUM DATA (output of Script3)
    
    sOI = [1 3 5 6 7 8];
    ctrl = [5 6 7 8 9 10 11 12 13];
    
    activities = activities(sOI,ctrl);
    behMats = behMats(sOI,:,ctrl);
    behState = behState(sOI,ctrl);
else
    disp('Error: Experiment name can be "Capsaicin", "SNI", or "Uninjured".')
end

%% SECTION 2: generate lick-probabilities for each animal

%OUTPUTS:
%   1. behProb: cell of predicted behaviors over time for each animal,
%   session, and behavior
%   2. stateBehModel: cell of glm parameters for each animal, session, and
%   behavior
%   1. sPBauc: tensor of auROCs of glms trained to predict behaviors for
%   each animal, session, and behavior

for a=1:nAnimals
    for m=1:nSesh
        
        %generate predictors of same length: state at time t, behavior at
        %t-1, and behavior at t-2. store in matrix x.
        
        state = behState{m,a};
        
        beh = behMats{m,2,a};
        if length(beh)>length(state)
            beh = beh(1:length(state),:);
        else
            state = state(1:length(beh));
        end
        
        x = [state(3:end)' beh(2:end-1,:) beh(1:end-2,:)];
        
        
        for n=1:nBeh
            
            disp(strcat('Animal ', num2str(a), ', Session ', num2str(m), ', Behavior ', num2str(n)))
            
            %generate binomial response variable y, occurrence of behavior n
            %truncate to length of predictor x
            bdat = behMats{m,1,a}(:,n);
            y=bdat(3:end,:);
            if length(x) < length(bdat)
                y = y(1:length(x),:);
            else
                x = x(1:length(y),:);
            end
            
            %bootstrap dataset to balance positive and negative samples
            if length(find(y==0))>length(find(y==1))
                resampIdx = find(y==0);
                shuff = randperm(length(resampIdx));
                offTarget = resampIdx(shuff(1:sum(y==1)));
                target = find(y==1);
            else
                resampIdx = find(y==1);
                shuff = randperm(length(resampIdx));
                target = resampIdx(shuff(1:sum(y==0)));
                offTarget = find(y==0);
            end
            
            
            if length(target)<1
                stateBehModel{a,m,n} = [];
                behProb{a,m,n} = [];
                sPBauc(a,m,n) = nan;
                disp('Skipping, no samples of behavior')
            else
                %train binomial glm on bootstrapped data
                mdl = fitglm(x([target; offTarget],:),y([target; offTarget]'),'Distribution','binomial','link','logit');
                
                %predict behavior at time t from whole dataset, in temporal
                %order
                behaviorProbabilityOverTime = glmval(mdl.Coefficients.Estimate,x,'logit');
                
                %evaluate model efficacy
                [X,Y,T,auROC] = perfcurve(y,behaviorProbabilityOverTime,1, 'XCrit','FPR','YCrit','TPR');
                
                %store model, behavior probability over time, and auROC
                stateBehModel{a,m,n} = mdl;
                behProb{a,m,n} = behaviorProbabilityOverTime;
                sPBauc(a,m,n) = auROC;
            end
        end
    end
end

%% SECTION 3: identify behavior probability-encoding principal components

%Outputs:
%   1. predCoeffs: cell of weights of PCs predicting behavior probability
%   for each animal and session
%   2. predPs: cell of p-values of PCs predicting behavior probability
%   for each animal and session
%   3. aurocs: tensor of auROCs of glms trained to predict behavior
%   probability for each animal, session, behavior, and shuffle validation

%set s = 1 for permutation test
shuffle = 0;

for a=1:nAnimals
    tic
    for m=1:nSesh
        if experiment ~= "Capsaicin"
            activities{m,a} = zscore(activities{m,a}); %the shared data was not z-scored for the sni/uninjured experiment
        end
        %PCA square-root transformed data (reduces sensitivity to noise in PCA)
        [co,sc,r,f,ex] = pca(sqrt(activities{m,a}-min(activities{m,a})));
        
        %store coefficients of neurons in each PC
        coeffs{m,a} = co;
        
        %generate predictors: PCs explaining 80% of variance
        d = cumsum(ex);
        f = find(d>80);
        nDims = f(1);
        
        if experiment=="Capsaicin"
            nDims=9;
        end
        
        %initialize cells to store magnitude and sig of coeffs of each predictor
        predCoeffs{m,a} = zeros(nDims,nBeh);
        predPs{m,a} = zeros(nDims,nBeh);
        
        for n=1:nBeh
            disp([a m n])
            
            %predictor matrix (neural PCs)
            x = sc(:,1:nDims);
            
            %response variable (behavior probability)
            y = behProb{a,m,n};
            
            %binarize response variable
            y(y>=.5) = 1;
            y(y<1) = 0;
            
            %render same length
            if length(y)<length(x)
                x=x(1:length(y),:);
            else
                y=y(1:length(x),:);
            end
            
            if isempty(y)
                disp('No samples')
                continue
            elseif sum(ismember([0 1],y))~=2
                disp('No samples')
                continue
            else
                
                %bootstrap unbalanced data
                if length(find(y==0))>length(find(y==1))
                    resampIdx = find(y==0);
                    shuff = randperm(length(resampIdx));
                    offTarget = resampIdx(shuff(1:sum(y==1)));
                    target = find(y==1);
                else
                    resampIdx = find(y==1);
                    shuff = randperm(length(resampIdx));
                    target = resampIdx(shuff(1:sum(y==0)));
                    offTarget = find(y==0);
                end
                y = y([target; offTarget]);
                x = x([target; offTarget],:);
                
                if shuffle==0
                    %train binomial glm to predict behavior probability from PCs
                    mdl = fitglm(x,y,'Distribution','binomial','link','logit');
                    pred = glmval(mdl.Coefficients.Estimate,x,'logit');
                    [X,Y,T,auROC] = perfcurve(y,pred,1, 'XCrit','FPR','YCrit','TPR');
                    
                    %save coefficients, p-values, and aurocs
                    predCoeffs{m,a}(:,n) = mdl.Coefficients.Estimate(2:end);
                    predPs{m,a}(:,n) = mdl.Coefficients.pValue(2:end);
                    aurocs(a,m,n,1) = auROC;
                    
                else
                    for s=1:100
                        %shuffle class labels if running permutation test
                        %and record aurocs
                        y = y(randperm(length(y)),:);
                        mdl = fitglm(x,y,'Distribution','binomial','link','logit');
                        pred = glmval(mdl.Coefficients.Estimate,x,'logit');
                        [X,Y,T,auROC] = perfcurve(y,pred,1, 'XCrit','FPR','YCrit','TPR');
                        
                        aurocs(a,m,n,s+1) = auROC;
                    end
                end
            end
        end
    end
    toc
end

%% SECTION 4: identify positive- and negative- behavior encoding cells. Reproduce Fig. Fig. 2k, S13k,l.

%OUTPUT:
% 1. encCells: cell containing positive and negative p(behavior)-encoding 
% cells for each animal, session, and behavior
% 2. fracCells: matrix containing fraction of neurons represented by
% p(behavior) encoding neurons of same dimensions as encCells

nBeh=5;
encCells = cell(nAnimals,nSesh,nBeh,2);

for a=1:nAnimals
    
    for m=1:nSesh
        %get the coefficient matrix
        data = coeffs{m,a};
        
        for b=1:5
            
            %find most highly weighted PCs
            if size(predCoeffs{m,a})>=b
                
                [B,I] = sort(abs(predCoeffs{m,a}(:,b)));
                pcs = I(length(I)-3:length(I));

                cosInt = zscore(data(:,pcs));
                pop = [];
                for p = 1:length(pcs)
                    %take neurons with z-scores greater than or equal to |2 or -2|
                    pop = [pop; find(abs(cosInt(:,p)>1.2))];%1
                end


                encodingCells{a,m,b} = unique(pop);
                
            end
        end
    end
end


%classify neurons as positive or negative
for a=1:nAnimals
    for m=1:nSesh
        for n=1:nBeh
            pop = encodingCells{a,m,n};
            data = psthStore{a,m,n};
            
            %z-score psths to baseline
            baseline = data(:,1:40,:);% 
            data = (data - mean(baseline,2)) ./ std(baseline,[],2);
            data(data>10) = 10;
            data(data<-10) = -10;
            data(isnan(data))=0;
            
            %average over bouts
            if size(data,1)>1
                data = squeeze(mean(data,1));
            elseif data==1
                data = squeeze(data);
            else
                continue
            end
            
            %if a cell is more active following behavior onset, it is
            %positive. otherwise negative.
            for p = 1:length(pop)
                if mean(data(41:50,pop(p))) > mean(data(1:40,pop(p)))
                    encCells{a,m,n,1} = [encCells{a,m,n,1} pop(p)];
                else
                    encCells{a,m,n,2} = [encCells{a,m,n,2} pop(p)];
                end
            end
            
            fracCells(a,m,n,1) = length(encCells{a,m,n,1})./size(activities{m,a},2);
            fracCells(a,m,n,2) = length(encCells{a,m,n,2})./size(activities{m,a},2);
        end
    end
end
%% SECTION 5: collect firing rates of behavior-encoding cells. Reproduce Fig. 2g, i, j; Fig. 3k.

%OUTPUT:
% 1. calciumRate: cell containing average calcium transient rate in all
% neurons, positive p(lick) neurons, or negative p(lick) neurons in all
% states, Pain State-4, or all but Pain State 4
% 2. summEventRates: summary matrix containing pooled mean, std, and sample
% size of event rates
% 3. figure: bar plots with SEM of event rates in all neurons, positive
% p(lick) neurons, or negative p(lick) neurons in all states, Pain State-4, 
% or all but Pain State 4

for a=1:nAnimals
    tic
    for m=1:nSesh
        for c=1:3
            if c==1
                sOI = [1:6];
            elseif c==2
                sOI = 4;
            else
                sOI = [1 2];
            end
            %take activities during states of interest
            ind=find(ismember(behState{m,a},sOI));
            act = activities{m,a};
            ind(ind>length(act)) = [];
            act=act(ind,:);
            
            cR = zeros(1,size(act,2));
            if ~isempty(act)

                %find peaks that are >= 1 z-score bigger than adjacent
                %troughs
                for n=1:size(act,2)
                    [pks,locs] = findpeaks(act(:,n),'MinPeakProminence',1);
                    cR(1,n) = length(locs)/(length(act)./10);
                end
            end
            %store single cell firing rates
            calciumRate{a,m,c,1} = cR;
            calciumRate{a,m,c,2} = cR(:,encCells{a,m,5,1});
            calciumRate{a,m,c,3} = cR(:,encCells{a,m,5,2});
        end
    end
    toc
end

%Gather pooled mean, STD, and sample size for all populations, conditions
for c=1:3
    for m=1:nSesh
        evRatesAll = [];
        evRatesLickPos = [];
        evRatesLickNeg = [];
        for a=1:nAnimals
            evRatesAll = [evRatesAll calciumRate{a,m,c,1}];
            evRatesLickPos = [evRatesLickPos calciumRate{a,m,c,2}];
            evRatesLickNeg = [evRatesLickNeg calciumRate{a,m,c,3}];
        end
        summEventRates(c,m,1,:) = [nanmean(evRatesAll) nanstd(evRatesAll) length(evRatesAll)];
        summEventRates(c,m,2,:) = [nanmean(evRatesLickPos) nanstd(evRatesLickPos) length(evRatesLickPos)];
        summEventRates(c,m,3,:) = [nanmean(evRatesLickNeg) nanstd(evRatesLickNeg) length(evRatesLickNeg)];
    end
end


%plot
popNames = {'All neurons', 'Pos p(lick)', 'Neg p(lick)'};
condNames = {'All states', 'Pain State-4', 'Non-pain states'};

figure
for p=1:3
    for c=1:3
        subplot(3,3,3*(p-1)+c)
        bar(squeeze(summEventRates(c,:,p,1)))
        hold on
        errorbar(squeeze(summEventRates(c,:,p,1)),squeeze(summEventRates(c,:,p,2))./sqrt(squeeze(summEventRates(c,:,p,3))))
        if c==1
            ylabel(strcat('Events/sec (',popNames{p},')'))
        end
        if p==1
            title(strcat('Activity in ', condNames{c}))
        end
        if p==3
            xticklabels(sessions)
            xtickangle(45)
        end
    end
end
%% SECTION 6: collect behavior-evoked activity and selectivity of behavior-encoding cells. Reproduce Fig. 2l,m; 3g,h,i,j; S10f,l; S13i,j; S14
%OUTPUTS:
% 1. encCellActs: AUCs during 0-1 and 1-2s post-onset of positive- and
% negative-behavior encoding cells at each behavior. in order, dimensions
% are- animal, session, behavior encoded by cell, behavior of interest,
% direction of cell at its own behavior, and time window
% 2. dprimes: cell of dprimes of each beahvior encoding cell (dimension n)
% over each behavior (dimension l) for each animal, session, and cell 
% population/direction. when n==l, d' will be 0. 
% 3. summEvokedActs: summary matrix containing pooled mean, std, and sample 
% size of AUC behavior-evoked activity in every animal, session, and cell
% pop during 0-1 and 1-2s following preferred behavior
% 4. summSelectivity: summary matrix containing pooled mean, std, and sample 
% size of d' behavior-evoked activity in every animal, session, and cell 
% pop
% 5. figure: bar plot with SEM for AUC behavior-evoked activity during time
% window of interest for each p(behavior) population at preferred behavior
% 6. figure: bar plot with SEM for d' of lick-evoked activity during time
% window of interest for each p(lick) population over each behavior
% 7. figure: average psths of each p(behavior) population at preferred 
% behavior

encCellActs = {nAnimals,nSesh,nBeh,nBeh,2,2};
w=40;
for a=1:nAnimals
    for m=1:nSesh
        for l=1:nBeh
            data = (psthStore{a,m,l}-mean(psthStore{a,m,l}(:,1:40,:),2))./std(psthStore{a,m,l}(:,1:40,:),[],2);
            data(isnan(data)) = 0;
            data(data>10)=10;
            data(data<-10) = 10;
            if size(data,1)==0
                continue
            elseif size(data,1)==1
                data = squeeze(data);
            else
                data = squeeze(mean(data));
            end
            
            for d=1:2
                for n=1:nBeh
                    pop = encCells{a,m,n,d};

                    act1 = nanmean(data(w+1:3*w/2,pop));
                    act2 = nanmean(data(3*w/2+1:2*w,pop));
                    act3 = nanstd(data(w+1:3*w/2,pop));

                    encCellActs{a,m,n,l,d,1} = act1;
                    encCellActs{a,m,n,l,d,2} = act2;
                    encCellActs{a,m,n,l,d,3} = act3;
                   
                    
                end
            end
        end
    end
end

%calculate d'
for a=1:nAnimals
    for m=1:nSesh
        for n=1:nBeh
            for l=1:nBeh
                for d=1:2
                    if ~isempty(encCellActs{a,m,n,n,d,1}) & ~isempty(encCellActs{a,m,n,l,d,1})

                        dprimes{a,m,n,l,d} = [(encCellActs{a,m,n,n,d,1}-encCellActs{a,m,n,l,d,1})./encCellActs{a,m,n,l,d,3}];
                    end

                end
            end
        end
    end
end

%Gather pooled mean, STD, and sample size for all populations in their own
%behavior

for n=1:nBeh
    for m=1:nSesh
        for d=1:2
            for t=1:2
                acts = [];
                for a=1:nAnimals
                    acts = [acts encCellActs{a,m,n,n,d,t}];
                end
                summEvokedActs(n,m,d,t,:) = [nanmean(acts) nanstd(acts) length(acts)];
            end
            
            for l=1:nBeh
                acts = [];
                for a=1:nAnimals
                    acts = [acts dprimes{a,m,n,l,d}];
                end
                summSelectivity(n,m,d,l,:) = [nanmean(acts) nanstd(acts) length(acts)];
            end
        end
    end
end

%plot AUC behavior-locked activity of p(behavior) neurons
directions = {'Pos','Neg'};

time = 1;
figure
for n=1:nBeh
    for d=1:2
        subplot(nBeh,2,2*(n-1)+d)
        bar(squeeze(summEvokedActs(n,:,d,time,1)))
        hold on
        errorbar(squeeze(summEvokedActs(n,:,d,time,1)),squeeze(summEvokedActs(n,:,d,time,2))./sqrt(squeeze(summEvokedActs(n,:,d,time,3))))
        if d==1
            ylabel(strcat('AUC (',behaviors{n},')'))
        end
        if n==1
            title(directions{d})
        end
        if n==nBeh
            xticklabels(sessions)
            xtickangle(45)
        end
    end
end

%plot selectivity of P(lick) neurons over each other behavior
behaviorOfInterest = 5; 
figure
for n=1:nBeh
    for d=1:2
        subplot(nBeh,2,2*(n-1)+d)
        bar(squeeze(summSelectivity(behaviorOfInterest,:,d,n,1)))
        hold on
        errorbar(squeeze(summSelectivity(behaviorOfInterest,:,d,n,1)),squeeze(summSelectivity(behaviorOfInterest,:,d,n,2))./sqrt(squeeze(summSelectivity(behaviorOfInterest,:,d,n,3))))
        if d==1
            ylabel(strcat('Selectivity over (',behaviors{n},')'))
        end
        if n==1
            title(directions{d})
        end
        if n==nBeh
            xticklabels(sessions)
            xtickangle(45)
        end
    end
end

sessionLegends = {};
for m=1:nSesh
    sessionLegends{4*(m-1)+1} = sessions{m};
    sessionLegends{4*(m-1)+2} = '';
    sessionLegends{4*(m-1)+3} = '';
    sessionLegends{4*(m-1)+4} = '';
end

%plot average pbths of p(behavior) neurons
figure
for n=1:nBeh
    for d=1:2
        subplot(nBeh,2,2*(n-1)+d)
        for m=1:nSesh
            data = [];
            for a=1:nAnimals
                
                psth = psthStore{a,m,n}(:,:,encCells{a,m,n,d});
                psth = (psth-mean(psth(:,1:40,:),2))./std(psth(:,1:40,:),[],2);
                psth(psth>10)=10;
                psth(psth<-10)=10;
                psth(isnan(psth))=0;
                
                if size(psth,1)==1
                    psth=squeeze(psth);
                elseif size(psth,1)>1
                    psth = squeeze(mean(psth));
                else
                    continue
                end
                
                data = [data psth];
            end
            
            win=[1:size(data,1)];
            hold on
            plot(nanmean(data(win,:),2),'color',sessCols{m},'Linewidth',1)
            hold on
            y_upper = nanmean(data(win,:),2)+nanstd(data(win,:),[],2)./(sqrt(size(data(win,:),2)-1));
            y_lower = nanmean(data(win,:),2)-nanstd(data(win,:),[],2)./(sqrt(size(data(win,:),2)-1));
            fill([1:size(data(win,:), 1), fliplr(1:size(data(win,:), 1))], [y_upper; flipud(y_lower)], sessCols{m}, 'FaceAlpha', 0.2, 'EdgeColor', 'none');
            hold on
            if d==1
            ylabel(strcat(behaviors{n},'-locked activity (z-score)'))
            end
            if n==1
                title(strcat(directions{d},' p(behavior) cells'))
            end
            hold on
            xline(40,'--')
            hold on
            yline(0,'--')
            xticklabels(linspace(-40,40,5))
            if n==nBeh & d==2
                legend(sessionLegends)
            end
        end
    end
end

%% Section 7: visualize behavior-evoked activity (heatmaps). Reproduce Fig. S10e

%INSTRUCTIONS: Run once as is, then run again after uncommenting figure code
%OUTPUT:
% 1. figure: heatmap of p(lick) neurons sorted to lick activity in each
% session over each behavior. sessions in columns, behavior in rows

figure
for m=1:nSesh
    tic
    for n=1:nBeh
        subplot(nBeh,nSesh,nSesh*(n-1)+m)
        data=[];
        for a=1:nAnimals
            
            dat = psthStore{a,m,n}(:,:,encodingCells{a,m,5});
            if size(dat,1) == 1
                dat=squeeze(dat);
            elseif size(dat,1) == 0
                continue
            else
                dat=squeeze(nanmean(dat));
            end
            
            data = [data (dat-nanmean(dat(1:40,:)))./nanstd(dat(1:40,:))];

        end
        
        [B,I] = sort(nanmean(data(41:50,:)));
        
        if n==5
           Is{m,n} = I;
        end
        
        heatmap(data(:,Is{m,5})','colorlimits',[-5 5]);
        
        colormap(palette('scheme',6))
        grid off
    end
    toc
end
    


%% SECTION 8: Fisher decoder of behaviors. Reproduce Fig. 2e, S10g,h, S13a-d

%OUTPUT
% 1. conmat: normalized confusion matrix of for beahvior decoding from 
% principal components, stored for each animal, session, validation, and
% shuffle condition
% 2. aurocsDisc: auROC of each decoder (animal, session, shuffle condition,
% validation)
% 3. figure: average confusion matrices for each session

for a=1:nAnimals
    tic
    for m=1:nSesh
        disp([a m])
        
        %PCA & normalize activities
        [co,sc,r,f,ex] = pca(sqrt(activities{m,a}-min(activities{m,a})));
        
        d = cumsum(ex);
        f = find(d>80);
        nDims = f(1);
        
        z = sc(:,1:nDims);
        
        
        b = behMats{m,2,a}+1;
        
        %align behavior and neural data
        if length(z)>length(b)
            z=z(1:length(b),:);
        else
            b=b(1:length(z));
        end
        
        %get rid of undersampled labels for right lick
        z(b==6,:) = [];
        b(b==6,:) = [];
        
        
        %loop for real vs. permutation tests
        for s=1:2
            for v=1:50 %this has been cut to 50 for time running script but was really 1000x cross-val
                
                %split_data is a custom function, described below that randomly selects test_size % of your data to test on and 1 - test_size to train on
                [X,Y,x,y] = split_data(z,b,.2); 
        
                %split randomly until all classes are sampled in test and training
                %data
                while length(unique(y))<length(unique(b))
                    [X,Y,x,y] = split_data(z,b,.2);
                end
                
                %shuffle if s==2 for permutation test
                if s==2
                    shuff = randperm(length(y));
                    y=y(shuff);
                end
                
                %bootstrap to balance dataset
                nSamps = [length(find(y==1)) length(find(y==2)) length(find(y==3)) length(find(y==4)) length(find(y==5))];
                nSamps(nSamps==0) = [];
                nSamps = min(nSamps);
                
                inds = [];
                for t=1:5
                    if ismember(t,y)
                        idx=find(y==t);
                        inds = [inds; reshape(idx(1:nSamps),nSamps,1)];
                    end
                end
                
                x=x(inds,:);
                y=y(inds)';
                
                
                %train model
                model = fitcdiscr(x,y); %fit discr fits a linear model to your training data & classes
                %test model
                [predictedLabels,scores]  = predict(model, X); %predict applies your model to your test data to generate class predictions
                
                %fill normalized confusion matrix
                cms = zeros(5,5);
                cm = confusionmat(Y,predictedLabels);
                statesPresent = unique(Y);
                
                for g=1:length(statesPresent)
                    for h=1:length(statesPresent)
                        cms(statesPresent(g),statesPresent(h)) = cm(g,h)./sum(cm(g,:));
                    end
                end
                
                %save confusion matrix for every animal, session, condition, and iteration
                conmat(a,m,:,:,s,v) = cms;
                
                % Compute AUROC
                [~, ~, ~, AUROC] = perfcurve(Y, scores(:,2), true);
                aurocsDisc(a,m,s,v) = AUROC;
                
            end
        end
    end
    toc
end


conmat(conmat==0) = nan;

for s=1:2
    figure
    for m=1:nSesh
        subplot(round(nSesh/2),round(nSesh/2),m)
        heatmap(squeeze(nanmean(nanmean(conmat(:,m,:,:,s,:),6))),'Colorlimits',[0 1],'XData',{'Still', 'Walk', 'Rear', 'Groom', 'Lick'},'YData',{'Still', 'Walk', 'Rear', 'Groom', 'Lick'})
        colormap(palette('scheme',4))
        if t>3
            xlabel('Predicted behavior')
        end
        if s==1
            title(sessions{m})
        end
        if t==1
            ylabel('True behavior')
        end
    end
end

%% SECTION 9: SVM decoder of states (Fig. S10g,h, S13a-d)

%OUTPUT
% 1. conmatState: normalized confusion matrix of for state decoding from 
% principal components, stored for each animal, session, validation, and
% shuffle condition
% 2. aurocsDiscS: auROC of each decoder (animal, session, shuffle condition,
% validation)
% 3. figure: average confusion matrices for each session


for a=1:nAnimals
    tic
    for m=1:nSesh
        disp([a m])
        
        if length([encCells{a,m,5,1} encCells{a,m,5,2}]) == 0
            continue
        else
            
            %predict pain vs. non-pain states from activities
            z = activities{m,a}(:,[encCells{a,m,5,1} encCells{a,m,5,2}]);
            
            b = behState{m,a,1};
            
            
            if length(z)>length(b)
                z=z(1:length(b),:);
            else
                b=b(1:length(z));
            end
            b(b<3) = 1;
            b(b>2) = 2;
            
            if sum(ismember([1 2],b)) == 2
                [X,Y,x,y] = split_data(z,b,.2); %split_data is a custom function, described below that randomly selects test_size % of your data to test on and 1 - test_size to train on
                while length(unique(y))<length(unique(b))
                    [X,Y,x,y] = split_data(z,b,.2);
                end
                
                for s=1:2
                    for v=1:50
                        if s==2
                            shuff = randperm(length(y));
                            y=y(shuff);
                        end
                        
                        %bootstrap
                        nSamps = [length(find(y==1)) length(find(y==2))];
                        nSamps(nSamps==0) = [];
                        nSamps = min(nSamps);
                        
                        inds = [];
                        for t=1:6
                            if ismember(t,y)
                                idx=find(y==t);
                                inds = [inds; reshape(idx(1:nSamps),nSamps,1)];
                            end
                        end
                        
                        x=x(inds,:);
                        y=y(inds)';
                        
                        %fit model
                        model = fitcsvm(x,y); %fit discr fits a linear model to your training data & classes
                        [predictedLabels, scores]  = predict(model, X); %predict applies your model to your test data to generate class predictions
                        
                        cms = zeros(2,2);
                        cm = confusionmat(Y,predictedLabels);
                        statesPresent = unique(Y);
                        for g=1:length(statesPresent)
                            for h=1:length(statesPresent)
                                cms(statesPresent(g),statesPresent(h)) = cm(g,h)./sum(cm(g,:));
                            end
                        end
                        
                        %take confusion matrix, auROC
                        conmatState(a,m,:,:,s,v) = cms;
                        [~, ~, ~, AUROC] = perfcurve(Y, scores(:,2), true);
                        aurocsDiscS(a,m,s,v) = AUROC;
                    end
                end
            end
        end
    end
    toc
end

conmatState(conmatState==0) = nan;

for s=1:2
    figure
    t=0;
    for m=[2 4]
        t=t+1;
        subplot(1,2,t)
        heatmap(squeeze(nanmean(nanmean(conmatState(:,m,:,:,2,:),6))),'Colorlimits',[0 1],'XData',{'Non-pain state','Pain state'},'YData',{'Non-pain state','Pain state'})
        colormap(palette('scheme',4))
        if t>3
            xlabel('Predicted state')
        end
        if s==1
            title(sessions{m})
        end
        if t==1
            ylabel('True state')
        end
    end
end

%% SECTION 10: Representative image in Fig. 2c

m=2;
a=2;
win = [5001:10000];
dat = activities{m,a}(win,:);
[i,c] = kmeans(dat',5);

dat = dat(:,[find(i==1); find(i==2); find(i==3); find(i==4); find(i==5)]);

dat(isnan(dat))=0;

figure
subplot(212)

for n=1:5
plot(behMats{m,1,a}(win,n)+2*(n-1))
hold on
end
for n=1:6
    l = ismember(behState{m,a}(win),n);
    plot(l+2*6+2*(n-1))
    hold on
end
for n=1:5
    plot(behProb{a,m,n}(win)+2*13+2*(n-1))
    hold on
end
xlim([0 5000])

subplot(211)
heatmap(dat','Colorlimits',[-3 3])
grid off
colormap(palette('scheme',6))
