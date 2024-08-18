%% In Vivo Imaging Analysis Pipeline for James, Oswell, McCall, Hsu et al., 2024
%
% Code here will produce the data for Fig. 2E-Q and all associated
% supplemental figures
%
% For aesthetic purposes, most data was exported from these sections in csv
% and txt files and imported in Prism, where most statistical tests were
% also done
%
%Written by Sophie A. Rogers, Corder Laboratory, University of Pennsylvania


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


oswell.animals(1).long = readtable('MM76_correspondencestable.csv');
oswell.animals(2).long = readtable('MM81_correspondencestable.csv');
oswell.animals(3).long = readtable('MM84_correspondencestable.csv');
% loop through sessions for analysis to tranform data for analysis. will loop through animals later
    %downsample behavior to match frame rate
    %binarize behavior
    %pca transform behavior across all sessions, store by session, and get
    %coefficients of original behaviors for PCA transform
nAnimals = 5;
nSesh = 4; %nSessions
dt=20; %neural data sampling rate
dtB = 60; %behavior sampling rate
pcBeh = cell(nSesh,nAnimals);


behCols  = {'r',[1.0000    0.4980    0.3137],'y','g','c','b'};
sessCols = {'k','r','b',[102 51 153]./255};
pcCols = {'r',[1.0000    0.4980    0.3137],'g','b'};

behaviors = {'Still','Walking','Rearing','Grooming','Left lick','Right lick'};
sessions = {'Baseline', 'Capsaicin', 'Morphine','Capsaicin+Morphine'};

%% Section 2: Clean data 

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
%%
co=pca(activities{2,5});
[B,I] = sort(co(:,1));
figure
heatmap(zscore(co(I,1:20)))
colormap(jet)
ylabel('Neurons')
xlabel('PCs')
title('Neuron coefficients in PCA')
grid off

%% binomial GLM on first 20PCs of ACC activity

% strs = zeros(nAnimals,nSesh,5,11,2);
% fracs = zeros(nAnimals,nSesh,5,11,2);
% aucs = zeros(nAnimals,nSesh,11,5);

for shuff = 0
alpha =  .05/20;%/size(activities,2);
coeffs = cell(4,5);
numDims = 20;

for a=1:5
    tic
    for m=1:4
        %pca zeroed activities
         sc = activities{m,a}-min(activities{m,a});
         [co,sc,l,k,ex] = pca(sc);
         
         %save coefficient matrix (neurons X PCs)
        coeffs{m,a} = co;

        for b=1:5 %loop through behaviors, ignore right-lick
            bdat = behMats{m,1,a}(:,b);
            
            %identify bout starts
            bouts = strfind(bdat',[0 1]);
            
            %skip if no bouts
            if length(bouts)==0
                continue
            else
                adat = sc(:,1:numDims);
                
                %fit binomial
                
                if shuff==1
                    for s=1:10
                        idx = randperm(length(adat));
                        
                        mdl = fitglm(adat(idx,:),bdat,'Distribution','binomial','Link','logit');
                        pred = glmval(mdl.Coefficients.Estimate,adat,'logit');
                        
                        [X,Y,T,AUC] = perfcurve(bdat,pred,1, 'XCrit','FPR','YCrit','TPR');
                        aucs(a,m,b,s+1) = AUC;
                        
                        betas = mdl.Coefficients.Estimate(2:end);
                        ps = mdl.Coefficients.pValue(2:end);
                        betaCell{a,m,b,s+1} = betas;
                        pCell{a,m,b,s+1} = ps;
                        
                        
                        id1 = find(ps<alpha);
                        id2 = find(ps<alpha & betas>0);
                        id3 = find(ps<alpha & betas<0);
                
                        sigBetas = betas(id1);
                        posBetas = betas(id2);
                        negBetas = betas(id3);
                        posBetas(isnan(posBetas)) = 0;
                        negBetas(isnan(negBetas)) = 0;
                        strs(a,m,b,s+1,1) = nanmean(posBetas);
                        strs(a,m,b,s+1,2) = nanmean(negBetas);

                        fracs(a,m,b,s+1,1) = length(posBetas)/length(betas);
                        fracs(a,m,b,s+1,2) = length(negBetas)/length(betas);
                    end
                else
                
                mdl = fitglm(adat,bdat,'Distribution','binomial','Link','logit');
                pred = glmval(mdl.Coefficients.Estimate,adat,'logit');
                %calculate auROC - "pred" is a vector of probabilities of
                %licking, perfcurve thresholds it to convert back to binary
                
                [X,Y,T,AUC] = perfcurve(bdat,pred,1, 'XCrit','FPR','YCrit','TPR');
                aucs(a,m,b,1) = AUC;
                
                %save models &  fits
                models{a,m} = mdl;
                d = devianceTest(mdl);
                ds(a,m) = d.pValue(2);
                
                %save model coefficients of PCs and p values
                betas = mdl.Coefficients.Estimate(2:end);
                ps = mdl.Coefficients.pValue(2:end);
                betaCell{a,m,b,1} = betas;
                pCell{a,m,b,1} = ps;
                
                %save signiticant PCs
                idxs{a,m,b,1} = find(ps<alpha);
                idxs{a,m,b,2} = find(ps<alpha & betas>0);
                idxs{a,m,b,3} = find(ps<alpha & betas<0);
                
                %save strengths and fractions of significant PCs
                sigBetas = betas(idxs{a,m,b,1});
                posBetas = betas(idxs{a,m,b,2});
                negBetas = betas(idxs{a,m,b,3});
                posBetas(isnan(posBetas)) = 0;
                negBetas(isnan(negBetas)) = 0;
                strs(a,m,b,1,1) = mean(posBetas);
                strs(a,m,b,1,2) = mean(negBetas);

                fracs(a,m,b,1,1) = length(posBetas)/length(betas);
                fracs(a,m,b,1,2) = length(negBetas)/length(betas);
                end
            end
        end

    end
    toc
end
end
%% collect highly weighted cells along significant PCs

for a=1:nAnimals
    
    for m=1:nSesh
        %get the coefficient matrix
        data = coeffs{m,a};
        for b=1:5
            for d=1:2
                %get the significant PCs
                pcs = idxs{a,m,b,d};
                %zscore coefficients
                cosInt = zscore(data(:,pcs));
                pop = [];
                for p = 1:length(pcs)
                    %take neurons with z-scores greater than or equal to |2 or -2|
                    pop = [pop; find(abs(cosInt(:,p)>2))];
                end
                
                %store unique neurons
                encodingCells{a,m,b} = unique(pop);

            end
            
        end
    end
end


%% get peri-behavior time histograms


w=40; %number of frames to take before and after behavior
for a=1:nAnimals
    for m=1:nSesh
        %get behavior and activities
        bData = behMats{m,1,a};
        aDat = activities{m,a};
        for b=1:5
            %identify bout starts
            bDat = bData(:,b);
            bouts = strfind(bDat',[0 1]);
            
            %cut bouts that are too close to the ends for our time window
            bouts(bouts<40) = [];
            bouts(bouts>length(bDat)-40) = [];
            
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
            
            %save
            psthStore{a,m,b} = psthMat;
            psthBehStore{a,m,b} = psthBeh;
        end
        
    end
end
%% create behavior PSTHs
figure
for n=1:5
subplot(1,5,n)
for m = 1:4
    psthBeh = [];
    for a = 1:nAnimals
        psthBeh = [psthBeh; psthBehStore{a, m, n}];
    end
    if n==5 & ismember(m,[1 3])
        continue
    else
    hold on
    plot(mean(psthBeh, 1), 'color', sessCols{m}, 'LineWidth', 2)
    hold on
    y_upper = mean(psthBeh, 1) + std(psthBeh, [], 1) ./ sqrt(size(psthBeh, 1));
    y_lower = mean(psthBeh, 1) - std(psthBeh, [], 1) ./ sqrt(size(psthBeh, 1));
    fill([1:size(psthBeh, 2), fliplr(1:size(psthBeh, 2))], [y_upper, fliplr(y_lower)], sessCols{m}, 'FaceAlpha', 0.2, 'EdgeColor', 'none');
    xline(40)
    yline(0)
    xticks(linspace(0, 80, 5))
    xticklabels([-2 -1 0 1 2])
    ylabel(strcat('Average',behaviors{n}))
    xlabel('Time from onset (s)')
    title(behaviors{n})
    
    %save mean, std, nSamples of behaviors to anakyze later if desired
    %(Prism can take, plot, & analyze data in this format)
    %0-1sec
    aucsBeh(m,n,1) = mean(psthBeh(:,w:1.5*w),'all');
    aucsBeh(m,n,2) = std(mean(psthBeh(:,w:1.5*w),2));
    aucsBeh(m,n,3) = size(psthBeh,2);
    
    %1-2sec
    aucsBeh(m,n,4) = mean(psthBeh(:,1.5*w+1:end),'all');
    aucsBeh(m,n,5) = std(mean(psthBeh(:,1.5*w+1:end),2));
    aucsBeh(m,n,6) = size(psthBeh,2);
    % legend(sessions{m})
    end
end
end

%% Create neural PSTHs of behavior-encoding neurons

for n=1:5
    figure
    for m=1:4
        
        psths1 = [];
        psths2 = [];
        for a=1:nAnimals
            %get all the encoding neurons
            pop = [encodingCells{a,m,n,1}];
            
            %get
            psth = psthStore{a,m,n};
            
            %z-score to the second before bout start
            psth = squeeze(mean(psth,1));
            ctrl = psth(w/2+1:w,:);
            psth = [psth-mean(ctrl)]./std(ctrl);
            
            %actually while we're at it, let's get the activities of neurons 
            %encoding behavior N during bouts of behavior L
            savePSTH = zeros(size(psth,2),5);
            for l=1:5
                psthPseudo = squeeze(mean(psthStore{a,m,l},1));
                ctrl = psthPseudo(w/2+1:w,:);
                psthPseudo = [psthPseudo-mean(ctrl)]./std(ctrl);
                savePSTH(:,l,1) = mean(psthPseudo(41:end,:));
                savePSTH(:,l,2) = std(psthPseudo(41:end,:));
            end
            selecMat{a,m,n} = savePSTH;
            
            %separate behavior-on from behavior-off neurons
            counter1 = 0;
            counter2 = 0;
            p1 = [];
            p2 = [];
            for p=1:length(pop)
                %if mean(activity during behavior) > mean(before behavior),
                %these are behavior on
                if mean(psth(41:end,pop(p)))>mean(psth(1:40,pop(p)))
                    %collect psths from lick on neurons to pool across
                    %animals
                    psths1 =  [psths1 psth(:,pop(p))];
                    
                    %count number of lick on neurons
                    counter1=counter1+1;
                    
                    %record neuron index
                    p1 = [p1; pop(p)];
                else %do the same for lick-off neurons
                    psths2 =  [psths2 psth(:,pop(p))];
                    counter2=counter2+1;
                    p2 = [p2; pop(p)];
                end
            end
            %save neuron IDs and the proportions of these neurons out of
            %the total number recorded in that animal and session to
            %analyze later (pie charts in prism)
            encCells{a,m,n,1} = p1;
            encCells{a,m,n,2} = p2;
            ratios(a,m,n,1) = counter1/size(psthStore{a,m,n},3);
            ratios(a,m,n,2) = counter2/size(psthStore{a,m,n},3);
            
            
        end
        
        %save mean, std, and nSamples to analyze in prism. we are now
        %working with pooled neurons
        %pos neurons
        aucis{m,n,1} = mean(psths1(w:1.5*w,:));
        aucis{m,n,2} = mean(psths1(1.5*w+1:end,:));
        aucis{m,n,3} = mean(psths2(w:1.5*w,:));
        aucis{m,n,4} = mean(psths2(1.5*w+1:end,:));
        
        aucsActs(m,n,1) = mean(psths1(w:1.5*w,:),'all');
        aucsActs(m,n,2) = std(mean(psths1(w:1.5*w,:)));
        aucsActs(m,n,3) = size(psths1(w:1.5*w,:),2);
        aucsActs(m,n,4) = mean(psths1(1.5*w+1:end,:),'all');
        aucsActs(m,n,5) = std(mean(psths1(1.5*w+1:end,:)));
        aucsActs(m,n,6) = size(psths1(1.5*w+1:end,:),2);
        %neg neurons
        aucsActs(m,n,7) = mean(psths2(w:1.5*w,:),'all');
        aucsActs(m,n,8) = std(mean(psths2(w:1.5*w,:)));
        aucsActs(m,n,9) = size(psths2(w:1.5*w,:),2);
        aucsActs(m,n,10) = mean(psths2(1.5*w+1:end,:),'all');
        aucsActs(m,n,11) = std(mean(psths2(1.5*w+1:end,:)));
        aucsActs(m,n,12) = size(psths2(1.5*w+1:end,:),2);
        
        %uncomment to save z-scored psths with all neurons
        %csvwrite(strcat(sessions{m},'_',behaviors{n},'_PSTH.csv'),psth);
        
        %skip sessions with no licks if behavior N = lick
        if n==5 & ismember(m,[1 3])
            continue
        else
        
        subplot(121)
        hold on
        plot(mean(psths1,2),'color',sessCols{m},'Linewidth',2)
        hold on
        y_upper = mean(psths1,2)+std(psths1,[],2)./sqrt(length(p1)-1);
        y_lower = mean(psths1,2)-std(psths1,[],2)./sqrt(length(p1)-1);
        fill([1:size(psths1, 1), fliplr(1:size(psths1, 1))], [y_upper; flipud(y_lower)], sessCols{m}, 'FaceAlpha', 0.2, 'EdgeColor', 'none');
        xline(40)
        yline(0)
        xticks(linspace(0,80,5))
        xticklabels([-2 -1 0 1 2])
        ylabel('z-score')
        xlabel('Time from onset (s)')
        title(strcat('positive ',behaviors{n},'-encoding neurons'))
        subplot(122)
        hold on
        plot(mean(psths2,2),'color',sessCols{m},'Linewidth',2)
        hold on
        y_upper = mean(psths2,2)+std(psths2,[],2)./sqrt(length(p2)-1);
        y_lower = mean(psths2,2)-std(psths2,[],2)./sqrt(length(p2)-1);
        fill([1:size(psths2, 1), fliplr(1:size(psths2, 1))], [y_upper; flipud(y_lower)], sessCols{m}, 'FaceAlpha', 0.2, 'EdgeColor', 'none');
        xline(40)
        yline(0)
        ylabel('z-score')
        xticks(linspace(0,80,5))
        xticklabels([-2 -1 0 1 2])
        ylabel('z-score')
        xlabel('Time from onset (s)')
        legend(sessions{m})
        title(strcat('Negative_',behaviors{n},'-encoding neurons'))
       
       end
        %csvwrite(strcat(behaviors{n},'_',sessions{m},'_up.csv'),psths1)
        %csvwrite(strcat(behaviors{n},'_',sessions{m},'_down.csv'),psths2)
    end
end
%% make heatmaps of encoding neurons for pre-set behavior around every behavior in sessions of interest

behOfInt = 1; %set from 1-5 for whatever behavior you want

%%% ALERT: run this once first to get all the sort orders you need, after 
%%% commenting out the figure producing code at the bottom. THEN, uncomment 
%%% figure code and run
for behOfInt = 1:5
figure
for m=[2 4] %loop through sessions you want. if changing number of sessions, you'll have to change number of subplots
    for n=1:5
        if m==2 & behOfInt==5%for some reason the psth of one mouse in one session is improperly transposed. i have been to lazy to fix it, this is my quick fix
            data = [squeeze(mean(psthStore{1,m,n}(:,:,encCells{1,m,behOfInt,1}),1))' squeeze(mean(psthStore{2,m,n}(:,:,encCells{2,m,behOfInt,1}),1)) squeeze(mean(psthStore{3,m,n}(:,:,encCells{3,m,behOfInt,1}),1)) squeeze(mean(psthStore{5,m,n}(:,:,encCells{5,m,behOfInt,1}),1))];
        else
            data= [squeeze(mean(psthStore{1,m,n}(:,:,encCells{1,m,behOfInt,1}),1)) squeeze(mean(psthStore{2,m,n}(:,:,encCells{2,m,behOfInt,1}),1)) squeeze(mean(psthStore{3,m,n}(:,:,encCells{3,m,behOfInt,1}),1)) squeeze(mean(psthStore{5,m,n}(:,:,encCells{5,m,behOfInt,1}),1))];
        end
        
    %get ordered indices of positive and negative encoding cells
    data = [data-mean(data(w/2+1:w,:))]./std(data(w/2+1:w,:));
    [B,I] = sort(mean(data(w+1:end,:)),'descend');
    %Is{m,n,behOfInt,1} = I; %save
    
    data1= [squeeze(mean(psthStore{1,m,n}(:,:,encCells{1,m,behOfInt,2}),1)) squeeze(mean(psthStore{2,m,n}(:,:,encCells{2,m,behOfInt,2}),1)) squeeze(mean(psthStore{3,m,n}(:,:,encCells{3,m,behOfInt,2}),1)) squeeze(mean(psthStore{5,m,n}(:,:,encCells{5,m,behOfInt,2}),1))];
    data1 = [data1-mean(data1(w/2+1:w,:))]./std(data1(w/2+1:w,:));
    [B,I1] = sort(mean(data1(w+1:end,:)),'descend');
    %Is{m,n,behOfInt,2} = I1; %save
    
    %figure producing code
    subplot(2,5,5*(m/2-1)+n)
    heatmap([data(:,Is{m,behOfInt,behOfInt,1}) data1(:,Is{m,behOfInt,behOfInt,2})]','colorlimits',[-5 5])
    colormap(jet)
    if m==2
        title(behaviors{n})
    end
    if n==1
        ylabel(sessions{m})
    end
    grid off
    end
end
end
%% store means z-score of activities to generate behavioral tuning curves
for a=1:5
    for m=1:4
        bDat = behMats{m,1,a};
        aDat = activities{m,a};
        for n=1:5
            for d=1:2
                pop = encCells{a,m,n,d};

                for l = 1:5
                    idx = find(bDat(:,l)==1);
                    idxb = find(bDat(:,l)==0);
                    %take average z-score of activity durign behavior with
                    %respect to all other behaviors over all encoding
                    %neurons
                    acts = mean([aDat(idx,pop)-mean(aDat(idxb,pop))]./std(aDat(idxb,pop)));
                    %take std
                    actsS = std([aDat(idx,pop)-mean(aDat(idxb,pop))]./std(aDat(idxb,pop)));
                    
                    %save
                    storeActs{a,m,n,l,d} = [acts; actsS];
                end
            end
        end
    end
end

%% Calculate d' for a given behavior compared to all others. pool over animals
behOfInt = 5; %set 1-5 for your behavior of interest
for n=1:5
    for m=[2 4] %just capsaicin and morphine bc i was just looking at lick neurons but you can change it
        for d=1:2
            ds = [];
            for a = 1:nAnimals
                data1 = storeActs{a,m,behOfInt,behOfInt,d};
                data2 = storeActs{a,m,behOfInt,n,d};
                ds = [ds abs(data1(1,:)-data2(1,:))./sqrt([std(data1(2,:)) + std(data2(2,:))]./2)];
            end
            %save
            dprimes{n,m,d} =  ds;
        end
    end
    
end

%% 5 fold cross validated fisher decoder
conMat = zeros(5,4,100,5,5,2);
for s=1:2
for a=1:nAnimals
    tic
    for m=1:nSesh
        for v=1:100
            %classes are the NON-BINARIZED vector of behavior labels
            classes = behMats{m,2,a};
            data = activities{m,a};
            lenClasses = sum(behMats{m,1,a});
            %get rid of right lick data (bc only a few frames in a couple
            %mice)
            idx = find(classes==5);
            classes(idx,:) = [];
            data(idx,:) = [];
            
            idxs = [];
            for b=1:sum(lenClasses(1:5)>0)
                subsamp = randperm(lenClasses(b));
                subsamp = subsamp(1:min(lenClasses(lenClasses>0)));
                subsampBeh = find(classes==b-1);
                idxs = [idxs subsampBeh(subsamp)];
            end
            
            classes = classes(idxs,:);
            data = data(idxs,:);
            
            if s==2
                idx = randperm(length(classes));
                classes = classes(idx,:);
            end
            
            %custom function to partition, see my github
            [X,Y,x,y] = split_data(data,classes,.5); %split_data is a custom function, described below that randomly selects test_size % of your data to test on and 1 - test_size to train on
            
            model = fitcdiscr(x,y); %fit discr fits a linear model to your training data & classes
            predictedLabels  = predict(model, X); %predict applies your model to your test data to generate class predictions
            
            %count true labels in your test set to normalize confusion
            %matrix
            normVec = zeros(length(unique(Y)),1);
            for n=1:length(unique(Y))
                normVec(n,1) = sum(Y==n-1);
            end
            
            %different sessions have different numbers of unique behaviors 
            %so save your confusion mat to the appropriate indices here
            conMat(a,m,v,1:length(unique([unique(Y); unique(predictedLabels)])),1:length(unique([unique(Y); unique(predictedLabels)])),s) = confusionmat(Y, predictedLabels)./normVec;

        end
    end
    toc
end
end
%average over crossvalidations and animals. replace nans (bc of division by
%zero in some cases) with zeros
avgConMat = squeeze(nanmean(nanmean(conMat([1 2 3 5],:,:,:,:,1),3),1));
%%
%plot
figure
for n=1:4
subplot(2,2,n)
heatmap(squeeze(avgConMat(n,:,:)),'XData',behaviors(1:5),'YData',behaviors(1:5),'Colorlimits',[0 1])
ylabel('Real')
xlabel('Predicted')
colormap('Parula')
title(sessions{n})

end

acm=squeeze(nanmean(conMat(:,:,:,:,:,:),3));
animals = [1 2 3 5];
for m=1:4
    for b=1:5
        for c=1:5
            for a=1:4
                newMat(b,4*(c-1)+a,m) = acm(animals(a),m,b,c,1);
            end
        end
    end
end


for m=1:4
    for b=1:5
        for c=1:5
            [h,p] = ttest(acm([1 2 3 5],m,b,c,1),acm([1 2 3 5],m,b,c,2));
            ps(m,b,c) = p;
        end
    end
end

figure
for m=1:4
    subplot(2,2,m)
    heatmap(squeeze(ps(m,:,:)),'colorlimits',[0 1],'XData',behaviors(1:5),'YData',behaviors(1:5))
    title(sessions{m})
    colormap(parula)
    
end
%% make the trace plot in panel F

%selected cells of interest
cOI = [190 162 23 30 63  56  97  155  157  158 6 33 134 183 201 ];

figure
for n=1:length(cOI)
    if n<6
        c = 'k';
    elseif n>10
        c = 'r';
    else
        c = 'b';
    end
    plot(activities{2,5}(400:800,cOI(n))+2*(n-1),c)   
    hold on
end
hold on
plot(behMats{2,1,5}(400:800,5)+32,'linewidth',2)
%% plot overlaps of behavior encoding cells

for a=1:5
    for m=1:4
        for n=1:5
            pop1 = encCells{a,m,n,1};
            for l=1:5
                pop2 = encCells{a,m,l,1};
                if length(unique([pop1; pop2])) == 0
                    overlap(a,m,n,l) = 0;
                else
                overlap(a,m,n,l) = sum(ismember(pop2,pop1))/length(unique([pop1; pop2]));
                end
            end
        end
    end
end

overlap(:,:,4,4) = 1;
overlap(:,:,5,5) = 1;
figure
for m=1:4
    subplot(2,2,m)
    if ismember(m,[1 3])
        heatmap(squeeze(median(overlap(:,m,1:4,1:4))),'XData',behaviors(1:4),'YData',behaviors(1:4))
    else
        heatmap(squeeze(median(overlap(:,m,:,:))),'XData',behaviors(1:5),'YData',behaviors(1:5))
    end
    title(sessions{m})
end

colormap('parula')
%% make spatial scatter plots
for n=5%:5
figure
for m=1:4
    for a=1:5
        subplot(5,4,4*(a-1)+m)
        data = table2array(oswell.animals(a).sessions(m).props(:,6:7));
        data = data.*-1;
        scatter(data(:,1),data(:,2),30,'k')
        hold on
        scatter(data(encCells{a,m,n,1},1),data(encCells{a,m,n,1},2),30,'r','filled')
        hold on
        scatter(data(encCells{a,m,n,2},1),data(encCells{a,m,n,2},2),30,'b','filled')
        hold on
        ylim([-200,0])
        xlim([-320,0])
        xticks(linspace(-320,0,6))
        xticklabels(linspace(-320,0,6)./3.125)
        yticks(linspace(-200,0,6))
        yticklabels(linspace(-200,0,6)./3.125)
        if m==4 & a==1
        legend({'Neurons','Positive-encoding','Negative-encoding'})
        end
        if m==1
            ylabel(strcat('Animal_ ',1))
        end
        if n==1
            title(strcat(sessions{m},' _ ',behaviors{n}))
        end
    end
end
end

