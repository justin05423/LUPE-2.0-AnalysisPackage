% This code will produce the within-state behavior dynamics and simulations
% displayed in Figs. 2n,o and S9

% Run dataset of interest through Sections 1 and 4 of script 1 first. Do
% not clear your variables

%SECTION 1: Calculate within-state behavior dynamics and plot cdf. Reproduce Fig. S9c,e; Fig. 2o
%SECTION 2: Plot within-state behavior dynamics as state progresses. Reproduce Fig. 2n, S9b
%SECTION 3: KS test heatmaps. Reproduce Fig. 2o, S9d,f
%SECTION 4: Markov simulation. Reproduce Fig. S9b top
%SECTION 5: CDF and KS test for simulated State 4 behavior. Reproduce Fig. S9c top

%Sophie A. Rogers, Corder Lab, University of Pennsylvania, March 24, 2025

%% SECTION 1: Calculate within-state behavior dynamics and plot cdf. Reproduce Fig. S9c,e; Fig. 2o

%OUTCOMES:
% 1. behaviorResampled: cell containing binary traces of behavior occurance
% over each bout of a given state across all animals, for each behavior and
% each state
% 2. pools: cell containing fraction time remaining in a state from each
% behavior instance detected, for each behavior and state
% 3. figure: cumulative density function for each behavior as a fraction time
% remaining

painUpStates = [3 4 6];
nonPainStates = [1 2 5];

behaviors = {'Still','Walk','Rear','Groom','Lick'};

clear behaviorResampled
clear pools
sessCols = {'r',[253,166,58]./255,[54,128,45]./255,[51,102,255]./255,[76,0,164]./255};
stateCols = {'c', 'b','y',[255,127,80]./255,'r','m'};
sCols = {'r',[76,0,164]./255};
figure
for s=1:8
    for n=1:5
        
        pool = [];
        psthvec=[];
        for a=1:size(behDS,2)
            
            %binarize with respect to given behavior
            d1 = behDS(:,a)+1;
            d2 = behState(:,a,1);
            d1(d1~=n) = 0;
            d1(d1==n) = 1;
            
            %find all instances of behavior
            behIndex = find(d1==1);
            
            %identify state/states of interest
            stateOfInt = s;
            if s==7
                stateOfInt = painUpStates;
            elseif s==8
                stateOfInt = nonPainStates;
            end
            
            %find all instances of state of interest
            d2=ismember(d2,stateOfInt);
            
            %identify bouts of state
            stateBoutsOn = strfind(d2',[0 1]);
            stateBoutsOff = strfind(d2',[1 0]);
            if isempty(stateBoutsOn)
                continue
            elseif isempty(stateBoutsOff)
                continue
            end
            
            %make sure onsets and offsets align
            stateBoutsOff(stateBoutsOff<stateBoutsOn(1)) = [];
            stateBoutsOn(stateBoutsOn>stateBoutsOff(end)) = [];
            
            
            %find behavior instances within each bout of state
            for b=1:length(stateBoutsOn)
                
                %bout duration
                dur = stateBoutsOff(b)-stateBoutsOn(b);
                
                %only look at behavior instances inside the bout
                behInState = behIndex(behIndex>=stateBoutsOn(b));
                behInState = behInState(behInState<stateBoutsOff(b));
                
                %index those behaviors with respect to bout start
                behInState = behInState-stateBoutsOn(b);
                
                %find time remaining in state as a fraction of total time
                normbehInState = (dur-behInState)./dur;
                pool = [pool; normbehInState];
                
                %create a binary instance vector of beahvior occurance over
                %state
                vec = zeros(dur,1);
                vec(behInState+1,1) = 1;
                
                %resample out of 100 to create % time in state
                psthvec = [psthvec resample(vec,100,dur)];
            end
            
        end
        
        
        subplot(1,8,s)
        hold on
        
        pools{n,s}=pool;
        if isempty(pools{n,s})
            continue
        end
        %plot cumulative density function
        h=cdfplot(pools{n,s})
        h.Color = sessCols{n};
        h.LineWidth = 2;
        if s<7
            title(strcat('State ',num2str(s)))
        elseif s==7
            title('Pain-up states')
        else
            title('Non-pain states')
        end
        if s==1
            ylabel('Cumulative probability')
        end
        if n==5
            xlabel('Fraction time remaining in state') 
            if s==8
                legend(behaviors)
            end
        end
        
        %save resampled data pooled over animals
        behaviorResampled{n,s} = psthvec;
 
    end
end

%% SECTION 2: Plot within-state behavior dynamics as state progresses. 
%OUTCOMES:
% 1. figure: mean and sem of behavior occurence over all animals in dataset
% in each state

behaviors = {'Still','Walk','Rear','Groom','Lick'};
figure
for s=1:8
    for n=1:5
        subplot(8,5,5*(s-1)+n)
        data = behaviorResampled{n,s};
        win=[1:size(data,1)];
        
        plot(nanmean(data(win,:),2),'color',sessCols{n},'Linewidth',1)
        hold on
        y_upper = nanmean(data(win,:),2)+nanstd(data(win,:),[],2)./(sqrt(size(data(win,:),2)-1));
        y_lower = nanmean(data(win,:),2)-nanstd(data(win,:),[],2)./(sqrt(size(data(win,:),2)-1));
        fill([1:size(data(win,:), 1), fliplr(1:size(data(win,:), 1))], [y_upper; flipud(y_lower)], sessCols{n}, 'FaceAlpha', 0.2, 'EdgeColor', 'none');
        
        xticklabels([[0:.2:1]])
        xticks([0 20 40 60 80 100])
        ylim([0 1])
        if n==5
            ylim([0 .25])
        end
        if n==1
            if s<7
                ylabel(strcat('State ',num2str(s)))
            elseif s==7
                ylabel('Pain state')
            else
                ylabel('Non-pain states')
            end
        end
        if s==1
            title(strcat('Probability ', behaviors{n}))
        end
        if s==6
            xlabel('Fraction time in State')
        end
    end
end

%% SECTION 3: KS test heatmaps. Reproduce Fig. 2o, S9d,f

%OUTCOMES:
% 1. pBeh: ks test pvalue comparing distribution of each
% behavior-over-state to each other behavior in that state
% 2. statBeh: ks test statistic for same comparison
% 3. figure: heatmaps of pBeh for each state
% 4. pState: ks test pvalue comparing distribution of each behavior over
% state to the same behavior in each other state
% 5. statState: ks test statistic for same comparison
% 6. figure: heatmaps of pState for each behavior

%compare within states, between behaviors
pBeh = zeros(5,5,6);
stateBeh = zeros(5,5,6);
for s=1:6
    for n=1:5
        x=pools{n,s};
        if ~isempty(x)
            for m=1:5
                y=pools{m,s};
                if ~isempty(y)
                    [h,pBeh(n,m,s),statBeh(n,m,s)] = kstest2(x,y);
                end
            end
        end
    end
end

pState = zeros(6,6,5);
stateState = zeros(6,6,5);
%compare within behaviors, between states
for s=1:5
    for n=1:6
        x=pools{s,n};
        if ~isempty(x)
            for m=1:6
                y=pools{s,m};
                if ~isempty(y)
                    [h,pState(n,m,s),statState(n,m,s)] = kstest2(x,y);
                end
            end
        end
    end
end

behaviors={'Still','Walk','Rear','Groom','Left lick'};

figure
for s=1:6
    subplot(3,2,s)
    heatmap(pBeh(:,:,s),'XData',behaviors,'YData',behaviors,'Colorlimits',[0 1])
    title(strcat('State ', num2str(s)))
end
colormap(palette('scheme',4))
figure
for s=1:5
    subplot(3,2,s)
    heatmap(pState(:,:,s),'Colorlimits',[0 1])
    title(behaviors{s})
end
colormap(palette('scheme',4))

%% Section 4: Markov simulation. Reproduce Fig. S9b top

%OUTCOMES
% 1. simBehavior: simulated Pain State 4 behavior from all possible initial conditions
% 2. figure: mean and sem of stimulated behavior occurence over all iterations in each state

initBeh = [1 2 3 4 5];
medLengthState = 30;
dt = 20;
timeSteps = medLengthState*dt;
%use pain state 4 for transition probabilities
tProb = reshape(squeeze(c(4,:,:)),6,6);
totIts = 100;
bucketSize = 1000;
simBehavior = zeros(timeSteps,totIts,length(initBeh));

%create a bucket where each behavior is present in proportion to the
%probability it would be chosen given each possible initial state

for b=1:length(initBeh) %given each initial state
    tic
    bucket = [];
    
    %probabilities of transitions from behavior b to each other behavior
    probabilities = tProb(b,:);
    
    for n=1:length(initBeh) %load bucket with the number of instances of each other behavior, proportionate to their probabilities, as given by the transition matrix (out of 1000)
        bucket = [bucket; n.*ones(round(probabilities(n).*bucketSize),1)];
    end
    
    %save buckets
    buckets{b} = bucket;

    toc
end

%simulate behavior given each initial condition
totIts = 1000;
for b=1:length(initBeh)
    tic
    iB = initBeh(b);
    for i = 1:totIts
        simBehavior(1,i,b) = iB; %initialize timeseries with initial behavior
        
        for t=2:timeSteps
            
            bucketToPick = buckets{simBehavior(t-1,i,b)};
            
            r = randperm(length(bucketToPick)); %pick an index from the bucket
            
            simBehavior(t,i,b) = bucketToPick(r(1)); %get the behavior at that index
        end
    end
    toc
end



%concatenate behavior over conditions
dat=[];
for l=1:5
    dat = [dat simBehavior(:,:,l)];
end


%plot
behaviors = {'Still','Walk','Rear','Groom','Lick'};
sessCols = {'r',[253,166,58]./255,[54,128,45]./255,[51,102,255]./255,[76,0,164]./255};

figure

for n=1:5
    %binarize data with respect to behavior of interest
    data = dat;
    data(data~=n) = 0;
    data(data==n) = 1;
    
    subplot(1,5,n)
    
    win=[1:size(data,1)];
    plot(nanmean(data(win,:),2),'color',sessCols{n},'Linewidth',1)
    hold on
    y_upper = nanmean(data(win,:),2)+nanstd(data(win,:),[],2)./(sqrt(size(data(win,:),2)-1));
    y_lower = nanmean(data(win,:),2)-nanstd(data(win,:),[],2)./(sqrt(size(data(win,:),2)-1));
    fill([1:size(data(win,:), 1), fliplr(1:size(data(win,:), 1))], [y_upper; flipud(y_lower)], sessCols{n}, 'FaceAlpha', 0.2, 'EdgeColor', 'none');
    title(strcat('Probability ', behaviors{n}, ' over Pain State'))
    xticklabels([0 30])
    xticks([0 600])
    if n<5
        ylim([0 .5])
    else
        ylim([0 0.25])
    end
    if l==5
        xlabel('"Seconds"')
    end
    if n==1
        ylabel('Simulated behavior probability')
    end
    
end

%% Section 5: CDF and KS test for simulated State 4 behavior. Reproduce Fig. S9c top

%OUTCOMES
% 1. poolSim: cell containing fraction time remaining in a state from each
% behavior instance detected for each simulated behavior
% 2. pSim: ks test pvalue comparing distribution of each simulated behavior
% in State 4 to each real behavior in State 4
% 3. statBeh: ks test statistic for same comparison
% 4. figure: heatmaps of pSim

sessCols = {'r',[253,166,58]./255,[54,128,45]./255,[51,102,255]./255,[76,0,164]./255};
figure
simBehavior2=dat(100:end,:);
for n=1:5
    pool = [];
    psthvec=[];
    for a=1:size(dat,2)
        d1 = dat(:,a);
        simBehavior2 = find(d1==n);
        
        normbehInState = (size(dat,1)-simBehavior2)./size(dat,1);
        pool = [pool; normbehInState];
        
    end
    
    
    hold on
    
    poolSim{n}=pool;
    if isempty(poolSim{n})
        continue
    end
    h=cdfplot(poolSim{n})
    h.Color = sessCols{n};
    h.LineWidth = 2;
    behaviorResampled{n} = psthvec;
    
end


pSim = nans(5,5);
figure
for n=1:5
    [h,p,statSim] = kstest2(poolSim{n},pools{n,4});
    pSim(n,m,1) = p;
    statSim(n,m,1) = k;
end
heatmap(ps(:,:,1),'XData',{'Still', 'Walk', 'Rear', 'Groom','Lick'},'YData',{'Still', 'Walk', 'Rear', 'Groom','Lick'},'Colorlimits',[0 1])
colormap(palette('scheme',4))

%%

