function changedCells = permTest(traces,baselineTime,targetTime,numIt)
%this function checks whether cells are significantly modulated by certain
%   stimuli
% INPUT:    traces  - cell traces over trial of interest
%           baselineTime - array of times considered baseline
%           targetTime - array of times of interest
% OUTPUT:   changedCells - cell containing cells that are up, down, and not
%           regulated during target time
%
%
%
%Written by Sophie A. Rogers, Corder Laboratory, University of Pennsylvania
%%

up = [];
down = [];
non = [];
shuffStat = zeros(numIt,1);

for n=1:length(traces(1,:))
    stat = mean(traces(targetTime,n))-(mean(traces(baselineTime,n))+.5*std(traces(baselineTime,n)));
    
    points = [traces(baselineTime,n); traces(targetTime,n)];
    
    for m=1:numIt
        idx = randperm(length(points));
        shuff = points(idx);
        shuffBase = shuff(1:length(baselineTime));
        shuffTarget = shuff(length(baselineTime)+1:end);
        shuffStat(m,1) = abs(median(shuffTarget) - (median(shuffBase)+.5*std(shuffBase)));
    end
    
    if sum(abs(stat) > shuffStat) >= .99*numIt
        sig = 1;
    else
        sig = 0;
    end
        
    
    if sig == 1
        if stat>0
            up = [up n];
        else
            down = [down n];
        end
    else
        non = [non n];
    end
end

changedCells{1,1} = up;
changedCells{1,2} = down;
changedCells{1,3} = non;


