
dotNum = 0:18;

switch set_i
    case 1
        %% Proportion
        pList = 0:1/6:1;
        
        idxWhite = [0:6; 0:2:12; 0:3:18];
        idxBlack = fliplr(idxWhite);
    case 2
        %% Difference
        pList = -6:2:6;
        
        idxWhite = [0:6; 3:9; 6:12];
        idxBlack = idxWhite - repmat(pList,[size(idxWhite,1) 1]);
end
idxWhite = idxWhite+1; idxBlack = idxBlack+1;

switch PE_i
    case 1 %%% Power
        a = 0.9; b = 0.7; c = 0;
        
        act_w = a*dotNum.^b+c;
        act_b = a*dotNum.^b+c;
        
    case 2 %%% Exp
        a = 5; b = 1.2; c = a;
        act_w = c-a*b.^-dotNum;
        act_b = c-a*b.^-dotNum;
end

%%%%
figure('Position',[300 300 600 300]); hold on;
subplot(121); hold on;
h = plot(dotNum,act_w,'-','LineWidth',2.0); h.MarkerSize = 5;
h.Color = colorMat(PE_i,:);

set(gca,'xtick',0:3:18);
set(gca,'TickDir','out')

xlim([-1 19]);
if PE_i == 1
    ylim([0-0.25 8]);
elseif PE_i == 2
    ylim([0-0.25 5]);
end

xlabel('#White dots'); ylabel('Response');

%%
tp_w = act_w(idxWhite); tp_b = act_b(idxBlack);

tpPlot = tp_w;

col = 'rgb';
markerType = 'ov^';

subplot(122); hold on;
for cond_i = [3 2 1]
    h = plot(pList,tpPlot(cond_i,:),'-'); h.LineWidth = 1.5;
    h.Color = col(cond_i);
end
set(gca,'xtick',pList(1:1:end));
set(gca,'TickDir','out');
xlim([min(pList)-(pList(2)-pList(1))/2 max(pList)+(pList(2)-pList(1))/2]);

if PE_i == 1
    ylim([0-0.4 8]);
elseif PE_i == 2
    ylim([0-0.25 5]);
end

if set_i == 1
    xlabel('Proporion'); ylabel('Response');
else
    xlabel('Difference'); ylabel('Response');
end

%% Weight
if PE_i == 1
    wList = [1 9; 1 1.5; 1 1.2;];
elseif PE_i == 2
    wList = [1 9; 1 1.8; 1 1.4;];
end
wList = cat(1,wList,[1 1],rot90(wList,2));

figure('Position',[300 100 1200 150]); hold on;
for p_i = 1:length(pList)
    subplot(1,length(pList),p_i); hold on;
    
    ww = wList(p_i,1);
    wb = wList(p_i,2);
    
    resp = ww*tp_w + wb*tp_b;
    resp(resp<0) = 0;
    
    for ii = 1:size(resp,1)
        h = plot(pList,resp(ii,:),'-');
        h.Color = col(ii);
        
        tp = resp(ii,:);
        [mm,ind] = max(tp);
        
        hh = plot(pList(ind),mm,'v');
        hh.Color = h.Color; hh.MarkerFaceColor = h.Color;
    end
    
    set(gca,'xtick',pList(1:1:end));
    set(gca,'TickDir','out');

    if set_i == 1
        xlabel('Proporion'); ylabel('Response');
    else
        xlabel('Difference'); ylabel('Response');
    end
    
    xlim([min(pList)-(pList(2)-pList(1))/2 max(pList)+(pList(2)-pList(1))/2]);
end



