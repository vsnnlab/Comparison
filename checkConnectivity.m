clc;

layer_i = 3;
switch type_i
    case 1
        totPS = totPS_prop_L3;
        pList = propList;
        totPS_IncDec = totPS_IncDec_prop;
    case 2
        totPS = totPS_diff_L3;
        pList = diffList;
        totPS_IncDec = totPS_IncDec_diff;
end

%%% Settings
arraySz_conv = [55 55 96; 27 27 256; 13 13 384; 13 13 384; 13 13 256];
arraySz_pre = [nan nan nan; 27 27 96; 13 13 256; 13 13 384; 13 13 384];

num_unit_IncDec = prod([27 27 96; 13 13 256; 13 13 384; 13 13 384; 13 13 256],2);

totPS_prop = totPS_prop_L3;
totPS_diff = totPS_diff_L3;

totNPS_both = cell(1,1,20,5);

rePS_prop = [];
for p_i = 1:length(pList)
    rePS_prop = cat(1,rePS_prop,totPS_prop{1,p_i,1,layer_i});
end

rePS_diff = [];
for p_i = 1:length(pList)
    rePS_diff = cat(1,rePS_diff,totPS_diff{1,p_i,1,layer_i});
end

reSel = cat(1,rePS_prop,rePS_diff); reSel = unique(reSel);
nonSel = 1:num_unit(layer_i); nonSel(reSel) = [];

totNPS_both{1,1,1,layer_i} = nonSel;

%%%
totNPS_IncDec_re = cell(1,1);
tmp_incdec = [];
for ii = 1:2
    tmp_incdec = cat(1,tmp_incdec,totPS_IncDec{1,ii});
end
tmp_incdec = unique(tmp_incdec);

tmp = 1:num_unit_IncDec(layer_i-1); tmp(tmp_incdec) = [];
totNPS_IncDec_re{1,1} = tmp';

totWD = cell(1,7,2,2); %%% repeat_i,p_i,unitType_i,Prop/Diff
totWD_NPS = cell(1,1,2,2); %%% repeat_i,p_i,unitType_i,Prop/Diff

%% Connectivity: Source -> Prop/Diff unit
for p_i = 1:length(pList)
    IND_Post = totPS{1,p_i,1,layer_i};
    
    W = net_test.Layers(idxShuffle(layer_i)).Weights;
    [Cell_IND_pre,Cell_W_pre] = weightTrack(net_test,IND_Post,W,arraySz_conv,arraySz_pre,idxShuffle,layer_i);
    
    for unitType_i = 1:2
        if unitType_i == 1
            totPS_pre = totPS_IncDec;
        elseif unitType_i == 2
            totPS_pre = totNPS_IncDec_re;
        end
        
        wMat = nan(length(Cell_IND_pre),size(totPS_pre,2));
        wCounts = nan(length(Cell_IND_pre),size(totPS_pre,2));
        for id_i = 1:size(totPS_pre,2)
            IND_cand = totPS_pre{1,id_i};
            
            for neuron_i = 1:length(Cell_IND_pre)
                check = ismember(Cell_IND_pre{neuron_i},IND_cand);
                
                w_target = Cell_W_pre{neuron_i}(check);
                wMat(neuron_i,id_i) = mean(w_target);
                wCounts(neuron_i,id_i) = mean(check);
            end
        end
        
        totWD{repeat_i,p_i,unitType_i,type_i} = wMat;
    end
end

%% Connectivity: Source -> Non-Prop/Diff unit

IND_Post = totNPS_both{1,1,1,layer_i}; % neuron index

W = net_test.Layers(idxShuffle(layer_i)).Weights;
[Cell_IND_pre,Cell_W_pre] = weightTrack(net_test,IND_Post,W,arraySz_conv,arraySz_pre,idxShuffle,layer_i);

for unitType_i = 1:2
    if unitType_i == 1
        totPS_pre = totPS_IncDec;
    elseif unitType_i == 2
        totPS_pre = totNPS_IncDec_re;
    end
    
    wMat = nan(length(Cell_IND_pre),size(totPS_pre,2));
    wCounts = nan(length(Cell_IND_pre),size(totPS_pre,2));
    for id_i = 1:size(totPS_pre,2)
        IND_cand = totPS_pre{1,id_i};
        
        for neuron_i = 1:length(Cell_IND_pre)
            check = ismember(Cell_IND_pre{neuron_i},IND_cand);
            
            w_target = Cell_W_pre{neuron_i}(check);
            wMat(neuron_i,id_i) = mean(w_target);
            wCounts(neuron_i,id_i) = mean(check);
        end
    end
    
    totWD_NPS{repeat_i,1,unitType_i,type_i} = wMat;
end

%% Plot
col = 'mck';
txt = {'P_{pref}','P_{pref}'};

totW = nan(size(totWD,1),7,3);
totW_NPS = nan(size(totWD,1),1,3);

for p_i = 1:7
    tpw = totWD{1,p_i,1,type_i};
    tpw_non = totWD{1,p_i,2,type_i};
    
    tp = cat(2,fliplr(tpw),tpw_non);
    
    totW(1,p_i,:) = mean(tp,1);
end

tpw = totWD_NPS{1,1,1,type_i};
tpw_non = totWD_NPS{1,1,2,type_i};

tp = cat(2,fliplr(tpw),tpw_non);

totW_NPS(1,1,:) = mean(tp,1);

figure('Position',[800 200 500 250]); hold on;
cnt = 1;
for p_i = [2 6]
    subplot(1,3,cnt); hold on;
    tot = [];
    
    plot([0 1]*4,[1 1]*0,'k--');
    for cond_i = 1:3
        tp = totW(:,p_i,cond_i);
        tp = squeeze(tp);
        
        tot = cat(2,tot,tp);
        h = plot(cond_i,nanmean(tp,1)','o');
        h.Color = col(cond_i);
        h.MarkerFaceColor = h.Color;
    end
    
    set(gca,'TickDir','out');
    set(gca,'xtick',1:3,'xtickLabel',{'Inc','Dec','Others'});
    xlim([0.5 3.5]);
    
    set(gca,'ytick',-5e-3:1e-3:5e-3);
    ylim([-1 1]*2e-3);
    
    box off
    if cnt == 1
        ylabel('Weight');
        title(['Small ' txt{1,cnt}]);
    elseif cnt == 2
        title(['Large ' txt{1,cnt}]);
    end
    cnt = cnt+1;
    
end

subplot(1,3,3); hold on;
plot([0 1]*4,[1 1]*0,'k--');
for cond_i = 1:3
    tp_nps = totW_NPS(:,1,cond_i);
    %     tot_nps = cat(2,tot_nps,tp_nps);
    
    h = plot(cond_i,nanmean(tp_nps,1),'o');
    h.Color = col(cond_i);
    h.MarkerFaceColor = h.Color;
end

set(gca,'TickDir','out');
set(gca,'xtick',1:3,'xtickLabel',{'Inc','Dec','Others'});
xlim([0.5 3.5]);

set(gca,'ytick',-5e-3:1e-3:5e-3);
ylim([-1 1]*2e-3);

box off
title('Non-selective');

%%%
figure('Position',[1300 200 250 250]); hold on;
plot([-7 7],[1 1]*0,'k--')
for cond_i = [3 2 1]
    tp = totW(:,:,cond_i);
    
    h = errorbar(pList,nanmean(tp,1),nanstd(tp,0,1)/sqrt(size(tp,1)),'o-');
    h.Color = col(cond_i); h.MarkerFaceColor = h.Color;
end

set(gca,'TickDir','out');
set(gca,'xtick',pList,'xtickLabel',list_xlabel{1,type_i});

dx = pList(2)-pList(1); dx = 0.5*dx;
xlim([min(pList)-dx max(pList)+dx]);

set(gca,'ytick',-5e-3:1e-3:5e-3);
ylim([-1 1]*2e-3);
ylabel('Weight');

switch type_i
    case 1
        xlabel('Preferred Proportion');
    case 2
        xlabel('Preferred Difference');
end












