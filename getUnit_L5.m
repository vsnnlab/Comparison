%% Find selective units
layerList = {'relu1','relu2','relu3','relu4','relu5'};
layer_i = 5; repeatN = 1;

input_x = 227; input_y = 227;
num_images = 200;

totAct_unit = cell(1,2);
for type_i = 1 : 2
    clc
    switch type_i
        case 1
            load('stimulusSets_proportion.mat');
            propList = pList;
        case 2
            load('stimulusSets_difference.mat');
            diffList = pList;
    end
    
    totPS = cell(2,length(pList),repeatN,length(layerList));
    totNPS = cell(2,1,repeatN,length(layerList));
    
    totRes = cell(1,size(totData,1));
    for cond_i = 1:size(totData,1)
        for p_i = 1:length(pList)
            imds = totData{cond_i,p_i};
            imds = repmat(imds,[1 1 3 1]);
            imds = single(imds);
            
            act = activations(net_test,imds,layerList{layer_i});
            act = reshape(act,[size(act,1)*size(act,2)*size(act,3) 1 size(act,4)]);
            act = permute(act,[3 2 1]);
            
            totRes{1,cond_i} = cat(2,totRes{1,cond_i},act);
        end
    end
    
    matRes = []; tmp = [];
    for cond_i = 1:size(totData,1)
        matRes = cat(1,matRes,totRes{1,cond_i});
        tmp = cat(1,tmp,mean(totRes{1,cond_i},1));
    end
    totAct_unit{1,type_i} = tmp;
    
    %% Two-way ANOVA
    setPS = []; setNPS = [];
    for neuron_i = 1:size(matRes,3)
        tempAct = matRes(:,:,neuron_i);
        
        if any(tempAct(:))
            
            peakMat = nan(1,size(totData,1));
            
            %%% Find zero firing for condition
            check = 0;
            for cond_i = 1:size(totData,1)
                idx = num_images*(cond_i-1)+1:num_images*(cond_i);
                
                tempfr = mean(mean(tempAct(idx,:),2));
                if isequal(tempfr,0); check = check +1; end
            end
            if check > 0; continue; end
            
            %%% ANOVA
            [p,tbl,stats] = anova2(tempAct,num_images,'off');
            
            pthre1 = 0.05; pthre2 = 0.05; pthre3 = 0.05;
            p1 = tbl{2,6}; p2 = tbl{3,6}; p3 = tbl{4,6};
            
            ind = logical(p1<pthre1) & logical(p2>pthre2) & logical(p3>pthre3);
            
            if ind == 1
                for cond_i = 1:size(totData,1)
                    idx = num_images*(cond_i-1)+1:num_images*(cond_i);
                    
                    %%% ANOVA
                    tempAct_cond = tempAct(idx,:);
                    
                    [~,idx_max] = max(mean(tempAct_cond,1));
                    peakMat(cond_i) = idx_max;
                end
                
                peakCheck = logical(length(unique(peakMat))==1);
                
                %%%
                tmp = mean(tempAct,1);
                maxVal = max(tmp); minVal = min(tmp);
                
                tmpSel = (maxVal-minVal)/(maxVal+minVal);
                isSel = tmpSel >= 0.4;
                
                if peakCheck && isSel
                    setPS = cat(2,setPS,neuron_i);
                end
            end
            
            %%% Non-selective
            ind_ns = logical(p1>pthre1) & logical(p2>pthre2) & logical(p3>pthre3);
            if ind_ns
                setNPS = cat(2,setNPS,neuron_i);
            end
            
        end
    end
    
    for neuron_i = 1:length(setPS)
        tp = mean(matRes(:,:,setPS(neuron_i)),1);
        
        [~,ind] = max(tp);
        totPS{1,ind,1,layer_i} = cat(1,totPS{1,ind,1,layer_i},setPS(neuron_i));
        totPS{2,ind,1,layer_i} = cat(1,totPS{2,ind,1,layer_i},tp);
    end
    
    totNPS{1,1,1,layer_i} = setNPS;
    
    switch type_i
        case 1
            totPS_prop_L5 = totPS; totNPS_prop = totNPS;
        case 2
            totPS_diff_L5 = totPS; totNPS_diff = totNPS;
    end
end

%% Plot sample responses
col = 'rgb';
xList ={'Preferred Proportion','Preferred Difference'};
p_i = 3;
for type_i = 1 : 2
    switch type_i
        case 1
            idx = totPS_prop_L5{1,p_i,1,layer_i};
            n_i = 11;
            
            pList = 0:1/6:1;
            whiteDots = [0:6;0:2:12;0:3:18];
            blackDots = fliplr(whiteDots);
            pList2 = whiteDots-blackDots;
            labelOrder = [1 2];
        case 2
            idx = totPS_diff_L5{1,p_i,1,layer_i};
            n_i = 5;
            
            pList = -6:2:6;
            whiteDots = [0:6;3:9;6:12];
            pList2 = whiteDots./repmat([6;12;18],[1 size(whiteDots,2)]);
            labelOrder = [2 1];
    end
    
    tp = totAct_unit{1,type_i}(:,:,idx(n_i));
    
    
    figure('Position',[100 + 300*(type_i-1) 600 300 200]); hold on;
    for ii = 1:2
        subplot(1,2,ii); hold on;
        if ii == 1
            xx = repmat(pList,[3 1]);
            
        elseif ii == 2
            xx = pList2;
        end
        
        for cond_i = 1:3
            h = plot(xx(cond_i,:),tp(cond_i,:),'-');
            h.Color = col(cond_i);
            h.MarkerFaceColor = h.Color;
            
            [~,ind] = max(tp(cond_i,:));
            h = plot([1 1]*xx(cond_i,ind),[0 1]*2,'--');
            if type_i == 2
                h = plot([1 1]*xx(cond_i,ind),[0 1]*2,'--');
            end
            h.Color = col(cond_i);
        end
        
        dx = xx(3,2)-xx(3,1); dx = dx*0.5;
        
        set(gca,'TickDir','out');
        set(gca,'xtick',xx(3,:),'xtickLabel',list_xlabel{1,labelOrder(ii)});
        xlabel(xList{1,labelOrder(ii)}); ylabel('Response (A.U.)')
        
        xlim([min(xx(:))-dx max(xx(:))+dx]);
        
        if type_i == 2 && ii == 2
            set(gca,'xtick',0:1/6:1);
        end
        ylim([0 2]);
    end
end

%% Plot distribution & avg curve
for type_i = 1 : 2
    switch type_i
        case 1
            totPS = totPS_prop_L5;
            pList = propList;
            txt = 'Proportion';
        case 2
            totPS = totPS_diff_L5;
            pList = diffList;
            txt = 'Difference';
    end
    
    totCount = nan(repeatN,length(pList));
    for p_i = 1:length(pList)
        tp = totPS{2,p_i,1,layer_i};
        totCount(1,p_i) = size(tp,1);
    end
    
    figure('Position',[100 + 300*(type_i-1) 150 300 350]); hold on;
    subplot(2,1,1); hold on;
    h = bar(pList,mean(totCount,1)); h.FaceColor = colorMat(type_i,:);
    h = errorbar(pList,mean(totCount,1),std(totCount,0,1),'k.');
    h.Marker = 'none';
    
    set(gca,'xtick',pList,'xtickLabel','');
    set(gca, 'YScale', 'log');
    set(gca,'ytick',10.^(-1:4));
    set(gca,'TickDir','out');
    ylabel('Counts');
    
    xlim([min(pList)-(pList(2)-pList(1))/2 max(pList)+(pList(2)-pList(1))/2]);
    ylim(10.^[0 3]);
    
    title([txt ' units']);
    
    %%% avg responses
    totCurve = cell(1,length(pList));
    for p_i = 1:length(pList)
        tp = totPS{2,p_i,1,layer_i};
        if isempty(tp); continue; end
        
        mm = max(tp,[],2);
        mm = repmat(mm,[1 size(tp,2)]);
        
        tp = tp./mm;
        tp_norm = mean(tp,1);
        
        totCurve{1,p_i} = cat(1,totCurve{1,p_i},tp_norm);
    end
    
    col = jet(7);
    
    subplot(2,1,2); hold on;
    for p_i = 1:length(pList)
        tp = totCurve{1,p_i};
        
        tp_norm = mean(tp,1);
        
        minVal = min(tp_norm); maxVal = max(tp_norm);
        tp_norm = (tp_norm-minVal) / (maxVal-minVal);
        
        h = plot(pList,mean(tp_norm,1),'-');
        h.LineWidth = 2;
        
        h.Color = col(p_i,:)*0.9;
    end
    
    xlim([min(pList)-(pList(2)-pList(1))/2 max(pList)+(pList(2)-pList(1))/2]);
    ylim([0 1]);
    
    set(gca,'xtick',pList,'xtickLabel',list_xlabel{1,type_i});
    set(gca,'TickDir','out');
    ylabel('Normalized response');
    xlabel(txt);
end
















