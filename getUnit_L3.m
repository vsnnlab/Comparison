%% Find selective units
layerList = {'relu1','relu2','relu3','relu4','relu5'};
layer_i = 3; repeatN = 1;

input_x = 227; input_y = 227;
num_images = 200;

totAct = cell(1,2);

for type_i = 1:2
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
            totPS_prop_L3 = totPS; totNPS_prop = totNPS;
        case 2
            totPS_diff_L3 = totPS; totNPS_diff = totNPS;
    end
end
