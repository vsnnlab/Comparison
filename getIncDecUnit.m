
layerList = {'pool2'};

for type_i = 1:2
    
    totPS_IncDec = cell(3,2,repeatN,length(layerList));
    totNPS_IncDec = cell(3,1,repeatN,length(layerList));
    
    totCorrs = cell(repeatN,length(layerList));
    totPvals = cell(repeatN,length(layerList));
    
    switch type_i
        case 1
            load('stimulusSets_proportion.mat');
        case 2
            load('stimulusSets_difference.mat');
    end
    
    layer_i = 1;
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
    
    matRes = [];
    for cond_i = 1:size(totData,1)
        matRes = cat(1,matRes,totRes{1,cond_i});
    end
    
    Corrs = zeros(size(matRes,3),3);
    Pvals = zeros(size(matRes,3),3);
    parfor n_i = 1 : size(matRes, 3)
        tmp_corr = nan(1,3); tmp_p = nan(1,3);
        for cond_i = 1:3
            resptmp = totRes{1,cond_i}(:,:,n_i);
            resptmp = mean(resptmp,1);
            
            px = repmat(pList,[size(resptmp,1) 1]);
            [r,p] = corr(px(:),resptmp(:),'Type','Kendall');
            
            tmp_corr(1,cond_i) = r; tmp_p(1,cond_i) = p;
        end
        
        Corrs(n_i,:) = tmp_corr;
        Pvals(n_i,:) = tmp_p;
    end
    
    %%% IncDec
    checkPval = Pvals < 0.05;
    idxP = logical(prod(checkPval,2) == 1);
    
    ind_inc = logical(prod(Corrs>0, 2)) & idxP;
    ind_dec = logical(prod(Corrs<0, 2)) & idxP;
    
    totCorrs{1,layer_i} = Corrs;
    totPvals{1,layer_i} = Pvals;
    
    for id_i = 1:2
        if id_i == 1
            ind = ind_dec;
        elseif id_i == 2
            ind = ind_inc;
        end
        
        totPS_IncDec{1,id_i,1,layer_i} = find(ind);
        totPS_IncDec{2,id_i,1,layer_i} = Pvals(ind,:);
        totPS_IncDec{3,id_i,1,layer_i} = Corrs(ind,:);
    end
    
    %%% Non IncDec
    checkPval_NPS = Pvals > 0.05;
    ind_NPS = logical(prod(checkPval_NPS,2) == 1);
    
    totNPS_IncDec{1,1,1,layer_i} = find(ind_NPS);
    totNPS_IncDec{2,1,1,layer_i} = Pvals(ind_NPS,:);
    totNPS_IncDec{3,1,1,layer_i} = Corrs(ind_NPS,:);
    
    switch type_i
        case 1
            totPS_IncDec_prop = totPS_IncDec;
        case 2
            totPS_IncDec_diff = totPS_IncDec;
    end
end