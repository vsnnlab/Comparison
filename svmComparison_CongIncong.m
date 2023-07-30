
load('stimulusSet_CongIncong.mat');
load('unitIdx_forSVM_CongIncong.mat');

%% Check cong/incong set
set_cong = 0; set_incong = 0;
for ii = 1:size(wbMat,1)
    for jj = 1:size(wbMat,1)
        if isequal(ii,jj); continue; end
        
        pval1 = wbMat(ii,1) / (sum(wbMat(ii,:),2));
        dval1 = wbMat(ii,1) - wbMat(ii,2);
        
        pval2 = wbMat(jj,1) / (sum(wbMat(jj,:),2));
        dval2 = wbMat(jj,1) - wbMat(jj,2);
        
        if pval1 < pval2
            if dval1 < dval2
                set_cong = set_cong+1;
            elseif dval1 > dval2
                set_incong = set_incong+1;
            end
            
        elseif pval1 > pval2
            if dval1 < dval2
                set_incong = set_incong+1;
            elseif dval1 > dval2
                set_cong = set_cong+1;
            end
        end
    end
end

%% Settings
repeatN = 1;
repeatPerNet = 1;

totPerf = nan(repeatN,repeatPerNet,2,3);
saveTestData = cell(3,repeatN,repeatPerNet,2,3);

trainN = 160; testN = 40;

neuronN = 500;

%% main
for repeat_i = seed
    
    totRes = cell(size(wbMat,1),3);
    unitN = zeros(1,3); tmpSel = cell(1,3);
    
    for layer_i = 3:5
        
        rePS = []; reDS = [];
        sel_prop = []; sel_diff = [];
        for p_i = 1:size(totPS_prop,2)
            rePS = cat(1,rePS,totPS_prop{1,p_i,repeat_i,layer_i});
            reDS = cat(1,reDS,totPS_diff{1,p_i,repeat_i,layer_i});
            
            %%% consider selectivity
            tmp = totPS_prop{2,p_i,repeat_i,layer_i};
            maxVal = max(tmp,[],2); minVal = min(tmp,[],2);
            sel_p = (maxVal-minVal)./(maxVal+minVal);
            sel_prop = cat(1,sel_prop,sel_p);
            
            tmp = totPS_diff{2,p_i,repeat_i,layer_i};
            maxVal = max(tmp,[],2); minVal = min(tmp,[],2);
            sel_d = (maxVal-minVal)./(maxVal+minVal);
            sel_diff = cat(1,sel_diff,sel_d);
        end
        reNPS = totNPS_both{1,1,repeat_i,layer_i};
        
        idx_del_p = ismember(rePS,reDS);
        idx_del_d = ismember(reDS,rePS);
        
        rePS(idx_del_p) = [];
        reDS(idx_del_d) = [];
        
        sel_prop(idx_del_p) = [];
        sel_diff(idx_del_d) = [];
        
        tmpSel{1,1} = cat(1,tmpSel{1,1},sel_prop);
        tmpSel{1,2} = cat(1,tmpSel{1,2},sel_diff);
        
        unitN(1) = unitN(1)+length(rePS);
        unitN(2) = unitN(2)+length(reDS);
        unitN(3) = unitN(3)+length(reNPS);
        
        for cond_i = 1:size(wbMat,1)
            imds = totData{1,cond_i};
            if isempty(imds); continue; end
            
            imds = repmat(imds,[1 1 3 1]);
            imds = single(imds);
            
            act = activations(net_test,imds,layerList{layer_i});
            act = reshape(act,[size(act,1)*size(act,2)*size(act,3) 1 size(act,4)]);
            act = permute(act,[3 2 1]);
            
            act = squeeze(act);
            
            totRes{cond_i,1} = cat(2,totRes{cond_i,1},act(:,rePS));
            totRes{cond_i,2} = cat(2,totRes{cond_i,2},act(:,reDS));
            totRes{cond_i,3} = cat(2,totRes{cond_i,3},act(:,reNPS));
        end
    end
    
    %% Sort unit by selectivity
    [sorted,idxSort] = sort(tmpSel{1,1},'descend');
    idxNeuron_prop = idxSort(1:neuronN);
    
    [sorted,idxSort] = sort(tmpSel{1,2},'descend');
    idxNeuron_diff = idxSort(1:neuronN);
    
    idxNeuron_nsel = randsample(1:unitN(3),neuronN);
    
    respData = cell(size(wbMat,1),3);
    for neuronSet_i = 1:3
        switch neuronSet_i
            case 1
                idxType = idxNeuron_prop;
            case 2
                idxType = idxNeuron_diff;
            case 3
                idxType = idxNeuron_nsel;
        end
        
        for wb_i = 1:size(wbMat,1)
            respData{wb_i,neuronSet_i} = cat(2,respData{wb_i,neuronSet_i},...
                totRes{wb_i,neuronSet_i}(:,idxType));
        end
    end
    
    for neuronSet_i = 1:3
        for k_i = 1:5
            idx_test = testN*(k_i-1)+1:testN*k_i;
            
            idx_train = 1:num_images;
            idx_train(idx_test) = [];
            
            trainData = cell(1,size(wbMat,1));
            testData = cell(1,size(wbMat,1));
            
            for wb_i = 1:size(wbMat,1)
                trainData{1,wb_i} = respData{wb_i,neuronSet_i}(idx_train,:);
                testData{1,wb_i} = respData{wb_i,neuronSet_i}(idx_test,:);
            end
            
            %% Make SVM inputs
            trainSet = cell(1); trainSet_ans = cell(1);
            
            trainSet_cong = nan(set_cong*trainN,neuronN*2); trainSet_cong_ans = nan(set_cong*trainN,1);
            trainSet_incong = nan(set_incong*trainN,neuronN*2); trainSet_incong_ans = nan(set_incong*trainN,1);
            
            testSet_cong = nan(set_cong*testN,neuronN*2); testSet_cong_ans = nan(set_cong*testN,1);
            testSet_incong = nan(set_incong*testN,neuronN*2); testSet_incong_ans = nan(set_incong*testN,1);
            
            ts_cong_train = 1; ts_cong_test = 1;
            ts_incong_train = 1; ts_incong_test = 1;
            
            %%%
            for sample_i = 1:size(wbMat,1)
                
                for test_i = 1:size(wbMat,1)
                    
                    if isequal(sample_i,test_i); continue; end
                    
                    tar_sample = wbMat(sample_i,1) - wbMat(sample_i,2);
                    tar_test = wbMat(test_i,1) - wbMat(test_i,2);
                    
                    con_sample = wbMat(sample_i,1)/sum(wbMat(sample_i,:),2);
                    con_test = wbMat(test_i,1)/sum(wbMat(test_i,:),2);
                    
                    if tar_sample < tar_test
                        if con_sample < con_test % congruent
                            
                            trainSet_cong(ts_cong_train:(ts_cong_train+trainN-1),:) = ...
                                cat(2,trainData{1,sample_i},trainData{1,test_i});
                            trainSet_cong_ans(ts_cong_train:(ts_cong_train+trainN-1),:) = 1;
                            
                            testSet_cong(ts_cong_test:(ts_cong_test+testN-1),:) = ...
                                cat(2,testData{1,sample_i},testData{1,test_i});
                            testSet_cong_ans(ts_cong_test:(ts_cong_test+testN-1),:) = 1;
                            
                            ts_cong_train = ts_cong_train+trainN;
                            ts_cong_test = ts_cong_test+testN;
                            
                        elseif con_sample > con_test % incongruent
                            
                            trainSet_incong(ts_incong_train:(ts_incong_train+trainN-1),:) = ...
                                cat(2,trainData{1,sample_i},trainData{1,test_i});
                            trainSet_incong_ans(ts_incong_train:(ts_incong_train+trainN-1),:) = 1;
                            
                            testSet_incong(ts_incong_test:(ts_incong_test+testN-1),:) = ...
                                cat(2,testData{1,sample_i},testData{1,test_i});
                            testSet_incong_ans(ts_incong_test:(ts_incong_test+testN-1),:) = 1;
                            
                            ts_incong_train = ts_incong_train+trainN;
                            ts_incong_test = ts_incong_test+testN;
                            
                        end
                        
                    elseif tar_sample > tar_test
                        if con_sample < con_test % incongruent
                            
                            trainSet_incong(ts_incong_train:(ts_incong_train+trainN-1),:) = ...
                                cat(2,trainData{1,sample_i},trainData{1,test_i});
                            trainSet_incong_ans(ts_incong_train:(ts_incong_train+trainN-1),:) = 0;
                            
                            testSet_incong(ts_incong_test:(ts_incong_test+testN-1),:) = ...
                                cat(2,testData{1,sample_i},testData{1,test_i});
                            testSet_incong_ans(ts_incong_test:(ts_incong_test+testN-1),:) = 0;
                            
                            ts_incong_train = ts_incong_train+trainN;
                            ts_incong_test = ts_incong_test+testN;
                            
                        elseif con_sample > con_test % congruent
                            
                            trainSet_cong(ts_cong_train:(ts_cong_train+trainN-1),:) = ...
                                cat(2,trainData{1,sample_i},trainData{1,test_i});
                            trainSet_cong_ans(ts_cong_train:(ts_cong_train+trainN-1),:) = 0;
                            
                            testSet_cong(ts_cong_test:(ts_cong_test+testN-1),:) = ...
                                cat(2,testData{1,sample_i},testData{1,test_i});
                            testSet_cong_ans(ts_cong_test:(ts_cong_test+testN-1),:) = 0;
                            
                            ts_cong_train = ts_cong_train+trainN;
                            ts_cong_test = ts_cong_test+testN;
                            
                        end
                    end
                end
            end
            
            %% define trainSet
            trainSet{1,1} = cat(1,trainSet_cong,trainSet_incong);
            trainSet_ans{1,1} = cat(1,trainSet_cong_ans,trainSet_incong_ans);
            
            
            %% SVM model
            switch_optimizer = 1;
            if ~switch_optimizer
                [SVMModel, FitInfo] = fitclinear(trainSet{1,1}, trainSet_ans{1,1});
                
            elseif switch_optimizer
                [SVMModel, FitInfo] = fitclinear(trainSet{1,1}, trainSet_ans{1,1},...
                    'OptimizeHyperparameters',{'Lambda'},...
                    'HyperparameterOptimizationOptions',struct('ShowPlots',false,'Verbose',0));
            end
            
            for set_i = 1:2
                if set_i == 1
                    testInput = testSet_cong;
                    testAns = testSet_cong_ans;
                elseif set_i == 2
                    testInput = testSet_incong;
                    testAns = testSet_incong_ans;
                end
                
                label = predict(SVMModel, testInput);
                
                totPerf(repeat_i,k_i,set_i,neuronSet_i) = mean(label == testAns);
                
                saveTestData{2,repeat_i,k_i,set_i,neuronSet_i} = testAns;
                saveTestData{3,repeat_i,k_i,set_i,neuronSet_i} = label;
            end
            
            
        end
    end
end

%% Plotting
plotPerf = totPerf;
plotPerf = plotPerf(:,:,:,[2 1 3]);

xList = [1 2];
dx = [-1 0 1]*0.2;

figure('Position',[100 300 300 300]); hold on;
for set_i = 1:2
    
    for ii = 1:3
        
        tp = plotPerf(:,:,set_i,ii);
        tp = squeeze(mean(tp,2));
        
        if size(tp,2) == 1; tp = tp'; end
        
        h = bar(xList(set_i)+dx(ii),nanmean(tp,1));
        h.BarWidth = 0.15; h.LineWidth = 1.0; h.EdgeColor = 'k';
        
        if ii == 1 % Diff
            h.FaceColor = colorMat(2,:);
        elseif ii == 2 % Prop
            h.FaceColor = colorMat(1,:);
        else % Non-selective
            h.FaceColor = [1 1 1]*0.8;
        end
        
        h = errorbar(xList(set_i)+dx(ii),nanmean(tp,1),nanstd(tp,0,1)/sqrt(size(tp,1)),'k.');
        h.LineWidth = 1.0;
        h.Marker = 'none';
    end
end

h = plot([0 1]*3,[1 1]*0.5,'k-');
h.Color = [1 1 1]*0.2;

set(gca,'TickDir','out');
set(gca,'xtick',[0.8 1 1.2  1.8 2 2.2],'xtickLabel',{'Diff','Prop','Non-sel','Diff','Prop','Non-sel'});
set(gca,'ytick',0:0.05:1);
ylabel('Performance');

xlim([0.5 2.5]); ylim([0.45 0.65]);


