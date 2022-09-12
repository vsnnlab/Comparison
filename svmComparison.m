
switch type_i
    case 1
        load('stimulusSet_PropSVM_30dots.mat');
        totPS = totPS_prop_L5;
    case 2
        load('stimulusSet_DiffSVM_30dots.mat');
        totPS = totPS_diff_L5;
end

totPerf = nan(repeatN,2);
saveTestData = cell(3,repeatN,2);

%%% SVM setting
trainN = 800; testN = 200;
neuronN = 120;

condN = size(totData,1);
%% SVM
rePS = []; idxP= [];
for p_i = 1:length(pList)
    rePS = cat(1,rePS,totPS{1,p_i,1,layer_i});
    idxP = cat(1,idxP,p_i*ones(size(totPS{1,p_i,1,layer_i},1),1));
end

reNPS = totNPS{1,1,1,layer_i}'; %%% non-significant for all sets

totResp = cell(repeatN,2);

idx_sel = randsample(1:length(rePS),neuronN);
idx_nsel = randsample(1:length(reNPS),neuronN);

idxNeuron_sel = rePS(idx_sel); ppUsed = idxP(idx_sel);
idxNeuron_nsel = reNPS(idx_nsel);

trainData = cell(1,length(pList));
testData = cell(1,length(pList));
for p_i = 1:length(pList)
    totRes = cell(1,condN);
    for cond_i = 1:condN
        imds = totData{cond_i,p_i};
        if isempty(imds); continue; end
        
        imds = repmat(imds,[1 1 3 1]);
        imds = single(imds);
        
        act = activations(net_test,imds,layerList{layer_i});
        act = reshape(act,[size(act,1)*size(act,2)*size(act,3) 1 size(act,4)]);
        act = permute(act,[3 2 1]);
        
        totRes{1,cond_i} = cat(2,totRes{1,cond_i},act);
    end
    
    matRes = [];
    for cond_i = 1:condN
        matRes = cat(1,matRes,totRes{1,cond_i});
    end
    matRes = squeeze(matRes);
    
    %%%
    idx_train = randsample(1:size(matRes,1),trainN);
    
    tmp = 1:size(matRes,1); tmp(idx_train) = [];
    idx_test = randsample(tmp,testN);
    
    trainData{1,p_i} = matRes(idx_train,[idxNeuron_sel; idxNeuron_nsel]);
    testData{1,p_i} = matRes(idx_test,[idxNeuron_sel; idxNeuron_nsel]);
end

%% Make SVM input
for neuronSet_i = 1:2
    if neuronSet_i == 1
        idxn = 1:neuronN;
    elseif neuronSet_i == 2
        idxn = (1:neuronN)+neuronN;
    end
    
    tot_train = nan(length(pList)*(length(pList)-1)*trainN,neuronN*2);
    tot_trainAns = nan(length(pList)*(length(pList)-1)*trainN,1);
    
    tot_test = nan(length(pList)*(length(pList)-1)*testN,neuronN*2);
    tot_testAns = nan(length(pList)*(length(pList)-1)*testN,1);
    
    ts_train = 1; ts_test = 1; sampleP = [];
    for sample_i = 1:length(pList)
        for test_i = 1:length(pList)
            if isequal(sample_i,test_i); continue; end
            
            tot_train(ts_train:(ts_train+trainN-1),:) = cat(2,trainData{1,sample_i}(:,idxn),trainData{1,test_i}(:,idxn));
            tot_trainAns(ts_train:(ts_train+trainN-1)) = (sample_i<test_i)*ones(trainN,1);
            
            tot_test(ts_test:(ts_test+testN-1),:) = cat(2,testData{1,sample_i}(:,idxn),testData{1,test_i}(:,idxn));
            tot_testAns(ts_test:(ts_test+testN-1)) = (sample_i<test_i)*ones(testN,1);
            
            ts_train = ts_train + trainN;
            ts_test = ts_test + testN;
            
            sampleP = cat(1,sampleP,sample_i*ones(testN,1));
        end
    end
    
    %% SVM training
    SVMModel = fitclinear(tot_train, tot_trainAns); % optimizer OFF
    %     [SVMModel, FitInfo] = fitclinear(tot_train, tot_trainAns,...
    %         'OptimizeHyperparameters',{'Lambda'},...
    %         'HyperparameterOptimizationOptions',struct('ShowPlots',false,'Verbose',0));
    
    label = predict(SVMModel, tot_test);
    
    totPerf(1,neuronSet_i) = mean(label == tot_testAns);
    
    saveTestData{2,1,neuronSet_i} = tot_testAns;
    saveTestData{3,1,neuronSet_i} = label;
    
    %% tuning curves
    if neuronSet_i == 1
        dp = pList(2)-pList(1);
        dList = (min(pList)-max(pList)):dp:(max(pList)-min(pList));
        
        tuningCurves_correct = nan(neuronN,length(dList));
        tuningCurves_wrong = nan(neuronN,length(dList));
        
        indCorrect = logical(label==tot_testAns);
        indWrong = ~logical(label==tot_testAns);
        
        for ii = 1:neuronN
            %%% Correct trials
            respCorrect = tot_test(indCorrect,ii);
            sampleP_correct = sampleP(indCorrect);
            
            tmp = cell(1,length(dList));
            for tr_i = 1:size(respCorrect,1)
                idxtmp = sampleP_correct(tr_i)-ppUsed(ii) +length(pList);
                tmp{idxtmp} = cat(1,tmp{idxtmp},respCorrect(tr_i));
            end
            
            tmpresp = nan(1,length(dList));
            for jj = 1:length(tmpresp)
                tmpresp(jj) = mean(tmp{jj});
            end
            sCorr = max(tmpresp);
            tmpresp_correct = tmpresp/sCorr;
            
            %%% Wrong trials
            respWrong = tot_test(indWrong,ii);
            sampleP_wrong = sampleP(indWrong);
            
            tmp = cell(1,length(dList));
            for tr_i = 1:size(respWrong,1)
                idxtmp = sampleP_wrong(tr_i)-ppUsed(ii) +length(pList);
                tmp{idxtmp} = cat(1,tmp{idxtmp},respWrong(tr_i));
            end
            
            tmpresp = nan(1,length(dList));
            for jj = 1:length(tmpresp)
                tmpresp(jj) = mean(tmp{jj});
            end
            tmpresp_wrong = tmpresp/sCorr;
            
            %%% save (exclude inf case)
            if any(tmpresp_correct==inf) || any(tmpresp_wrong==inf)
                continue;
            else
                tuningCurves_correct(ii,:) = tmpresp_correct;
                tuningCurves_wrong(ii,:) = tmpresp_wrong;
            end
            
        end
        
        totResp{1,1} = cat(1,totResp{1,1},nanmean(tuningCurves_correct,1));
        totResp{1,2} = cat(1,totResp{1,2},nanmean(tuningCurves_wrong,1));
    end
end

%% Average performance
figure('Position',[800 550 1000 300]); hold on;
subplot(1,4,1); hold on;
tp = totPerf;

col = 'bw';
for ii = 1:size(tp,2)
    h = bar(ii,mean(tp(:,ii),1)); h.FaceColor = col(ii); h.EdgeColor = 'k';
    h.LineWidth = 1.5;
    if ii == 1; h.FaceColor = colorMat(type_i,:); end
    
    h = errorbar(ii,mean(tp(:,ii),1),std(tp(:,ii),0,1),'k.'); h.Marker = 'none';
    h.LineWidth = 1.5;
end
plot([0 1]*3,[1 1]*0.5,'k--');

set(gca,'xtick',1:2);
set(gca,'TickDir','out');
set(gca,'xtickLabel',{'Selective';'Non-Selective'});
xtickangle(45);
ylabel('Correct ratio');

xlim([0.2 2.8]); ylim([0.4 0.8]);

%%
neuronSet_i = 1;

map_correctness = nan(length(pList),length(pList),repeatN);

for repeat_i = 1:repeatN

        ts = 1;
        for sample_i = 1:length(pList)
            for test_i = 1:length(pList)
                
                if isequal(sample_i,test_i); continue; end
                
                idx = ts:(ts+testN-1);
                
                tp_ans = saveTestData{2,1,neuronSet_i};
                tp_pred = saveTestData{3,1,neuronSet_i};
                
                map_correctness(sample_i,test_i) = mean(tp_ans(idx) == tp_pred(idx));
                
                ts = ts + testN;
            end
        end

end

tp = nanmean(nanmean(map_correctness,3),4);

% figure('Position',[200 300 300 300]); hold on;
subplot(1,4,2); hold on;
imagesc(tp); axis image;
colormap(hot); colorbar; caxis([0.5 1]);
colormap(parula); colorbar; caxis([0.5 1]);

set(gca,'xtick',1:7,'xtickLabel',list_xlabel{1,type_i});
set(gca,'ytick',1:7,'ytickLabel',list_xlabel{1,type_i});
set(gca,'TickDir','out');

xlabel('Sample'); ylabel('Test');
title('SVM performance');

%% Distance effect (w/Data)
[xx,yy] = meshgrid(1:length(pList),1:length(pList));
ax = xx + 1i * yy;

temp = squeeze(nanmean(map_correctness,3));

xdiff = 1:length(pList)-1;
tp_mean = nan(1,length(xdiff));
tp_sd = nan(1,length(xdiff));

for p_i = xdiff
    idx = logical(abs(real(ax)-imag(ax)) == p_i);
    idx = repmat(idx,[1 1 size(temp,3)]);
    
    tp = temp.*idx;
    tp(isnan(tp)) = 0;
    tp(logical(tp==0)) = nan;
    
    tp = nanmean(nanmean(tp,2),1);
    
    tp_mean(p_i) = nanmean(tp);
    tp_sd(p_i) = nanstd(tp);
end

% figure('Position',[200 300 300 300]); hold on;
subplot(1,4,3); hold on;
x = xdiff*1/6; x = x';
y = tp_mean'; dy = tp_sd';

h=fill([x;flipud(x)],[y-dy;flipud(y+dy)],'-','linestyle','none');
h.FaceColor = col(type_i);
set(h,'facealpha',0.2);
h = plot(x,y,'o-','LineWidth',1.5); h.MarkerSize = 8;
h.Color = colorMat(type_i,:); h.MarkerFaceColor = h.Color;

set(gca,'xtick',pList,'xtickLabel',list_xlabel{1,type_i});
set(gca,'TickDir','out');

plot([-1 7],[1 1]*0.5,'k--')

dx = pList(2)-pList(1); dx = dx*0.5;
xlim([0 max(pList)+dx]); ylim([0.4 1.0]);
xlabel('d'); ylabel('Correct ratio');

%%
tot = cell(1,2);

tp_cor = totResp{1,1};
tp_cor = nanmean(tp_cor,1);

scaleF = tp_cor(length(pList));
tp_cor = tp_cor/scaleF;

tp_inc = totResp{1,2};
tp_inc = nanmean(tp_inc,1);
tp_inc = tp_inc/scaleF;

tot{1,1} = cat(1,tot{1,1},tp_cor);
tot{1,2} = cat(1,tot{1,2},tp_inc);

dataP = [1 0.77];
dxList = [-1 1]*0.15;

% figure('Position',[300 300 300 300]); hold on;
subplot(1,4,4); hold on;
for ii = 1:2
    if ii == 1
        tp = cat(2,tot{1,1}(:,7),tot{1,2}(:,7));
        h = bar(ii+dxList,mean(tp,1));
        h.FaceColor = colorMat(type_i,:);
        h = errorbar(ii+dxList,mean(tp,1),std(tp,0,1),'k.');
        h.Marker = 'none';
        
    elseif ii == 2
        h = bar(ii+dxList,dataP);
        h.FaceColor = 'w';
    end
end

set(gca,'TickDir','out');
set(gca,'xtick',[1+dxList 2+dxList],'xtickLabel',{'C','I','C','I'});
xlim([0.4 2.6])
xlabel('Untrained Network / Data'); ylabel('Correct ratio');




