
%% Gather IncDec units
load('stimulusSets_dots_bySets.mat');

layerList = {'relu1','relu2','relu3','relu4','relu5'};
layerList_pre = {'','pool1','pool2','relu3','relu4'};

arraySz_conv = [55 55 96; 27 27 256; 13 13 384; 13 13 384; 13 13 256];
arraySz_pre = [nan nan nan; 27 27 96; 13 13 256; 13 13 384; 13 13 384];

layer_i = 3;
layer_connected = layer_i-1;

totIncDec = cell(repeatN,2,2);
saveResult = cell(repeatN,2,2);

totRes_pre = cell(1,size(totData,1));
for cond_i = 1:size(totData,1)
    for p_i = 1:size(totData,2)
        imds = totData{cond_i,p_i};
        if isempty(imds); continue; end
        imds = repmat(imds,[1 1 3 1]);
        imds = single(imds);
        
        %%% act_pre
        act = activations(net_test,imds,layerList_pre{layer_i});
        act = reshape(act,[size(act,1)*size(act,2)*size(act,3) 1 size(act,4)]);
        act = permute(act,[3 2 1]);
        
        totRes_pre{1,cond_i} = cat(2,totRes_pre{1,cond_i},act);
    end
end

for type_i = 1:2
    type_i
    
    switch type_i
        case 1
            totPS = totPS_prop_L3;
            totPS_IncDec = totPS_IncDec_prop;
            
            pList = propList;
            whiteDots = [0:6; 0:2:12; 0:3:18];
            blackDots = fliplr(whiteDots);
            
        case 2
            totPS = totPS_diff_L3;
            totPS_IncDec = totPS_IncDec_diff;
            
            pList = diffList;
            whiteDots = [0:6; 3:9; 6:12];
            blackDots = whiteDots - repmat(pList,[size(whiteDots,1) 1]);
    end
    
    for p_i = 2:6
        IND_Post = totPS{1,p_i,1,layer_i}; % neuron index
        
        W = net_test.Layers(idxShuffle(layer_i)).Weights;
        [Cell_IND_pre,Cell_W_pre] = weightTrack(net_test,IND_Post,W,arraySz_conv,arraySz_pre,idxShuffle,layer_i);
        
        for neuron_i = 1:length(Cell_IND_pre)
            idx_conn = Cell_IND_pre{neuron_i};
            w_target = Cell_W_pre{neuron_i};
            
            for id_i = 1:2
                idx_mono = totPS_IncDec{1,id_i};
                idxCheck = ismember(idx_conn,idx_mono);
                
                idx_IncDec = idx_conn(idxCheck);
                totIncDec{1,id_i,type_i} = cat(1,totIncDec{1,id_i,type_i},idx_IncDec);
            end
        end
    end
    
    for id_i = 1:2
        totIncDec{1,id_i,type_i} = unique(totIncDec{1,id_i,type_i});
        
        tmpAct = cell(1,3);
        for cond_i = 1:3
            tp = totRes_pre{1,cond_i}(:,:,totIncDec{1,id_i,type_i});
            tp = squeeze(nanmean(tp,1))';
            
            tmpAct{1,cond_i} = tp;
        end
        
        saveResult{1,id_i,type_i} = tmpAct;
    end
end

%% PCA
cond_i = 3;
tot = cell(2,2);
for type_i = 1:2
    for id_i = 1 : 2
        tp = saveResult{1,id_i,type_i}{1,cond_i};
        if id_i == 1; tp = fliplr(tp); end
        
        %%% PCA
        [coeff,score,latent,tsquared,explained,mu] = pca(tp);
        tmp = coeff(:,1); tmp = tmp';
        
        tot{type_i,id_i} = cat(1,tot{type_i,id_i},tmp);
    end
end

%% Fitting
xx = 0:dotNumList(cond_i); %%% for fitting
nx = 7:dotNumList(cond_i);

tarR = cell(2,2);
for type_i = 1:2
    for id_i = 1:2
        tp = tot{type_i,id_i};
        
        tpR2 = nan(size(tp,1),2);
        tarR2 = nan(size(tp,1),2);
        
        for repeat_i = 1:size(tp,1)
            rng(repeat_i)
            
            tmp = tp(repeat_i,:);
            
            yy = tmp(1:length(xx));
            [xData, yData] = prepareCurveData( xx, yy );
            
            %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %% Curve fitting
            %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %% Power function
            ft = fittype( 'a*x^b+c', 'independent', 'x', 'dependent', 'y' );
            func = @(x,a,b,c) a*x.^b+c;
            
            opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
            opts.Display = 'Off';
            opts.Lower = [0 0 -Inf];
            opts.Upper = [Inf 1 Inf];
            opts.StartPoint = rand(1,3);
            
            [fitresult, gof] = fit( xData, yData, ft, opts );
            a = fitresult.a; b = fitresult.b; c = fitresult.c;
            
            fy = func(nx,a,b,c);
            ny = yy(nx+1);
            
            tarR2(repeat_i,1) = sqrt(mean((ny-fy).^2));
            
            %% Exponential function
            ft = fittype( 'c-a*b^-x', 'independent', 'x', 'dependent', 'y' );
            func = @(x,a,b,c) c-a*b.^-x;
            
            opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
            opts.Display = 'Off';
            opts.Lower = [0 1 -Inf];
            opts.Upper = [Inf Inf Inf];
            opts.StartPoint = rand(1,3);
            
            [fitresult, gof] = fit( xData, yData, ft, opts );
            a = fitresult.a; b = fitresult.b; c = fitresult.c;
            
            fy = func(nx,a,b,c);
            ny = yy(nx+1);
            
            tarR2(repeat_i,2) = sqrt(mean((ny-fy).^2));
        end
        tarR{type_i,id_i} = tarR2;
    end
end

%% Plotting
col = 'br';
figure('Position',[1600 200 300 250]); hold on;
for type_i = 1:2
    subplot(1,2,type_i); hold on;
%     figure('Position',[300 + 200*(type_i-1) 200 200 300]); hold on;
    
    totPlot = [];
    for id_i = 1:2
        tp = tarR{type_i,id_i};
        totPlot = cat(3,totPlot,tp);
    end
    totPlot = mean(totPlot,3);
    
    h = errorbar(1:2,mean(totPlot,1),std(totPlot,0,1)/sqrt(size(totPlot,1)),'o-');
    h.Color = colorMat(type_i,:); h.MarkerFaceColor = h.Color;
    h.LineWidth = 1.5;
    
    xlim([0.5 2.5]);
    
    set(gca,'xtick',1:2,'xtickLabel',{'Pow','Exp'});
    set(gca,'TickDir','out');

    switch type_i
        case 1
            ylim([0.9 1.1]*1e-3);
            title('Source of Prop unit');
            ylabel('RMSE');
        case 2
            ylim([1.3 1.5]*1e-3);
            title('Source of Diff unit');
    end
end























