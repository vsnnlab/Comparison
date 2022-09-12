function net = randomizeNet(net,idxShuffle)

for s_i = 1:length(idxShuffle)
    layerind = idxShuffle(s_i);
    
    tw = net.Layers(layerind).Weights; sidx_w = randperm(numel(tw));
    tb = net.Layers(layerind).Bias; sidx_b = randperm(numel(tb));
    
    fan_in = size(tw,1)*size(tw,2)*size(tw,3);
    
    %%% He
    Wtmp = randn(size(tw))*sqrt(2/fan_in);
    Btmp = randn(size(tb));
    
    wmean = mean(tw(:)); wstd = std(tw(:));
    bmean = mean(tb(:)); bstd = std(tb(:));
    
    wrand = wmean-wmean + Wtmp*wstd/wstd;
    brand = bmean-bmean + Btmp*0*bstd/bstd;
    
    
    
    net_tmp = net.saveobj;
    
    net_tmp.Layers(layerind).Weights = wrand;
    net_tmp.Layers(layerind).Bias = brand;
    
    net = net.loadobj(net_tmp);
end