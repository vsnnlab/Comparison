function [Cell_IND_pre,Cell_W_pre] = weightTrack(net_test,IND_Post,W,arraySz_conv, arraySz_pre,idxShuffle,layer_i)

wSize = size(W);
[c,r] = meshgrid(1:size(W,1),1:size(W,2)); 
IND_rc = [r(:), c(:)];

Cell_IND_pre = cell(length(IND_Post),1); Cell_W_pre = cell(length(IND_Post),1);

for n_i = 1:length(IND_Post)
    ind_all_XY = IND_Post(n_i);
    
    [row,col,IND_post_chan] = ind2sub(arraySz_conv(layer_i,:),ind_all_XY);
    tw = W(:,:,:,IND_post_chan); tw = tw(:);
    
    %%%
    w = size(W(:,:,:,IND_post_chan),1);
    s = net_test.Layers(idxShuffle(layer_i)).Stride;
    p = net_test.Layers(idxShuffle(layer_i)).PaddingSize;

    row_pre = (row-1)*s(1) +w -2*p(1);
    col_pre = (col-1)*s(1) +w -2*p(1);

    [cc,rr] = meshgrid(-(w-1)/2:1:(w-1)/2);
    
    row_range = rr+row_pre; col_range = cc+col_pre;
    row_range = row_range(:); col_range = col_range(:);
    
    IND_pre_rc = zeros(size(W,1)*size(W,2),2);
    IND_pre_rc(:,1) = row_range; IND_pre_rc(:,2) = col_range;

    %%%
    temp_IND_rc = IND_rc;
    
    idx1 = find((IND_pre_rc(:,1)<=0 | IND_pre_rc(:,1)>arraySz_pre(layer_i,1)));
    if ~isempty(idx1)
        IND_pre_rc(idx1,:) = []; temp_IND_rc(idx1,:) = [];
    end
    
    idx2 = find((IND_pre_rc(:,2)<=0 | IND_pre_rc(:,2)>arraySz_pre(layer_i,1)));
    if ~isempty(idx2)
        IND_pre_rc(idx2,:) = []; temp_IND_rc(idx2,:) = [];
    end
    
    %%% backtracking
    if layer_i == 5 || layer_i == 4 || layer_i == 2
        %%% Grouped convolution
        if IND_post_chan <= arraySz_conv(layer_i,3)/2
            temp_IND_pre_row = repmat(IND_pre_rc(:,1),[arraySz_pre(layer_i,3)/2,1]);
            temp_IND_pre_col = repmat(IND_pre_rc(:,2),[arraySz_pre(layer_i,3)/2,1]);
            temp_IND_pre_chan = repmat((1:arraySz_pre(layer_i,3)/2),[length(IND_pre_rc(:,1)),1]); temp_IND_pre_chan = temp_IND_pre_chan(:);
            temp_IND_pre = sub2ind(arraySz_pre(layer_i,:),temp_IND_pre_row,temp_IND_pre_col,temp_IND_pre_chan);
        else
            temp_IND_pre_row = repmat(IND_pre_rc(:,1),[arraySz_pre(layer_i,3)/2,1]);
            temp_IND_pre_col = repmat(IND_pre_rc(:,2),[arraySz_pre(layer_i,3)/2,1]);
            temp_IND_pre_chan = repmat(arraySz_pre(layer_i,3)/2+(1:arraySz_pre(layer_i,3)/2),[length(IND_pre_rc(:,1)),1]); temp_IND_pre_chan = temp_IND_pre_chan(:);
            temp_IND_pre = sub2ind(arraySz_pre(layer_i,:),temp_IND_pre_row,temp_IND_pre_col,temp_IND_pre_chan); 
        end
        
        temp_IND_pre_row_w = repmat(temp_IND_rc(:,1),[arraySz_pre(layer_i,3)/2,1]);
        temp_IND_pre_col_w = repmat(temp_IND_rc(:,2),[arraySz_pre(layer_i,3)/2,1]);
        temp_IND_pre_chan_w = repmat((1:arraySz_pre(layer_i,3)/2),[length(temp_IND_rc(:,1)),1]); temp_IND_pre_chan_w  = temp_IND_pre_chan_w(:);
        temp_IND_pre_w = sub2ind(wSize,temp_IND_pre_row_w,temp_IND_pre_col_w,temp_IND_pre_chan_w);

    elseif layer_i == 3
        temp_IND_pre_row = repmat(IND_pre_rc(:,1),[arraySz_pre(layer_i,3),1]);
        temp_IND_pre_col = repmat(IND_pre_rc(:,2),[arraySz_pre(layer_i,3),1]);
        temp_IND_pre_chan = repmat((1:arraySz_pre(layer_i,3)),[length(IND_pre_rc(:,1)),1]); temp_IND_pre_chan = temp_IND_pre_chan(:);
        temp_IND_pre = sub2ind(arraySz_pre(layer_i,:),temp_IND_pre_row,temp_IND_pre_col,temp_IND_pre_chan);
        
        temp_IND_pre_row_w = repmat(temp_IND_rc(:,1),[arraySz_pre(layer_i,3),1]);
        temp_IND_pre_col_w = repmat(temp_IND_rc(:,2),[arraySz_pre(layer_i,3),1]);
        temp_IND_pre_chan_w = repmat((1:arraySz_pre(layer_i,3)),[length(temp_IND_rc(:,1)),1]); temp_IND_pre_chan_w  = temp_IND_pre_chan_w(:);
        temp_IND_pre_w = sub2ind(wSize,temp_IND_pre_row_w,temp_IND_pre_col_w,temp_IND_pre_chan_w);
    end
    
    Cell_IND_pre{n_i} = temp_IND_pre;
    Cell_W_pre{n_i} = tw(temp_IND_pre_w);
end
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    









