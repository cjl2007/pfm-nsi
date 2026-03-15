function OUT = pfm_qc_plots(Input, UsabilityMdl)

% -------------------- defaults --------------------
if nargin < 2 || isempty(UsabilityMdl)
    UsabilityMdl = [];
end

OUT = struct();
headline_lambda = 10;
ridge_tag = sprintf('Lambda%d', headline_lambda);

% -------------------- extract core vectors --------------------
NSI_r2 = full(double(Input.NSI.Ridge.(ridge_tag).R2(:)));
MI     = full(double(Input.MoransI.mI(:)));
Slope  = full(double(Input.SpectralSlope.slope(:)));

OUT.data.NSI_r2 = NSI_r2;
OUT.data.MI     = MI;
OUT.data.Slope  = Slope;

% =====================================
% Prospective PFM usability projection 
% =====================================
hasUsability = ...
    isstruct(UsabilityMdl) && ...
    isfield(UsabilityMdl,'model') && ...
    isfield(UsabilityMdl,'grid')  && ...
    isfield(UsabilityMdl.grid,'x') && ...
    isfield(UsabilityMdl.grid,'ciLo') && ...
    isfield(UsabilityMdl.grid,'ciHi') && ...
    isfield(UsabilityMdl,'thresholds');

if hasUsability

    NSI_use = nanmean(NSI_r2);
    p_hat   = local_predict_binom_logit(UsabilityMdl.model, NSI_use);
    p_lo    = interp1(UsabilityMdl.grid.x, UsabilityMdl.grid.ciLo, NSI_use, 'linear','extrap');
    p_hi    = interp1(UsabilityMdl.grid.x, UsabilityMdl.grid.ciHi, NSI_use, 'linear','extrap');

    if p_hat >= 0.8
        label = 'HIGH confidence usable';
    elseif p_hat >= 0.6
        label = 'MODERATE–HIGH confidence';
    elseif p_hat >= 0.4
        label = 'LOW–MODERATE confidence';
    else
        label = 'LOW confidence usable';
    end

    % ---- store results ----
    OUT.usability.NSI_mean   = NSI_use;
    OUT.usability.p_hat      = p_hat;
    OUT.usability.ci95       = [p_lo p_hi];
    OUT.usability.decision   = label;
    OUT.usability.thresholds = UsabilityMdl.thresholds;

    % ---- terminal output ----
    fprintf('\n=== Prospective PFM usability projection ===\n\n');
    fprintf('Dataset summary\n');
    fprintf('  Mean NSI (R^2, λ=%d):        %.3f\n\n', headline_lambda, NSI_use);

    fprintf('Predicted usability (from trained NSI model)\n');
    fprintf('  P(PFM-usable | NSI):         %.2f\n', p_hat);
    fprintf('  95%% confidence interval:     [%.2f, %.2f]\n', p_lo, p_hi);
    fprintf('  Decision band:               %s\n\n', label);

    fprintf('Model reference thresholds (for context only)\n');
    for i = 1:numel(UsabilityMdl.thresholds.P)
        fprintf('  NSI corresponding to P=%.1f:  ~%.2f\n', ...
            UsabilityMdl.thresholds.P(i), ...
            UsabilityMdl.thresholds.NSI(i));
    end
    fprintf('\n===========================================\n\n');

    % ---- traffic-light confidence curve ----
    try
        h = plot_nsi_usability_curve_from_model(UsabilityMdl, ...
            'ShowTrafficLight', true, ...
            'ShowPoints', true, ...
            'PointNSI', NSI_use, ...
            'PointP',  p_hat);

        OUT.figures.usability_curve = h;
    catch ME
        warning('Usability curve plotting failed: %s', ME.message);
    end
end

% ==========================
% Figure 1: gray histograms 
% ==========================
figure('Color','w','Units','inches','Position',[1 1 9 3]);

subplot(1,3,1);
plot_hist_gray(NSI_r2, 'NSI');

subplot(1,3,2);
plot_hist_gray(MI, 'Moran''s I');

subplot(1,3,3);
plot_hist_gray(Slope, 'Spectral slope');

% =================================
% Figure 2: spectral power spectra 
% =================================
if isfield(Input,'SpectralSlope') && ...
   isfield(Input.SpectralSlope,'freq')  && ~isempty(Input.SpectralSlope.freq) && ...
   isfield(Input.SpectralSlope,'power') && ~isempty(Input.SpectralSlope.power)

    freq  = double(Input.SpectralSlope.freq(:));
    power = double(Input.SpectralSlope.power);
    [k, Nt] = size(power);

    smooth_win = 5;
    power_sm = zeros(k, Nt);
    for j = 1:Nt
        power_sm(:,j) = smooth_vec(power(:,j), smooth_win);
    end

    meanPower_sm = mean(power_sm, 2, 'omitnan');

    figure('Color','w','Units','inches','Position',[1 1 4 3]);
    hold on;
    for j = 1:Nt
        loglog(freq, power_sm(:,j), 'Color',[0.85 0.85 0.85],'LineWidth',0.5);
    end
    loglog(freq, meanPower_sm, 'k','LineWidth',2);

    vals = power_sm(:);
    vals = vals(vals > 0 & isfinite(vals));
    if ~isempty(vals)
        logvals = log10(vals);
        ylim([10^prctile(logvals,1) 10^prctile(logvals,99)]);
    end

    xlabel('Graph frequency (Laplacian eigenvalue)');
    ylabel('Power');
    title('Spatial power spectra');
    hold off;
end

% =============================================
% Figure 3 (advanced): NSI histograms by network
% =============================================
if isfield(Input,'Params') && isfield(Input.Params,'compute_network_histograms') && ...
        Input.Params.compute_network_histograms && ...
        isfield(Input,'NSI') && isfield(Input.NSI,'NetworkAssignment')
    netTag = ['Lambda' num2str(Input.Params.network_assignment_lambda)];
    if isfield(Input.NSI.NetworkAssignment, netTag)
        net = Input.NSI.NetworkAssignment.(netTag);
        if isfield(net,'NetworkIndex') && isfield(net,'NetworkLabels')
            netIdx = double(net.NetworkIndex(:));
            labels = net.NetworkLabels;
            nNet = numel(labels);
            cmap = lines(max(nNet, 7));
            if isfield(net,'NetworkColors') && ~isempty(net.NetworkColors)
                c = double(net.NetworkColors);
                if size(c,1) >= nNet && size(c,2) >= 3
                    cmap = c(1:nNet,1:3);
                end
            end
            xlims = local_finite_x_limits(NSI_r2);
            nRows = 10;
            nCols = max(2, ceil(nNet / nRows));
            figure('Color','w','Units','inches','Position',[1 1 max(8.8,4.4*nCols) 11.5]);
            tiledlayout(nRows,nCols,'Padding','compact','TileSpacing','compact');
            for k = 1:nNet
                nexttile;
                vals = NSI_r2(netIdx == k);
                vals = vals(isfinite(vals));
                if ~isempty(vals)
                    histogram(vals, 30, ...
                        'FaceColor', cmap(k,:), ...
                        'EdgeColor', 'none', ...
                        'FaceAlpha', 0.75);
                    hold on;
                    m = median(vals);
                    yl = ylim;
                    plot([m m], yl, 'k-', 'LineWidth', 1);
                    hold off;
                end
                if ~isempty(xlims), xlim(xlims); end
                title(sprintf('%s', labels{k}), ...
                    'FontWeight','normal', 'HorizontalAlignment','left');
                ylabel('Count');
                set(gca,'TickDir','out','Box','off','FontName','Arial','FontSize',9);
            end
            for kk = (nNet+1):(nRows*nCols)
                nexttile;
                axis off;
            end
            xlabel('NSI (R^2)');
        end
    end
end

% =============================================
% Figure 4 (advanced): NSI histograms by structure (LH/RH-collapsed)
% =============================================
if isfield(Input,'Params') && isfield(Input.Params,'compute_structure_histograms') && ...
        Input.Params.compute_structure_histograms && ...
        isfield(Input,'NSI') && isfield(Input.NSI,'StructureAssignment')
    structTag = ['Lambda' num2str(Input.Params.structure_assignment_lambda)];
    if isfield(Input.NSI.StructureAssignment, structTag)
        st = Input.NSI.StructureAssignment.(structTag);
        if isfield(st,'StructureLabelsByTarget') && isfield(st,'StructureLabelsUnique')
            labelsByTarget = st.StructureLabelsByTarget(:);
            uniq = st.StructureLabelsUnique(:);
            counts = zeros(numel(uniq),1);
            for k = 1:numel(uniq)
                counts(k) = sum(strcmp(labelsByTarget, uniq{k}));
            end
            [~, ix] = sort(counts, 'descend');
            uniq = uniq(ix);
            nStruct = numel(uniq);
            gray = linspace(0.25, 0.75, max(nStruct,2));
            xlims = local_finite_x_limits(NSI_r2);
            nRows = 10;
            nCols = max(2, ceil(nStruct / nRows));
            figure('Color','w','Units','inches','Position',[1 1 max(8.8,4.4*nCols) 11.5]);
            tiledlayout(nRows,nCols,'Padding','compact','TileSpacing','compact');
            for k = 1:nStruct
                nexttile;
                mask = strcmp(labelsByTarget, uniq{k});
                vals = NSI_r2(mask);
                vals = vals(isfinite(vals));
                if ~isempty(vals)
                    histogram(vals, 30, ...
                        'FaceColor', [gray(k) gray(k) gray(k)], ...
                        'EdgeColor', 'none', ...
                        'FaceAlpha', 0.85);
                    hold on;
                    m = median(vals);
                    yl = ylim;
                    plot([m m], yl, 'k-', 'LineWidth', 1);
                    hold off;
                end
                if ~isempty(xlims), xlim(xlims); end
                title(sprintf('%s', uniq{k}), ...
                    'FontWeight','normal', 'HorizontalAlignment','left');
                ylabel('Count');
                set(gca,'TickDir','out','Box','off','FontName','Arial','FontSize',9);
            end
            for kk = (nStruct+1):(nRows*nCols)
                nexttile;
                axis off;
            end
            xlabel('NSI (R^2)');
        end
    end
end

end

function xlims = local_finite_x_limits(vals)
v = vals(isfinite(vals));
if isempty(v)
    xlims = [];
    return;
end
lo = min(v);
hi = max(v);
if hi <= lo
    hi = lo + eps;
end
pad = 0.02 * (hi - lo);
xlims = [lo-pad hi+pad];
end

% ----------------- helpers -----------------

function y_s = smooth_vec(y, win)
    % Simple moving average smoother for column vectors
    if nargin < 2, win = 5; end
    y = y(:);  % enforce column
    if win <= 1 || numel(y) <= win
        y_s = y;
        return;
    end
    k = ones(win,1) / win;
    y_s = conv(y, k, 'same');
end
function plot_hist_gray(vals, ttl)

vals = vals(isfinite(vals));

histogram(vals, 40, ...
    'FaceColor',[0.75 0.75 0.75], ...
    'EdgeColor','none');

hold on;
m = median(vals);
yl = ylim;
plot([m m], yl, 'k-', 'LineWidth', 1);

set(gca, ...
    'FontName','Arial', ...
    'FontSize',10, ...
    'TickDir','out', ...
    'Box','off');

title(ttl, 'FontWeight','normal');
ylabel('Count');

hold off;
end
function h = plot_nsi_usability_curve_from_model(UsabilityMdl, varargin)

P = inputParser;
P.addParameter('ShowTrafficLight', true);
P.addParameter('ShowPoints', false);
P.addParameter('PointNSI', []);
P.addParameter('PointP', []);
P.addParameter('FigPos', [200 200 520 320]);
P.addParameter('AxPos',  [0.15 0.22 0.75 0.62]);
P.addParameter('FontSize', 13);
P.parse(varargin{:});
S = P.Results;

xgrid = UsabilityMdl.grid.x(:);
pHat  = UsabilityMdl.grid.p(:);
ciLo  = UsabilityMdl.grid.ciLo(:);
ciHi  = UsabilityMdl.grid.ciHi(:);

fig = figure('Color','w','Position',S.FigPos);
ax  = axes(fig); hold(ax,'on');
set(ax,'Position',S.AxPos);
set(ax,'FontName','Arial','FontSize',S.FontSize,'TickDir','out','Box','off');

xlabel(ax,'NSI');
ylabel(ax,'P(PFM-usable)');
ylim(ax,[0 1]); yticks(ax,0:0.2:1);

xl = [min(xgrid) max(xgrid)];
pad = 0.03*range(xl);
xlim(ax,[xl(1)-pad xl(2)+pad]);

% --- soft traffic light gradient + band overlays ---
if S.ShowTrafficLight
    xL = xlim(ax);
    yL = [0 1];
    n = 300;
    red    = [0.85 0.30 0.30];
    yellow = [0.95 0.85 0.30];
    green  = [0.35 0.75 0.45];
    mid = 0.55;
    C = zeros(n,1,3);
    for i = 1:n
        t = (i-1)/(n-1);
        if t <= mid
            a = t / mid;
            col = (1-a)*red + a*yellow;
        else
            a = (t-mid) / (1-mid);
            col = (1-a)*yellow + a*green;
        end
        C(i,1,:) = col;
    end
    hImg = image(ax, [xL(1) xL(2)], [yL(1) yL(2)], C);
    set(hImg, 'AlphaData', 0.18);
    set(ax, 'YDir', 'normal');

    % Keep four decision ranges, but split the low range into dark/light red
    % to match the updated Python traffic-light styling.
    bands = [ ...
        0.0 0.2 0.65 0.15 0.15; ...
        0.2 0.4 0.85 0.35 0.35; ...
        0.4 0.6 0.95 0.85 0.30; ...
        0.6 0.8 0.70 0.85 0.45; ...
        0.8 1.0 0.35 0.75 0.45];
    for b = 1:size(bands,1)
        y0 = bands(b,1); y1 = bands(b,2);
        col = bands(b,3:5);
        patch(ax,[xL(1) xL(2) xL(2) xL(1)],[y0 y0 y1 y1], ...
            col,'EdgeColor','none','FaceAlpha',0.08);
    end
end

% --- CI band ---
patch(ax,[xgrid; flipud(xgrid)], ...
         [ciLo;  flipud(ciHi)], ...
         [0.8 0.8 0.8],'EdgeColor','none','FaceAlpha',0.35);

% --- fitted curve ---
plot(ax,xgrid,pHat,'k-','LineWidth',1.8);

% --- optional dataset point ---
if S.ShowPoints && ~isempty(S.PointNSI)
    scatter(ax,S.PointNSI,S.PointP,45,'k','filled');
end

% --- thresholds ---
for i = 1:numel(UsabilityMdl.thresholds.P)
    x0 = UsabilityMdl.thresholds.NSI(i);
    if ~isnan(x0)
        xline(ax,x0,'k--','LineWidth',1);
    end
end

hold(ax,'off');
h.fig = fig;
h.ax  = ax;
end
