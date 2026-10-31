function export_matlab_sessions()
% Decode MATLAB MCOS tables/Tobii gaze objects for step 2.
% Input: data/raw_use; outputs ONLY outputs/data_preprocessing/matlab_exports.
root = fileparts(fileparts(mfilename('fullpath')));
input_dir = fullfile(root,'data','raw_use');
output_dir = fullfile(root,'outputs','data_preprocessing','matlab_exports');
summary_file = fullfile(root,'outputs','data_preprocessing','session_summary.csv');
assert(isfile(summary_file),'Run step 1 before exporting MATLAB objects.');
assert(isfolder(input_dir),'Missing data/raw_use folder.');
if ~isfolder(output_dir), mkdir(output_dir); end
S = readtable(summary_file,'TextType','string');
assert(all(ismember(["source_file","session_order"],string(S.Properties.VariableNames))), ...
    'Session summary lacks source_file/session_order.');
for k=1:height(S)
    name=char(S.source_file(k));
    matfile=fullfile(input_dir,strrep(name,'/',filesep));
    assert(isfile(matfile),'Missing raw file: %s',matfile);
    fprintf('[%d/%d] %s\n',k,height(S),name);
    item=load(matfile,'infos'); infos=item.infos;
    assert(istable(infos.table),'MATLAB could not deserialize infos.table: %s',name);
    [~,stem]=fileparts(matfile);
    writetable(infos.table,fullfile(output_dir,[stem '_behavior.csv']));
    trial_col=zeros(0,1); sample_col=zeros(0,1); ts_col=zeros(0,1);
    lx_col=zeros(0,1); ly_col=zeros(0,1); rx_col=zeros(0,1); ry_col=zeros(0,1);
    sx_col=zeros(0,1); sy_col=zeros(0,1); fallback_col=zeros(0,1);
    n=height(infos.table); n_gaze=0;
    if isfield(infos,'dataMat') && isfield(infos.dataMat,'trial')
        n_gaze=numel(infos.dataMat.trial);
    end
    for tr=1:n
        if tr>n_gaze || ~isfield(infos.dataMat.trial(tr),'gazeData'), continue; end
        E=infos.dataMat.trial(tr).gazeData;
        if isempty(E), continue; end
        ns=numel(E); ts=nan(ns,1);
        lx=nan(ns,1); ly=nan(ns,1); rx=nan(ns,1); ry=nan(ns,1);
        for j=1:ns
            try
                ts(j)=double(E(j).SystemTimeStamp);
                left=double(E(j).LeftEye.GazePoint.OnDisplayArea);
                right=double(E(j).RightEye.GazePoint.OnDisplayArea);
                if numel(left)>=2, lx(j)=left(1); ly(j)=left(2); end
                if numel(right)>=2, rx(j)=right(1); ry(j)=right(2); end
            catch ME
                error('Tobii object decode failed: %s trial %d sample %d. Add Tobii SDK to MATLAB path. %s',name,tr,j,ME.message);
            end
        end
        if infos.adv_eye==1
            sx=rx; sy=ry; fx=lx; fy=ly;
        elseif infos.adv_eye==2
            sx=lx; sy=ly; fx=rx; fy=ry;
        else
            error('Unknown infos.adv_eye in %s',name);
        end
        invalid=~isfinite(sx)|~isfinite(sy)|sx<0|sx>1|sy<0|sy>1;
        fallback=invalid & isfinite(fx)&isfinite(fy)&fx>=0&fx<=1&fy>=0&fy<=1;
        sx(fallback)=fx(fallback); sy(fallback)=fy(fallback);
        trial_col=[trial_col;repmat(tr,ns,1)]; %#ok<AGROW>
        sample_col=[sample_col;(1:ns)']; %#ok<AGROW>
        ts_col=[ts_col;ts]; %#ok<AGROW>
        lx_col=[lx_col;lx];ly_col=[ly_col;ly]; %#ok<AGROW>
        rx_col=[rx_col;rx];ry_col=[ry_col;ry]; %#ok<AGROW>
        sx_col=[sx_col;sx];sy_col=[sy_col;sy]; %#ok<AGROW>
        fallback_col=[fallback_col;double(fallback)]; %#ok<AGROW>
    end
    gaze=table(trial_col,sample_col,ts_col,lx_col,ly_col, ...
        rx_col,ry_col,sx_col,sy_col,fallback_col, ...
        'VariableNames',{'trial','sample_index','system_timestamp_us', ...
        'left_x_norm','left_y_norm','right_x_norm','right_y_norm', ...
        'selected_x_norm','selected_y_norm','eye_fallback_used'});
    writetable(gaze,fullfile(output_dir,[stem '_gaze.csv']));
    fprintf('    %d behavioral rows; %d gaze samples\n',n,height(gaze));
end
fprintf('Exports saved under %s\n',output_dir);
end