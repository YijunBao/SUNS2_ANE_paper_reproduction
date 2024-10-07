function parsave(filename, varargin)
if strcmp(filename(end-3:end), '.mat')
    var_names = cell(1,length(varargin));
    for i = 1:length(varargin)
        eval(sprintf('%s=varargin{%d};',inputname(i+1),i));
        var_names{i} = inputname(i+1);
    end
    save(filename,var_names{:}, '-v7.3');
else
    error('filename must end with ".mat".')
end
end