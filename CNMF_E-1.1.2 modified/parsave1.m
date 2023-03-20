function parsave1(filename, data, dataname)
if strcmp(filename(end-3:end), '.mat')
    eval([dataname,'=data;']);
    save(filename,dataname, '-v7.3');
else
    error('filename must end with ".mat".')
end
end