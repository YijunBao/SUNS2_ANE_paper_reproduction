parfor ii = 1:9
    test = ii;
    parsave1([num2str(ii),'.mat'],test, 'test')
end