Lx = 64; Ly = 128;
Y_clean = reshape(1:Lx*Ly,Lx,Ly);
mean(Y_clean,'all')
scale_noise = 64;
Y_pois = poissrnd(Y_clean*scale_noise)/scale_noise;
noise_sigma = std(Y_pois - Y_clean,1,'all')
Y = Y_pois + randn(Lx,Ly)*noise_sigma;    
std(Y - Y_clean,1,'all')
mean(Y,'all')

