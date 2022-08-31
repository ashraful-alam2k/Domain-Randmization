 function [X_des_train,X_in_train,Y_des_train,Y_in_train, X_des_test,X_in_test,Y_des_test,Y_in_test] = Sim_data_func(QAM,Span_l,n_spans,Power,S_rate,Size_data,beta2_in,gamma_in,alpha_in, rolloff_in,noise_figure_in)
% function [Signal_x_after_fiber] = Sim_data_func(QAM,Span_l,n_spans,Power,S_rate,Size_data,beta2_in,gamma_in,alpha_in, rolloff_in,noise_figure_in)
 % function [dataIn,dataIny,dataMod,dataModsav,dataModsavy,dataMody,DBP_signalsx,DBP_signalsy,noise,noise_transmitter,noise_transmitter0,rrcFilter,rrcxFilter,rxFiltSignal,rxFiltSignaldown,rxFiltSignaldowny,rxFiltSignaltemp,rxFiltSignaly,Signal_x_after_fiber,Signal_x_before_DBP,txFiltSignal,txFiltSignaly,X_des_train,X_des_test,X_in_test,X_in_train,Y_des_test,Y_des_train,Y_in_test,Y_in_train] = Sim_data_func(QAM,Span_l,n_spans,Power,S_rate,Size_data,beta2_in,gamma_in,alpha_in, rolloff_in,noise_figure_in)

Pdbm = Power;

for t=1:2
    % for round=1:1
    %% System Parameters
    M = double(QAM); % Modulation 
    k = log2((M)); % Number of bscatteits per symbol
    numBits = double(k*Size_data);% was k*2^20; % Number of bits to process
    sps = 8; % 40 Number of samples per symbol (oversampling factor)
    filtlen = 2^12; % Filter length in symbols
    rolloff = rolloff_in; % Filter rolloff factor
    BAUD_RATE = S_rate; %34.4;
    R = double(BAUD_RATE)*10^9;               % baud rate
    Fs =R * sps;   % Sampling frequency
%     txFiltSignal2 =0;
%     txFiltSignaly2 =0;
    %% RRC filter
    rrcFilter = rcosdesign(double(rolloff),filtlen,sps);
%     WDMSpacing=50e9;
%     N_channels = 0; %10
    StPs = 1;
    DBP = 0; % 0 CDC, 1 DBP
    dwn=sps/2;
    Seed=[123,456;321,654];

%     [txFiltSignal,txFiltSignaly,Sca,dataMod,dataMody,dataIn,dataIny,dataModsav,dataModsavy]=signal_generator_local(Seed,t,rrcFilter,numBits,k,M,Pdbm,sps,filtlen);
%     txFiltSignal=txFiltSignal+txFiltSignal2;
%     txFiltSignaly=txFiltSignaly+txFiltSignaly2;
 if t==1
    seed1 = Seed(1,1);%123;
    seed2 = Seed(1,2);%456;
    rng(seed1);
    s1a = RandStream('swb2712','Seed',seed1);
    %     s1a = RandStream('swb2712','Seed','shuffle');
    dataIn = round(rand(s1a,double(numBits),1)); % Generate vector of binary data  dsfmt19937
    dataInMatrix = reshape(dataIn,length(dataIn)/k,k); % Reshape data into binary 4-tuples
    dataSymbolsIn = bi2de(dataInMatrix); % Convert to integers
    rng(seed2);
    s1b = RandStream('swb2712','Seed',seed2);
    dataIny =  round(rand(s1b,double(numBits),1)); % Generate vector of binary data
end
if t ==2
    seed1 = Seed(2,1);%321;
    seed2 = Seed(2,2);%654;
    rng(seed1);
    s1a = RandStream('dsfmt19937','Seed',seed1);
    dataIn = round(rand(s1a,double(numBits),1)); % Generate vector of binary data  dsfmt19937
    dataInMatrix = reshape(dataIn,length(dataIn)/k,k); % Reshape data into binary 4-tuples
    dataSymbolsIn = bi2de(dataInMatrix); % Convert to integers
    rng(seed2);
    s1b = RandStream('dsfmt19937','Seed',seed2);
    dataIny =  round(rand(s1b,double(numBits),1)); % Generate vector of binary data
end
dataInMatrixy = reshape(dataIny,length(dataIny)/k,k); % Reshape data into binary 4-tuples
dataSymbolsIny = bi2de(dataInMatrixy); % Convert to integers

%% Modulation with unit average power
dataMod =qammod(dataSymbolsIn,M,'UnitAveragePower',true);
dataMod2=[zeros(2,1); dataMod ;zeros(2,1)];
dataMod=dataMod2;
dataModsav=dataMod;
dataMody =qammod(dataSymbolsIny,M,'UnitAveragePower',true);
dataMod2y=[zeros(2,1); dataMody ;zeros(2,1)];
dataMody=dataMod2y;
dataModsavy=dataMody;
%% Changing Launch Power of signal
Pnois =  0.00 ; %0.008   %Important  Quantization Noise
Ns=size(dataModsav);
dataMod=dataModsav + ifft(fft(randn(Ns(1),1)).*sqrt(Pnois)) + 1i*ifft(fft(randn(Ns(1),1)).*sqrt(Pnois));
Sca=10^(double(Pdbm)/10)*1e-3;
% dataMod=sqrt(double(Sca/2.0))*dataMod;
dataMod=sqrt((Sca/2.0))*dataMod;

dataMody=dataModsavy + ifft(fft(randn(Ns(1),1)).*sqrt(Pnois)) + 1i*ifft(fft(randn(Ns(1),1)).*sqrt(Pnois));
dataMody=sqrt(double(Sca/2.0))*dataMody;

%% Applying filter and usampling it
txFiltSignal = upfirdn(dataMod,rrcFilter,sps,1)*sqrt(sps);
txFiltSignal = txFiltSignal((filtlen*sps/2)+1:end-sps*filtlen/2); %Filter Delay removal
txFiltSignaly = upfirdn(dataMody,rrcFilter,sps,1)*sqrt(sps);
txFiltSignaly = txFiltSignaly((filtlen*sps/2)+1:end-sps*filtlen/2); %Filter Delay removal
%% Channel Parameters
    alphadb=alpha_in; %dB
    alpha=double(alphadb)/(10*log10(exp(1)));
    gamma = double(gamma_in); %Fibre nonlinearity [1/(W.km)]
    DD = double(beta2_in);  %16.8 // ps/nm/km  dispersion parameter
    bb = -(1550e-9 ^ 2) * (double(DD) * 1e-3) / (2 * pi * 3e8);  % equation 2.3.5 agrawal
    beta_2=[0 0 bb]; %GVD s^2/km
    delta_t=1/Fs;
    spanlength=Span_l;
    h=1; %SSFM resolution in km
    nz= spanlength/h; 
    NFdb=noise_figure_in; %[db] 4.5
    NF=10^(double(NFdb)/10);
    span=double(n_spans); %16
    Gain=exp(alpha*double(spanlength));

    N=length(txFiltSignal);
    hc=6.63e-34;
    fc=193.1e12; %Hz
    No=hc*fc*(Gain-1)*NF;   %% Power spectral density of the noise
%     method = 'circular';
    %% SSFM, Gain and Noise of EDFA
    trans = 0.0000 ; % MZI Noise
    noise_transmitter = wgn(N,1,trans,1,'linear','complex');
    noise_transmitter0 = wgn(N,1,trans,1,'linear','complex');

    txFiltSignal = txFiltSignal + noise_transmitter;
    txFiltSignaly = txFiltSignaly + noise_transmitter0;
    signal_before_fiber = txFiltSignal;
    for i=1:span
%             disp(i)
%         [txFiltSignal,txFiltSignaly] = ssprop_2(txFiltSignal,txFiltSignaly,delta_t,h,nz,double(alpha),beta_2,gamma);
        [txFiltSignal,txFiltSignaly] = ssprop_2(txFiltSignal,txFiltSignaly,delta_t,h,nz,double(alpha),beta_2,gamma);

        noise=wgn(N,1,double(Fs*No),1,'linear','complex');

        txFiltSignal = txFiltSignal*sqrt(Gain);
        txFiltSignal = txFiltSignal+noise;
        txFiltSignaly = txFiltSignaly*sqrt(Gain);
        txFiltSignaly = txFiltSignaly+noise;
    end

    %% Receiver Filter Definition
    Signal_x_after_fiber = txFiltSignal;


    rrcxFilter = rcosdesign(double(rolloff),filtlen,sps);
    rxFiltSignaltemp = upfirdn(txFiltSignal,rrcxFilter,1,dwn)/sqrt(dwn);
    rxFiltSignaltemp = rxFiltSignaltemp(filtlen*(sps/dwn)/2 + 1:end - filtlen*(sps/dwn)/2); % Account for delay

    rxFiltSignaltempy = upfirdn(txFiltSignaly,rrcxFilter,1,dwn)/sqrt(dwn);
    rxFiltSignaltempy = rxFiltSignaltempy(filtlen*(sps/dwn)/2 + 1:end - filtlen*(sps/dwn)/2); % Account for delay

    %% CD or DBP Compensation
    a = mean(abs(Signal_x_after_fiber).^2);
    b = mean(abs(rxFiltSignaltemp).^2);
    c = a/b; 
    rxFiltSignal = rxFiltSignaltemp*sqrt(c);
    rxFiltSignaly = rxFiltSignaltempy*sqrt(c);
    step = spanlength / StPs;

    Signal_x_before_DBP = rxFiltSignal;
    DBP_signalsx = zeros(double(span),length(Signal_x_before_DBP));
    DBP_signalsy = zeros(double(span),length(Signal_x_before_DBP));

    for i=1:span
        rxFiltSignal = rxFiltSignal/sqrt(Gain);
        rxFiltSignaly = rxFiltSignaly/sqrt(Gain);
        [rxFiltSignal,rxFiltSignaly] = ssprop_2(rxFiltSignal,rxFiltSignaly,1/(Fs/dwn),step,StPs,-double(alpha),-beta_2,-gamma*DBP);
        DBP_signalsx(i,:)= rxFiltSignal;
        DBP_signalsy(i,:)= rxFiltSignaly;
    end
    
    %% Downsample
    rxFiltSignaldown= downsample(rxFiltSignal,sps/dwn)/sqrt(sps/dwn);
    rxFiltSignal=rxFiltSignaldown;

    rxFiltSignaldowny= downsample(rxFiltSignaly,sps/dwn)/sqrt(sps/dwn);
    rxFiltSignaly=rxFiltSignaldowny;

    %%  Carrier Phase Recovery
    errork=(dataMod'*dataMod)/(dataMod'*rxFiltSignal);
    rxFiltSignal=errork.*rxFiltSignal;

    errorky=(dataMody'*dataMody)/(dataMody'*rxFiltSignaly);
    rxFiltSignaly=errorky.*rxFiltSignaly;

    % Demodulation
    dataSymbolsOut1 = qamdemod((1/sqrt(Sca/2))*rxFiltSignal,M,'UnitAveragePower',true);
    dataSymbolsOut = dataSymbolsOut1(3:end-2);
    dataOutMatrix = de2bi(dataSymbolsOut,k);
    dataOut = dataOutMatrix(:); % Return data in column vector
    %%
    dataSymbolsOut1y = qamdemod((1/sqrt(Sca/2))*rxFiltSignaly,M,'UnitAveragePower',true);
    dataSymbolsOuty = dataSymbolsOut1y(3:end-2);
    dataOutMatrixy = de2bi(dataSymbolsOuty,k);
    dataOuty = dataOutMatrixy(:); % Return data in column vector

    % Calculating BER and Q^2 factor
    [numErrors,ber] = biterr(dataIn(2049:end-2048),dataOut(2049:end-2048));
    % [numErrors,ber] = biterr(dataIn,dataOut);
    Q2factor=20*log10(sqrt(2)*erfcinv(2*ber));
    fprintf('\n, the bit error rate is %5.2e, based on %d errors.\n, and Q factir = %f.\n', ...
        ber,numErrors,Q2factor)
    [numErrorsy,bery] = biterr(dataIny(2049:end-2048),dataOuty(2049:end-2048));
    % [numErrorsy,bery] = biterr(dataIny,dataOuty);
    Q2factory=20*log10(sqrt(2)*erfcinv(2*bery));
    fprintf('\n, the bit error rate is %5.2e, based on %d errors.\n, and Q factir = %f.\n', ...
        bery,numErrorsy,Q2factory)
    if t ==1
        X_des_train = (dataModsav(513:end-512));
        X_in_train = (rxFiltSignal(513:end-512)/sqrt(Sca/2));
        Y_des_train = (dataModsavy (513:end-512));
        Y_in_train = (rxFiltSignaly(513:end-512)/sqrt(Sca/2));    
    end
    if t==2
        X_des_test = (dataModsav(513:end-512));
        X_in_test = (rxFiltSignal(513:end-512)/sqrt(Sca/2));
        Y_des_test = (dataModsavy (513:end-512));
        Y_in_test = (rxFiltSignaly(513:end-512)/sqrt(Sca/2));  
    end
    
    %fprintf("%f",X_in_train(1))
end
 


end