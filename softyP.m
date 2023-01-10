clear all;
%Getting Data from Excel Sheet
Data = xlsread('SC Data.xlsx');
count = 1;
%L = input neurons, M = Hidden neurons, N = Output Neurons
%P = Training patterns, PT = Testing patterns 
%%Since patterns are less, P and PT are defined directly.
L = 4, M = 7, N = 2, P = 16, n = 1, nmax = 1000000 , PT = 4;eta = 0.5; 
E = zeros(100,1), ITER = zeros(100,1); EP = zeros(4,2);
B = 13; iteration = zeros(B,1);
accur = zeros(B,2);
H = zeros(B,1);
V3d = zeros(L,16,B);
W3d = zeros(16,N,B);
%%Files for storing outputs
file1 = fopen('outputfile1.txt','w');
fprintf(file1,'Iterations\tMSE\n');
file2 = fopen('outputfile2.txt','w');
fprintf(file2,'Predicted Values using ANN\n');
fprintf(file2,' Y1\t\tY2\n');
file3 = fopen('outputfile3.txt','w');
fprintf(file3,'Error in Output Prediction \n');
fprintf(file3,'Variable 1\t Variable 2\n');

%Loop for changing number of hidden neurons everytime
for b=1:B
Vij = zeros(L+1,M), IH = zeros(P,M);
Wjk = zeros(M+1,N), OH = zeros(P,M);
IO  = zeros(P,N), OO = zeros(P,N); 
dVij= zeros(L,M), dWjk= zeros(M,N), error= zeros(N,1),Error = 1,tol = 0.001;
TGV = zeros(PT,N); 
rng(1); Vij = rand(L+1,M);
rng(2); Wjk = rand(M+1,N);
Accuracy = zeros(PT,N);
%IIFT,TGFT are total data
IIFT = Data(:,1:4);
TGFT = Data(:,5:6);

Imaxi = max(IIFT);
Imini = min(IIFT);

Omaxi = max(TGFT);
Omini = min(TGFT);
count = 1;

for i = 1:P+PT+PT
    j = rem(i,3);
    if(j~=0)
        II(count,:) = IIFT(i,:);
        TG(count,:) = TGFT(i,:);
        count = count + 1;
    end
    
end
count1 = 1,count2 = 1;
for k = 1:P+PT+PT
    j = rem(k,3);
    if(j==0 && count1~= 5)
        IIT(count1,:) = IIFT(k,:);
        TGT(count1,:) = TGFT(k,:);
        count1 = count1 + 1;
    else
        if(j==0 && count2 ~= 5)
            IIV(count2,:) = IIFT(k,:);
            TGV(count2,:) = TGFT(k,:);
            count2 = count2 + 1;
        end
    end
end
%IHT,OHT,IOT,OOT are used in Testing.
IHT = zeros(PT,M);  OHT = zeros(PT,M);  IOT  = zeros(PT,N); OOT = zeros(PT,N); 
%IHT,OHT,IOT,OOT are used in Validation.
IHV = zeros(PT,M);  OHV = zeros(PT,M);  IOV  = zeros(PT,N); OOV = zeros(PT,N); 

Vij(L+1,:) = 0;Wjk(M+1,:) = 0;

%Normalising input data
for q = 1:P
    for i = 1:L
        II(q,i) = 0.1 + (0.8 * ((II(q,i) - Imini(1,i))/(Imaxi(1,i) - Imini(1,i))));
    end 
end
for q = 1:PT
    for i = 1:L
        IIT(q,i) = 0.1 + (0.8 * ((IIT(q,i) - Imini(1,i))/(Imaxi(1,i) - Imini(1,i))));
        IIV(q,i) = 0.1 + (0.8 * ((IIV(q,i) - Imini(1,i))/(Imaxi(1,i) - Imini(1,i))));
    end 
end
%Normalising Target data
for q = 1:P 
    for k = 1:N
        TG(q,k) = 0.1 + (0.8 * ((TG(q,k) - Omini(1,k))/(Omaxi(1,k) - Omini(1,k))));
    end 
end
%Calculating Optimum Weights
while((Error>tol))
    IH = zeros(P,M);
    OH = zeros(P,M);
    IO = zeros(P,N);
    OO = zeros(P,N);
    error = zeros(N);
 %Calculating Input and Output to Hidden layer neurons for each pattern
for q = 1:P 
    for j = 1:M
        for i = 1:L
           IH(q,j) = IH(q,j) + (Vij(i,j) * II(q,i)); 
        end
        IH(q,j) = IH(q,j) + (Vij(L+1,M) * 1);
        OH(q,j) = 1/(1+exp(-IH(q,j)));
    end 
end

 %Calculating Input and Output to Output layer neurons for each pattern
for q = 1:P
    for k = 1:N
        for j = 1:M
           IO(q,k) = IO(q,k) + (Wjk(j,k) * OH(q,j)); 
        end
        IO(q,k) = IO(q,k) + (Wjk(M+1,N) * 1);
        OO(q,k) = 1/(1+exp(-IO(q,k)));
    end 
end

 %Updating Wjk values
for j = 1:M    
    for k = 1:N
        for q = 1:P    
           dWjk(j,k) = dWjk(j,k) + (eta*(TG(q,k)-OO(q,k))*(1-OO(q,k))*OO(q,k)*OH(q,j)); 
        end
           dWjk(j,k) = dWjk(j,k)/P;
           Wjk(j,k) = Wjk(j,k) + dWjk(j,k);
    end 
end

 %Updating Vij values
for i = 1:L
    for j = 1:M  
        for k = 1:N
            for q = 1:P 
                dVij(i,j) = dVij(i,j) + (eta*(TG(q,k)-OO(q,k))*(1-OO(q,k))*OO(q,k)*Wjk(j,k)*OH(q,j)*(1-OH(q,j))*II(q,i)); 
            end
        end
        dVij(i,j) = dVij(i,j)/(P*N); 
        Vij(i,j) = Vij(i,j) + dVij(i,j);
    end 
end

 %Error Calculation
for k = 1:N
    for q = 1:P
        error(k) = (error(k) + 0.5*((TG(q,k) - OO(q,k))^2));
    end
    error(k) = error(k)/P;
    Error = Error + error(k);
end
       
Error = Error/N;
Error
n = n + 1;
end
iteration(b,1) = n;

%Capturing Weights
for i=1:L
    for j=1:M
        V3d(i,j,b) = Vij(i,j);
    end
end
for i=1:M
    for j=1:N
        W3d(i,j,b) = Wjk(i,j);
    end
end

%De-Normalising Output data and Target Data
for q = 1:P
    for k = 1:N
        OO(q,k) = Omini(1,k) + ((Omaxi(1,k) - Omini(1,k)) * ((OO(q,k) - 0.1)/0.8));
        TG(q,k) = Omini(1,k) + ((Omaxi(1,k) - Omini(1,k)) * ((TG(q,k) - 0.1)/0.8));
    end 
end

%Code for Data Validation 
 %Calculating Input and Output  to Hidden layer neurons for each pattern
for q = 1:PT 
    for j = 1:M
        for i = 1:L
           IHV(q,j) = IHV(q,j) + (Vij(i,j) * IIV(q,i)); 
        end
        OHV(q,j) = 1/(1+exp(-IHV(q,j)));
    end 
end

 %Calculating Input and Output to Output layer neurons for each pattern
for q = 1:PT
    for k = 1:N
        for j = 1:M
           IOV(q,k) = IOV(q,k) + (Wjk(j,k) * OHV(q,j)); 
        end
        OOV(q,k) = 1/(1+exp(-IOV(q,k)));
    end 
end

 %De-Normalising Output data
for q = 1:PT
    for k = 1:N
        OOV(q,k) = Omini(1,k) + ((Omaxi(1,k) - Omini(1,k)) * ((OOV(q,k) - 0.1)/0.8));
        Accuracy(q,k) = abs(1 - (abs((TGV(q,k) - OOV(q,k)))/abs(TGV(q,k))));
    end 
end
 %%Finding MSE to find Optimum number of Hidden Neurons
accur(b,:) = mean(Accuracy);
H(b,1) = M;
diff = abs(TGV-OOV);
mse = zeros(1,2);
for d=1:2
    for c=1:4
        mse(1,d) = mse(1,d) + diff(c,d)*diff(c,d);
    end
    mse(1,d) = mse(1,d)/PT;
end
    
rmse = sqrt(mse);
RMSE(b,:) = rmse;
M = M + 1;
RSE(b,1) = (RMSE(b,1) + RMSE(b,2))*0.5;
end

%Plots between MSE and Hidden neurons
plot(H,RMSE(:,1));
plot(H,RMSE(:,2));

Opt = min(RSE);

for i=1:B
    if(Opt == RSE(i))
        OptH = H(i);
        pos = i;
    end
end

%Code for Final Testing
M = OptH
% Vij(L,M) = V3d(:,:,pos);
% Wjk(M,N) = W3d(:,:,pos);
for i=1:L
    for j=1:M
        Vij(i,j) = V3d(i,j,pos);
    end
end

for i=1:M
    for j=1:N
        Wjk(i,j) = W3d(i,j,pos);
    end
end
 %Calculating Input and Output to Hidden layer neurons for each pattern
for q = 1:PT 
    for j = 1:M
        for i = 1:L
           IHT(q,j) = IHT(q,j) + (Vij(i,j) * IIT(q,i)); 
        end
        OHT(q,j) = 1/(1+exp(-IHT(q,j)));
    end 
end

 %Calculating Input and Output to Output layer neurons for each pattern
for q = 1:PT
    for k = 1:N
        for j = 1:M
           IOT(q,k) = IOT(q,k) + (Wjk(j,k) * OHT(q,j)); 
        end
        OOT(q,k) = 1/(1+exp(-IOT(q,k)));
    end 
end

 %De-Normalising Obtained Output data
for q = 1:PT
    for k = 1:N
        OOT(q,k) = Omini(1,k) + ((Omaxi(1,k) - Omini(1,k)) * ((OOT(q,k) - 0.1)/0.8));
        FinalAccuracy(q,k) = 1 - (abs((TGT(q,k) - OOT(q,k)))/abs(TGT(q,k)));
        EP(q,k) = (abs((TGT(q,k) - OOT(q,k)))/abs(TGT(q,k)));
        fprintf(file2,'%f\t',OOT(q,k));
        fprintf(file3,'%f\t',EP(q,k));
    end 
    fprintf(file2,'\n');
    fprintf(file3,'\n');
end

mean(FinalAccuracy)


%%%%%%%%%   MSE vs NO.OF ITERATIONS GRAPH CODE %%%%%%%%%%%%
Vij = zeros(L+1,M), Wjk = zeros(M+1,N);
dVij= zeros(L,M), dWjk= zeros(M,N), error= zeros(N,1),Error = 1
n = 1 , nmax = 1000000; s = 1;
for q = 1:P
    for i = 1:L
        II(q,i) = 0.1 + (0.8 * ((II(q,i) - Imini(1,i))/(Imaxi(1,i) - Imini(1,i))));
    end 
end
for q = 1:P 
    for k = 1:N
        TG(q,k) = 0.1 + (0.8 * ((TG(q,k) - Omini(1,k))/(Omaxi(1,k) - Omini(1,k))));
    end 
end
while(n<=nmax)
    IH = zeros(P,M);
    OH = zeros(P,M);
    IO = zeros(P,N);
    OO = zeros(P,N);
    error = zeros(N);
%Calculating Input and Output to Hidden layer neurons for each pattern
for q = 1:P 
    for j = 1:M
        for i = 1:L
           IH(q,j) = IH(q,j) + (Vij(i,j) * II(q,i)); 
        end
        IH(q,j) = IH(q,j) + (Vij(L+1,M) * 1);
        OH(q,j) = 1/(1+exp(-IH(q,j)));
    end 
end

%Calculating Input and Output to Output layer neurons for each pattern
for q = 1:P
    for k = 1:N
        for j = 1:M
           IO(q,k) = IO(q,k) + (Wjk(j,k) * OH(q,j)); 
        end
        IO(q,k) = IO(q,k) + (Wjk(M+1,N) * 1);
        OO(q,k) = 1/(1+exp(-IO(q,k)));
    end 
end

%Updating Wjk values
for j = 1:M    
    for k = 1:N
        for q = 1:P    
           dWjk(j,k) = dWjk(j,k) + (eta*(TG(q,k)-OO(q,k))*(1-OO(q,k))*OO(q,k)*OH(q,j)); 
        end
           dWjk(j,k) = dWjk(j,k)/P;
           Wjk(j,k) = Wjk(j,k) + dWjk(j,k);
    end 
end

%Updating Vij values
for i = 1:L
    for j = 1:M  
        for k = 1:N
            for q = 1:P 
                dVij(i,j) = dVij(i,j) + (eta*(TG(q,k)-OO(q,k))*(1-OO(q,k))*OO(q,k)*Wjk(j,k)*OH(q,j)*(1-OH(q,j))*II(q,i)); 
            end
        end
        dVij(i,j) = dVij(i,j)/(P*N); 
        Vij(i,j) = Vij(i,j) + dVij(i,j);
    end 
end

%Error Calculation
for k = 1:N
    for q = 1:P
        error(k) = (error(k) + 0.5*((TG(q,k) - OO(q,k))^2));
    end
    error(k) = error(k)/P;
    Error = Error + error(k);
end
if(rem(n,10000) == 0)
    E(s,1) = Error;
    ITER(s,1) = n;
    s = s+1;
end
Error = Error/N;
Error
fprintf(file1,'%d\t%f\n',n,Error);
n = n + 1;
end
 % plot of MSE v iterations
plot(ITER,E);
fclose(file1);
fclose(file2);
fclose(file3);