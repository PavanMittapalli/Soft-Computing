clear all;
N = 6; %Population Size
rng(77);     
P = randi([0 1],N,10);  %Population of Solutions
P1 = zeros(N,10);       
x1 = zeros(N,1);     x1t = zeros(N,10);
x2 = zeros(N,1);     x2t = zeros(N,10);
prob = zeros(N,1);  cprob = zeros(N,1);
fit = zeros(N,1);

for k = 1:10             %Generation count
s = 0;

%Binary to Decimal Conversion & Fitness Calculation
for i = 1:N
    x1(i,1) = (0.5/31)*(16*P(i,1)+8*P(i,2)+4*P(i,3)+2*P(i,4)+P(i,5));
    x2(i,1) = (0.5/31)*(16*P(i,6)+8*P(i,7)+4*P(i,8)+2*P(i,9)+P(i,10));
    fit(i,1)= 1/(1+((x1(i,1) + x2(i,1) + x1(i,1)*x2(i,1) - x2(i,1)*x2(i,1) - 2*x1(i,1)*x1(i,1))));
end
x1t(:,k) = x1;      x2t(:,k) = x2;

%Cumulative Probability Calculation
prob = fit/sum(fit);
for i = 1:N
    s = s + prob(i,1);
    cprob(i,1) = s;
end
%Roulette-Wheel Selection
ran1 = rand(N,1);     
for i = 1:N
    if (ran1(i,1)>0) && (ran1(i,1) <= cprob(1,1))  
        P1(i,:) = P(1,:);
    end
    for j = 2:N
       if (ran1(i,1) > cprob(j-1,1)) && (ran1(i,1) <= cprob(j,1))
          P1(i,:) = P(j,:);
       end
    end   
end
P  =  P1;

%Single Point Cross-over
ran2 = randi([1,10]);
P1(:,1:ran2) = P(:,1:ran2);
i = 1;
while i<N
    P1(i,ran2+1:10) = P(i+1,ran2+1:10);
    P1(i+1,ran2+1:10) = P(i,ran2+1:10);
    i = i + 2;
end
P  = P1;
end