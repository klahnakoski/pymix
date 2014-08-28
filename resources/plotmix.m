function res = plotmix(delta)

x = [-4:0.01:10];
y = zeros(1,length(x));


w = [0.3,0.4,0.3]

comp1 = normpdf(x,0.0,1.0);
comp2 = normpdf(x,1.5,0.6);
comp3 = normpdf(x,6.0,0.8);

%for i=1:length(x)
%    p = rand;
%    if p <= w(1)
%        y(i) =  normpdf(x(i),0.0,1.0);
%    else 
%        y(i) = normpdf(x(i),delta,2);   
%    end
%end


%comp2 = exppdf(x,2.0)

%y = (comp1) + (comp2);
y = (w(1) * comp1) + (w(2) * comp2)+ (w(3) * comp3);

prior = normpdf(x,2.5,3.0);

subplot(2,1,1)
plot(x,prior,'.b');
title('Prior over mean parameter')



subplot(2,1,2)
hold on
%plot(x,y,'*b');    
plot(x,comp1*w(1),'.g');
plot(x,comp2*w(2),'.r');
plot(x,comp3*w(3),'.y');
title('3 component mixture')
xlabel('x')
ylabel('P(x)')
res =1;
hold off
