function res = plotmix(delta)

n = 10000;
x = [-3:0.01:7];
y = zeros(1,length(x));

for i=1:n
    n1 = normpdf(x(i),0,1);
    n2 = normpdf(x(i),delta,1);
    y(i) = (0.5 * n1) + (0.5 * n2);
end
    
hold on
plot(x,y);    

return 1;