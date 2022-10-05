m = 1000;
g = 100;
sig = 1e-9; 
mpl = 2.44e19;

f = @(x, y) (-1/(x.^2)).*(y.^2-(0.192*m*mpl*sig*(x.^(3/2)).*exp(-x)).^2); 
xrange = [1, 1e4];
y0 = 0.192*m*mpl*sig*exp(-1);
[x, y] = ode15s(f, xrange, y0);

figure;
loglog(x, y);
hold on
yeq = 0.192*m*mpl*sig*(x.^(3/2)).*exp(-x);
loglog(x, yeq);
xlabel("x");
ylabel("y(x)");
xlim([1, 1e4]);
ylim([1, 1e14]);
title("Toy freeze out calculations");
grid on


legend('Y', 'Yeq'),

hold off
