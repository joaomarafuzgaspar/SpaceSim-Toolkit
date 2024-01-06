close all

data = readtable('../data/data.csv');

% Simulation parameters
dt = 60.0; % Time step [s]
T = length(data.dev_chief); % Duration [min]
t = 0:dt:dt * T; % Time vector [s]

% Setting the tick labels in LaTeX style
ax = gca;
ax.TickLabelInterpreter = 'latex';
ax.FontSize = 14;

% Plotting
figure(1)
hold on
plot(t(1:T) / (dt * dt), data.dev_chief, "LineWidth", 2)
plot(t(1:T) / (dt * dt), data.dev_deputy1, "LineWidth", 2)
plot(t(1:T) / (dt * dt), data.dev_deputy2, "LineWidth", 2)
plot(t(1:T) / (dt * dt), data.dev_deputy3, "LineWidth", 2)
grid on
set(gca, 'YScale', 'log')
legend show
ylabel('$\left\|\hat{\mathbf{r}}_k^{(i)} - \mathbf{r}_k^{(i)}\right\|_\mathrm{av}$ [km]', 'Interpreter', 'latex', 'FontSize', 16)
xlabel('$t$ [h]', 'Interpreter', 'latex', 'FontSize', 16);

% Legend
lgd = legend('$\mathcal{C}_1$', '$\mathcal{D}_1$', '$\mathcal{D}_2$', '$\mathcal{D}_3$');
lgd.FontSize = 16; % Adjust font size
lgd.Interpreter = 'latex'; % Set interpreter to latex

% Wait for this figure window to close before continuing
uiwait(gcf);