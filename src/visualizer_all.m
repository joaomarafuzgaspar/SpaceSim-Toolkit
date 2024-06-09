close all

fcekf_data = readtable('data/fcekf_form1.csv');
hcmci_data = readtable('data/hcmci_form1.csv');
ccekf_data = readtable('data/ccekf_form1.csv');

fcekf_data = readtable('data/fcekf_form2.csv');
hcmci_data = readtable('data/hcmci_form2.csv');
ccekf_data = readtable('data/ccekf_form2.csv');

% Simulation parameters
dt = 60.0; % Time step [s]
T = length(fcekf_data.dev_chief); % Duration [min]
t = 0:dt:dt * T; % Time vector [s]

% Get the screen size
screenSize = get(0, 'ScreenSize');
screenWidth = screenSize(3);
screenHeight = screenSize(4);

% Define positions for the figures based on screen size relative to the center
figWidth = 560; % Leave some space between figures
figHeight = 420; % Leave some space for title bars
centerX = screenWidth / 2;
centerY = screenHeight / 2;
positions = [
    centerX - figWidth, centerY, figWidth, figHeight;
    centerX, centerY, figWidth, figHeight;
    centerX - figWidth, centerY - (figHeight + 80), figWidth, figHeight;
    centerX, centerY - (figHeight + 80), figWidth, figHeight;
    ];


% Plot 1: Chief
figure(1)
set(gcf, 'Position', positions(1, :))
ax = gca;
ax.TickLabelInterpreter = 'latex';
ax.FontSize = 14;
hold on
plot(t(1:T) / (dt * dt), fcekf_data.dev_chief, "LineWidth", 2)
plot(t(1:T) / (dt * dt), hcmci_data.dev_chief, "LineWidth", 2)
plot(t(1:T) / (dt * dt), ccekf_data.dev_chief, "LineWidth", 2)
grid on
set(gca, 'YScale', 'log')
legend show
xlabel('$t$ [h]', 'Interpreter', 'latex', 'FontSize', 16);
ylabel('$\left\|\hat{\mathbf{r}}_k^{\mathcal{C}_1} - \mathbf{r}_k^{\mathcal{C}_1}\right\|_\mathrm{av}$ [km]', 'Interpreter', 'latex', 'FontSize', 16)
ylim([1e-5 1e0])
lgd = legend('FCEKF', 'HCMCI', 'CCEKF');
lgd.FontSize = 16; % Adjust font size
lgd.Interpreter = 'latex'; % Set interpreter to latex

% Plot 2: Deputy 1
figure(2)
set(gcf, 'Position', positions(2, :))
ax = gca;
ax.TickLabelInterpreter = 'latex';
ax.FontSize = 14;
hold on
plot(t(1:T) / (dt * dt), fcekf_data.dev_deputy1, "LineWidth", 2)
plot(t(1:T) / (dt * dt), hcmci_data.dev_deputy1, "LineWidth", 2)
plot(t(1:T) / (dt * dt), ccekf_data.dev_deputy1, "LineWidth", 2)
grid on
set(gca, 'YScale', 'log')
legend show
xlabel('$t$ [h]', 'Interpreter', 'latex', 'FontSize', 16);
ylabel('$\left\|\hat{\mathbf{r}}_k^{\mathcal{D}_1} - \mathbf{r}_k^{\mathcal{D}_1}\right\|_\mathrm{av}$ [km]', 'Interpreter', 'latex', 'FontSize', 16)
ylim([1e-4 1e0])
lgd = legend('FCEKF', 'HCMCI', 'CCEKF');
lgd.FontSize = 16; % Adjust font size
lgd.Interpreter = 'latex'; % Set interpreter to latex

% Plot 3: Deputy 2
figure(3)
set(gcf, 'Position', positions(3, :))
ax = gca;
ax.TickLabelInterpreter = 'latex';
ax.FontSize = 14;
hold on
plot(t(1:T) / (dt * dt), fcekf_data.dev_deputy2, "LineWidth", 2)
plot(t(1:T) / (dt * dt), hcmci_data.dev_deputy2, "LineWidth", 2)
plot(t(1:T) / (dt * dt), ccekf_data.dev_deputy2, "LineWidth", 2)
grid on
set(gca, 'YScale', 'log')
legend show
xlabel('$t$ [h]', 'Interpreter', 'latex', 'FontSize', 16);
ylabel('$\left\|\hat{\mathbf{r}}_k^{\mathcal{D}_2} - \mathbf{r}_k^{\mathcal{D}_2}\right\|_\mathrm{av}$ [km]', 'Interpreter', 'latex', 'FontSize', 16)
ylim([1e-4 1e0])
lgd = legend('FCEKF', 'HCMCI', 'CCEKF');
lgd.FontSize = 16; % Adjust font size
lgd.Interpreter = 'latex'; % Set interpreter to latex

% Plot 4: Deputy 3
figure(4)
set(gcf, 'Position', positions(4, :))
ax = gca;
ax.TickLabelInterpreter = 'latex';
ax.FontSize = 14;
hold on
plot(t(1:T) / (dt * dt), fcekf_data.dev_deputy3, "LineWidth", 2)
plot(t(1:T) / (dt * dt), hcmci_data.dev_deputy3, "LineWidth", 2)
plot(t(1:T) / (dt * dt), ccekf_data.dev_deputy3, "LineWidth", 2)
grid on
set(gca, 'YScale', 'log')
legend show
xlabel('$t$ [h]', 'Interpreter', 'latex', 'FontSize', 16);
ylabel('$\left\|\hat{\mathbf{r}}_k^{\mathcal{D}_3} - \mathbf{r}_k^{\mathcal{D}_3}\right\|_\mathrm{av}$ [km]', 'Interpreter', 'latex', 'FontSize', 16)
ylim([1e-4 1e0])
lgd = legend('FCEKF', 'HCMCI', 'CCEKF');
lgd.FontSize = 16; % Adjust font size
lgd.Interpreter = 'latex'; % Set interpreter to latex

% Wait for this figure window to close before continuing
uiwait(gcf);