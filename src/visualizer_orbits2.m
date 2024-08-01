% Read the data
data = readtable('data/data_x_true.csv');

% Convert units from km to m
data.x_chief = data.x_chief * 1e3;
data.y_chief = data.y_chief * 1e3;
data.z_chief = data.z_chief * 1e3;
data.x_deputy1 = data.x_deputy1 * 1e3;
data.y_deputy1 = data.y_deputy1 * 1e3;
data.z_deputy1 = data.z_deputy1 * 1e3;
data.x_deputy2 = data.x_deputy2 * 1e3;
data.y_deputy2 = data.y_deputy2 * 1e3;
data.z_deputy2 = data.z_deputy2 * 1e3;
data.x_deputy3 = data.x_deputy3 * 1e3;
data.y_deputy3 = data.y_deputy3 * 1e3;
data.z_deputy3 = data.z_deputy3 * 1e3;

% Simulation parameters
dt = 60.0; % Time step [s]
T = length(data.x_chief); % Duration [min]
t = 0:dt:dt * (T - 1); % Time vector [s]

% Set the specific time index to plot
time_index = 70;
max_points = 50;
start_idx = max(1, time_index - max_points + 1);
range = start_idx:time_index;

% Get the screen size
screenSize = get(0, 'ScreenSize');
screenWidth = screenSize(3);
screenHeight = screenSize(4);

% Create figure and increase size
factor = 1.5;
figure_width = 560 * factor;
figure_height = 420 * factor;
figure('Position', [(screenWidth - figure_width) / 2, (screenHeight - figure_height) / 2, figure_width, figure_height]);
opts.FaceAlpha = 0.4;
hold on;
planet3D('Earth', opts);

% Plot the orbits up to the specific time index
orbit1 = plot3(data.x_chief(range), data.y_chief(range), data.z_chief(range), 'Color', [0, 0.4470, 0.7410], 'LineWidth', 1.5);
orbit2 = plot3(data.x_deputy1(range), data.y_deputy1(range), data.z_deputy1(range), 'Color', [0.8500 0.3250 0.0980], 'LineWidth', 1.5);
orbit3 = plot3(data.x_deputy2(range), data.y_deputy2(range), data.z_deputy2(range), 'Color', [0.9290 0.6940 0.1250], 'LineWidth', 1.5);
orbit4 = plot3(data.x_deputy3(range), data.y_deputy3(range), data.z_deputy3(range), 'Color', [0.4940 0.1840 0.5560], 'LineWidth', 1.5);

% Plot the positions at the specific time index
scatter3(data.x_chief(time_index), data.y_chief(time_index), data.z_chief(time_index), 100, 'MarkerEdgeColor', 'k', 'MarkerFaceColor', [0, 0.4470, 0.7410]);
scatter3(data.x_deputy1(time_index), data.y_deputy1(time_index), data.z_deputy1(time_index), 100, 'MarkerEdgeColor', 'k', 'MarkerFaceColor', [0.8500 0.3250 0.0980]);
scatter3(data.x_deputy2(time_index), data.y_deputy2(time_index), data.z_deputy2(time_index), 100, 'MarkerEdgeColor', 'k', 'MarkerFaceColor', [0.9290 0.6940 0.1250]);
scatter3(data.x_deputy3(time_index), data.y_deputy3(time_index), data.z_deputy3(time_index), 100, 'MarkerEdgeColor', 'k', 'MarkerFaceColor', [0.4940 0.1840 0.5560]);

% Add labels, legend, and format ticks in LaTeX
xlabel('$x$ [m]', 'Interpreter', 'latex', 'FontSize', 16);
ylabel('$y$ [m]', 'Interpreter', 'latex', 'FontSize', 16);
zlabel('$z$ [m]', 'Interpreter', 'latex', 'FontSize', 16);
lgd = legend([orbit1, orbit2, orbit3, orbit4], {'$\mathcal{C}_1$', '$\mathcal{D}_1$', '$\mathcal{D}_2$', '$\mathcal{D}_3$'}, 'Interpreter', 'latex');
lgd.FontSize = 16; % Adjust font size
lgd.Interpreter = 'latex'; % Set interpreter to latex
lgd.Position(1) = 0.721190476190476;
lgd.Position(2) = 0.778571428571429;

% Set axis properties
axis equal;
grid on;
ax = gca;
ax.TickLabelInterpreter = "latex";
ax.FontSize = 14;
ax.XAxis.Exponent = 6;
ax.YAxis.Exponent = 6;
ax.ZAxis.Exponent = 6;
ax.XAxis.TickLabelsMode = 'auto';
ax.YAxis.TickLabelsMode = 'auto';
ax.ZAxis.TickLabelsMode = 'auto';
xticklabels(strrep(xticklabels, '-', '$-$'));
yticklabels(strrep(yticklabels, '-', '$-$'));
zticklabels(strrep(zticklabels, '-', '$-$'));
ax.XDir = 'reverse';
ax.YDir = 'reverse';

% Set axis limits
ax.XLim = [-6.5e6 6.5e6];
ax.YLim = [-6.5e6 6.5e6];
ax.ZLim = [-6.5e6 6.5e6];

% Set renderer to painters before saving
set(gcf, 'Renderer', 'painters');

% Save the figure as an SVG file
print(gcf, '-painters', '-dsvg', '/Users/joaogaspar/Desktop/test.svg');

plot2svg('sphere.svg');
