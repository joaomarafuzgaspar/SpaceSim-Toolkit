function visualizer_orbits(data_filepath, formation)
save_video = 1;
data = load(data_filepath);

% Simulation parameters
dt = 60.0; % Time step [s]
T = size(data.X, 2); % Duration [min]
t = 0:dt:dt * T; % Time vector [s]

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

% Create the VideoWriter object to write the video
if save_video
    v = VideoWriter(['videos/orbits_form' num2str(formation)], 'MPEG-4'); % Specify the 'MPEG-4' profile
    v.Quality = 100; % Optional: set video quality. 0 (worst) - 100 (best)
    open(v);
end

% Plot initial orbits with only the first point
h_orbit1 = plot3(data.X(1, 1), data.X(2, 1), data.X(3, 1), 'Color', [0, 0.4470, 0.7410], 'LineWidth', 1.5);
h_orbit2 = plot3(data.X(7, 1), data.X(8, 1), data.X(9, 1), 'Color', [0.8500 0.3250 0.0980], 'LineWidth', 1.5);
h_orbit3 = plot3(data.X(13, 1), data.X(14, 1), data.X(15, 1), 'Color', [0.9290 0.6940 0.1250], 'LineWidth', 1.5);
h_orbit4 = plot3(data.X(19, 1), data.X(20, 1), data.X(21, 1), 'Color', [0.4940 0.1840 0.5560], 'LineWidth', 1.5);

% Plot initial positions
h_satellite1 = scatter3(data.X(1, 1), data.X(2, 1), data.X(3, 1), 100, 'MarkerEdgeColor', 'k', 'MarkerFaceColor', [0, 0.4470, 0.7410]);
h_satellite2 = scatter3(data.X(7, 1), data.X(8, 1), data.X(9, 1), 100, 'MarkerEdgeColor', 'k', 'MarkerFaceColor', [0.8500 0.3250 0.0980]);
h_satellite3 = scatter3(data.X(13, 1), data.X(14, 1), data.X(15, 1), 100, 'MarkerEdgeColor', 'k', 'MarkerFaceColor', [0.9290 0.6940 0.1250]);
h_satellite4 = scatter3(data.X(19, 1), data.X(20, 1), data.X(21, 1), 100, 'MarkerEdgeColor', 'k', 'MarkerFaceColor', [0.4940 0.1840 0.5560]);

% Add legend, and format ticks in LaTeX
lgd = legend([h_orbit1, h_orbit2, h_orbit3, h_orbit4], {'$\mathcal{C}_1$', '$\mathcal{D}_1$', '$\mathcal{D}_2$', '$\mathcal{D}_3$'}, 'Interpreter', 'latex');
lgd.FontSize = 16; % Adjust font size
lgd.Interpreter = 'latex'; % Set interpreter to latex
lgd.Position(1) = 0.711666666666669;
lgd.Position(2) = 0.754761904761905;

% Set axis properties
axis equal;
grid on;
ax = gca;
ax.TickLabelInterpreter = "latex";
ax.FontSize = 14;
ax.XTick = ax.YTick;
xticklabels(strrep(xticklabels, '-', '$-$'));
yticklabels(strrep(yticklabels, '-', '$-$'));
zticklabels(strrep(zticklabels, '-', '$-$'));
ax.XDir = 'reverse';
ax.YDir = 'reverse';

% Set axis limits
ax.XLim = [-6.5e6 6.5e6];
ax.YLim = [-6.5e6 6.5e6];
ax.ZLim = [-6.5e6 6.5e6];

% Set isometric view
view(-45, 35.264);

% Add labels
xlabel('$x$ [$\times 10^6$ m]', 'Interpreter', 'latex', 'FontSize', 16);
ylabel('$y$ [$\times 10^6$ m]', 'Interpreter', 'latex', 'FontSize', 16);
zlabel('$z$ [$\times 10^6$ m]', 'Interpreter', 'latex', 'FontSize', 16);
ax.XLabel.Position = [-236831.4959593415, 7607883.297623575, -7371074.998906076];
ax.XLabel.Rotation = 30;
ax.YLabel.Position = [7679844.607475638, -281788.9704890251, -7398079.230938435];
ax.YLabel.Rotation = -30;

% Create the annotation for the time display
time_annotation = annotation('textbox', [0.380770174662274, 0.843015873242938, 0.274173936389741, 0.042539682085552], 'String', '', 'FitBoxToText', 'on', 'EdgeColor', 'k', 'BackgroundColor', 'white', 'FontSize', 14, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'Interpreter', 'latex');

% Animation loop
max_points = 50; % Maximum number of points to display
for k = 1:T
    % Determine the range of points to display
    start_idx = max(1, k - max_points + 1);
    range = start_idx:k;
    
    % Update the position of the orbits
    set(h_orbit1, 'XData', data.X(1, range), 'YData', data.X(2, range), 'ZData', data.X(3, range));
    set(h_orbit2, 'XData', data.X(7, range), 'YData', data.X(8, range), 'ZData', data.X(9, range));
    set(h_orbit3, 'XData', data.X(13, range), 'YData', data.X(14, range), 'ZData', data.X(15, range));
    set(h_orbit4, 'XData', data.X(19, range), 'YData', data.X(20, range), 'ZData', data.X(21, range));
    
    % Update the position of the satellites
    set(h_satellite1, 'XData', data.X(1, k), 'YData', data.X(2, k), 'ZData', data.X(3, k));
    set(h_satellite2, 'XData', data.X(7, k), 'YData', data.X(8, k), 'ZData', data.X(9, k));
    set(h_satellite3, 'XData', data.X(13, k), 'YData', data.X(14, k), 'ZData', data.X(15, k));
    set(h_satellite4, 'XData', data.X(19, k), 'YData', data.X(20, k), 'ZData', data.X(21, k));
    
    % Update the time display
    elapsed_time_seconds = t(k) / dt * 60; % Convert time to seconds
    
    % Create a duration object
    elapsed_duration = seconds(elapsed_time_seconds);
    
    % Extract hours, minutes, and seconds with fractional part
    total_seconds = 12 * 3600 + seconds(elapsed_duration);
    hours = floor(total_seconds / 3600);
    remaining_seconds = total_seconds - hours * 3600;
    minutes = floor(remaining_seconds / 60);
    seconds_to_print = remaining_seconds - minutes * 60;
    
    % Format the time string with fractional seconds
    time_string = sprintf('%02d:%02d:%02d', hours, minutes, seconds_to_print);
    
    set(time_annotation, 'String', sprintf('Epoch: 01/01/2000 %s TDB', time_string));
    
    drawnow; % Refresh the plot
    
    % Capture the current frame and write it to the video
    if save_video
        frame = getframe(gcf);
        writeVideo(v, frame);
    end
    
    pause(0.01); % Pause for 0.01 seconds before updating to the next position
end

hold off;

% Close the video file
if save_video
    close(v);
end

% Wait for this figure window to close before continuing
uiwait(gcf);
end

% To convert the videos to GIF, use the following commands:
% ffmpeg -i videos/orbits_form1.mp4 -vf "crop=in_w-200:in_h:100:0,fps=30,scale=960:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=256:reserve_transparent=true[p];[s1][p]paletteuse=dither=sierra2_4a" gifs/orbits_form1.gif
% ffmpeg -i videos/orbits_form2.mp4 -vf "crop=in_w-200:in_h:100:0,fps=30,scale=960:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=256:reserve_transparent=true[p];[s1][p]paletteuse=dither=sierra2_4a" gifs/orbits_form2.gif