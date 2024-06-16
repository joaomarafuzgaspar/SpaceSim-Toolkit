clc
clear
close all

%% Test conversion from cartesian to keplerian orbital elements [IT WORKS!]

mu = 3.986004418e5; % EGM-96, km^3/sec^2

a = 6978;
e = 2.6e-6;
i = deg2rad(97.79);
p = deg2rad(303.34);
n = deg2rad(1.5e-5);
w = deg2rad(157.36);

KOE = struct;
KOE.sma = a;
KOE.ecc = e;
KOE.incl = i;
KOE.argp = p;
KOE.raan = n;
KOE.tran = w;

kep2cart(KOE, mu)

%% Initialization

x_initial = [
    -1295.584507660883, -929.374090790716, 6793.411681382508, -7.426511598857044, 0.1901977700449196, -1.390296386102339, -171.9053207863305, 6857.222886707187, -1281.649066658106, 1.00700996311517, -1.399928822038102, -7.358476293144188, -4251.965594108594, 1619.96360478974, -5291.008175332099, -4.678160885930184, 3.46354521932496, 4.820200886960163, -2214.904606485801, -3967.018695794123, -5296.066519714539, 1.361455163361585, 5.66586154474086, -4.813337608337905
];
x_0 = reshape(x_initial, 6, 4);

t = 0:60:360*60;

%% State propagation (perturbations)
epoch  = datenum('1 Jan 2000 12:00:00');
mass = 1; 
area_sqm = 0.01; 
cD = 2.22;

jOptions = odtbxOptions('force');
jOptions = setOdtbxOptions(jOptions, 'epoch', epoch); % datenum format

% Gravity
jOptions = setOdtbxOptions(jOptions, 'earthGravityModel', 'WGS84_EGM96');
jOptions = setOdtbxOptions(jOptions, 'mass', mass); % kg

% Oblateness J2
jOptions = setOdtbxOptions(jOptions, 'gravDeg', 2, 'gravOrder', 0); % max 20x20

% Atmospheric Drag
jOptions = setOdtbxOptions(jOptions, 'dragArea', area_sqm);
jOptions = setOdtbxOptions(jOptions, 'cD', 2.2);
jOptions = setOdtbxOptions(jOptions, 'atmosphereModel', 'HP');
jOptions = setOdtbxOptions(jOptions, 'useAtmosphericDrag', true);

% Create a java object that stores the default information for propagating Earth-centric orbits using JAT force models.
jatWorld = createJATWorld(jOptions);

eOpts = odtbxOptions('estimator');
eOpts = setOdtbxOptions(eOpts, 'ValidationCase', 0);
eOpts = setOdtbxOptions(eOpts, 'OdeSolver', @ode45);

fprintf('Starting orbit propagation...\n');
figure(4);
scatter3(x_0(1, :), x_0(2, :), x_0(3, :), 'MarkerEdgeColor', 'k', 'DisplayName', 'initial states');
hold on

x_nk = zeros(6, 4, length(t));
A_nk = zeros(length(t), 4, 6, 6); 
for i = 1:4
    fprintf('Propagating orbit for spacecraft %d...\n', i);
    [~, x] = integ(@jatForces_km, t, x_0(:, i), eOpts, jatWorld);
    for j = 1:length(t)
        [~, A, ~] = jatForces_km(t(j), x(:, j), jatWorld);
        A_nk(j, i, :, :) = A;
        if mod(j, 60) == 0
            fprintf('Spacecraft %d: processed %d/%d time steps...\n', i, j, length(t));
        end
    end
    plot3(x(1, :), x(2, :), x(3, :), 'DisplayName', sprintf('S/C %d', i), 'LineWidth', 2);
    hold on
    x_nk(:, i, :) = x;
end

xlabel('X (km)') 
ylabel('Y (km)') 
zlabel('Z (km)') 
grid on; grid minor;
legend

fprintf('Saving data to MAT files...\n');

chief = transpose(reshape(x_nk(1:3, 1, :), [3, 361]));
deputy1 = transpose(reshape(x_nk(1:3, 2, :), [3, 361]));
deputy2 = transpose(reshape(x_nk(1:3, 3, :), [3, 361]));
deputy3 = transpose(reshape(x_nk(1:3, 4, :), [3, 361]));
save('../data_odtbx/data_odtbx_x_true.mat', 'chief', 'deputy1', 'deputy2', 'deputy3');

chief = reshape(A_nk(:, 1, :, :), [361, 6, 6]);
deputy1 = reshape(A_nk(:, 2, :, :), [361, 6, 6]);
deputy2 = reshape(A_nk(:, 3, :, :), [361, 6, 6]);
deputy3 = reshape(A_nk(:, 4, :, :), [361, 6, 6]);
save('../data_odtbx/data_odtbx_F.mat', 'chief', 'deputy1', 'deputy2', 'deputy3');

fprintf('Data saved successfully.\n');