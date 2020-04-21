
% This demo script runs the BiCF tracker with hand-crafted features on the
% included "person11" video (UAV123@10fps dataset).

% Add paths
setup_paths();

% Load video information
video_path = './sequences/person11';
[seq, ground_truth] = load_video_info(video_path);

% Run ECO
results = run_BiCF(seq);

close all;