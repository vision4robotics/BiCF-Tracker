function [seq, ground_truth] = load_video_info(video_path)

ground_truth = dlmread([video_path '/groundtruth_rect.txt']);

seq.format = 'otb';
seq.len = size(ground_truth, 1);
seq.init_rect = ground_truth(1,:);

img_path = [video_path '/img/'];

if exist([img_path num2str(1, '%04i.png')], 'file')
    img_files = num2str((1:seq.len)', [img_path '%04i.png']);
elseif exist([img_path num2str(1, '%04i.jpg')], 'file')
    img_files = num2str((1:seq.len)', [img_path '%04i.jpg']);
elseif exist([img_path num2str(1, '%04i.bmp')], 'file')
    img_files = num2str((1:seq.len)', [img_path '%04i.bmp']);
elseif exist([img_path num2str(1, '%05i.png')], 'file')
    img_files = num2str((1:seq.len)', [img_path '%05i.png']);
elseif exist([img_path num2str(1, '%05i.jpg')], 'file')
    img_files = num2str((1:seq.len)', [img_path '%05i.jpg']);
elseif exist([img_path num2str(1, '%05i.bmp')], 'file')
    img_files = num2str((1:seq.len)', [img_path '%05i.bmp']);
elseif exist([img_path num2str(1, '%06i.png')], 'file')
    img_files = num2str((1:seq.len)', [img_path '%06i.png']);
elseif exist([img_path num2str(1, '%06i.jpg')], 'file')
    img_files = num2str((1:seq.len)', [img_path '%06i.jpg']);
elseif exist([img_path num2str(1, '%06i.bmp')], 'file')
    img_files = num2str((1:seq.len)', [img_path '%06i.bmp']);
elseif exist([img_path 'img' num2str(1, '%06i.png')], 'file')
    img_files = num2str((1:seq.len)', [img_path, 'img', '%06i.png']);
elseif exist([img_path 'img' num2str(1, '%06i.jpg')], 'file')
    img_files = num2str((1:seq.len)', [img_path, 'img', '%06i.jpg']);
elseif exist([img_path 'img' num2str(1, '%06i.bmp')], 'file')
    img_files = num2str((1:seq.len)', [img_path, 'img', '%06i.bmp']);
else
    error('No image files to load.')
end

seq.s_frames = cellstr(img_files);

end

