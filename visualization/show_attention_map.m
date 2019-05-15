path = 'oxford/2015-06-09-15-06-29/pointcloud_20m/1433864320698910.bin';

file_n = strsplit(path,'/');

file_n = file_n(4);

fid = fopen(['/home/zwx/ZWX/benchmark_datasets/',path],'rb');

file = fread(fid,'double');
file = reshape(file,3,4096);

pc = pointCloud(file');
%pc4 = pointCloud(file4');
ax1=subplot(2,1,1);
pcshow(pc);

%fn = fullfile('/home/zwx/ZWX/pointnetvlad (3rd copy)/Oxford_Paper_Weights',file_n);
fn = fullfile('oxford_weights',file_n);

colorid = fopen(char(fn),'rb');
colorfile = fread(colorid,'single');
colorfile = reshape(colorfile,1,4096);
%colorfile = colorfile*100 - 99;
bot = min(colorfile);
top = max(colorfile);
value = top-bot;
color = (colorfile-bot)/value;

ax2=subplot(2,1,2);
pcshow(file',color);
colormap(ax2,jet)
%axis off;