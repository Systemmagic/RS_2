%% Moving Horizon Hankel-DMD Forecasting
%Highway Traffic Dynamics: Data-Driven Analysis and Forecast 
%Allan M. Avila & Dr. Igor Mezic 2019
%University of California Santa Barbara
function [Prediction,Ptrue,MAE,MRE,RMSE,SAE,TMAE,SMAE,AvgTMAE,AvgSMAE]=...
          MovingHorizonHankelDMD2(data,max_f)
%% Load Data
clc; close all;
tic
disp('Loading Data Set...')
if strcmp(data,'Day_mean') % 检查输入参数data是否等于'Day_mean'
Data = readtable('国控日均_2023_2024_notitle.xlsx');% 从文件中读取日平均PM2.5数据
Data = table2array(Data);
Data = Data';
dtype='Mean'; %表示数据的统计类型为 “平均值”（Mean）
hwy='day'; %主要用于后续可视化标题（如title(['True Data ' hwy])）
%表示数据的时间间隔（单位由数据类型决定）
% 在计算特征频率（omega=log(diag(eigval))./delt）时用于将特征值转换为实际时间维度的演化速率，确保预测的时间尺度与原始数据一致。
delt=1;

[nbx,nbt]=size(Data); 
delt=1; 
min_s = 3;% 最小采样窗口大小（单位：步），对应 3×1=3天
min_f = min_s;% 最小预测窗口大小（单位：步），同样为3天
max_s=max_f;% 最大采样窗口大小 = 最大预测窗口大小（由输入参数 max_f 决定）
% 预测结果存储：单元格数组（存储变长结果），维度为 [max窗口大小 × max窗口大小]
Prediction{max(min_f,max_f),max(min_f,max_f)}=[]; 
% 误差指标存储：数值矩阵，维度同上
MAE=zeros(max(min_f,max_f),max(min_f,max_f)); 
MRE=zeros(max(min_f,max_f),max(min_f,max_f));
RMSE=zeros(max(min_f,max_f),max(min_f,max_f));
% 细分误差指标：单元格数组（存储变长结果）存储每个窗口组合的详细误差
SAE{max(min_f,max_f),max(min_f,max_f)}=[]; % 缩放绝对误差
TMAE{max(min_f,max_f),max(min_f,max_f)}=[]; % 时间平均绝对误差
SMAE{max(min_f,max_f),max(min_f,max_f)}=[]; % 空间平均绝对误差
% 平均误差指标：数值矩阵存储误差的平均值
AvgTMAE=zeros(max(min_f,max_f),max(min_f,max_f)); % 总平均时间绝对误差
AvgSMAE=zeros(max(min_f,max_f),max(min_f,max_f)); % 总平均空间绝对误差

%% Loop over Sampling and Forecasting Windows
disp('Generating Forecasts for Various Forecasting and Sampling Windows')
tic
for f=min_f:min_f:max_f % 遍历预测窗口大小（步长为min_f）
for s=min_s:min_s:max_s % 遍历采样窗口大小（步长为min_f）
%以下变量在每次滑动窗口循环前被初始化为空，目的是清除上一轮循环的残留数据，避免对当下产生干扰
P=[];%存储预测值（Prediction）
PE=[]; %猜测是“预测误差”（Prediction Error），但未在代码中出现
E=[];%存储误差矩阵，后续会通过 “预测值 - 真实值” 计算得到
I=[];%存储误差矩阵的总元素数量，用于计算平均误差时的分母（如MAE = sum(sum(abs(E))) ./ I）
R=[];%未在该代码中出现
Ptrue=[];%存储预测窗口对应的真实值（True Values）

for t=s:f:nbt-f % 
if mod(t,100)==0  % 如果t为100整数，执行，即每100步显示一次进度
disp(['Delay=' num2str(delay) ' Forecasting Window=' num2str(f)...
    ' Sampling Window=' num2str(s) ' Current Time=' num2str(t)...
    ' Out of ' num2str(nbt-f)])
end
% 变量清零（避免上一次迭代的残留数据）
omega=[]; %存储特征频率，（omega = log(diag(eigval)) ./ delt），模态的演化速度
eigval=[]; %特征值反映了系统动态模态的稳定性(虚部)和衰减 / 增长特性（实部）
Modes1=[]; %存储空间模态矩阵，列对应不同模态，行对应不同观测点，描述系统的空间分布特征
bo=[];  %初始系数向量，
Xdmd=[]; %存储 DMD 模型重构 / 预测的数据
Xtrain=[]; %存储当前窗口的训练数据（采样窗口数据）
det=[];  %存储训练窗口数据的时间平均值，用于数据预处理（Xtrain - det）以消除静态偏移

% 1. 提取训练数据和真实预测值
Xtrain=Data(:,t-s+1:t); % [nbx × s]的采样窗口数据
Xfor=Data(:,t+1:t+f); %  [nbx × f] 的真实值矩阵（后续评估误差）
% 2. 计算时间平均值（去趋势处理，增强模型对动态变化的捕捉能力）
det=mean(Xtrain,2);% 对每一行（空间维度）计算时间平均值
% 3. 设置延迟参数并执行Hankel-DMD
delay=min(s-1);% 最大可能延迟（由采样窗口大小决定）
[eigval,Modes1,bo] = H_DMD(Xtrain-det,delay); % 调用H-DMD核心函数（去除均值后的训练数据）
% 4. 计算特征频率和模态
omega=log(diag(eigval))./delt; % 累积数组
Modes1=Modes1(1:nbx,:);%只有前nbx行对应原始的空间维度（即监测点），其余行是延迟维度的信息，对最终预测无用。
% 5. 预测未来数据（并行计算加速）
parfor time=1:s+f %覆盖两个阶段（前s（评估），后f（预测））
Xdmd(:,time)=diag(exp(omega.*(time-1)))*bo;
end
% 6. 重构数据并恢复均值
Xdmd=Modes1*Xdmd; % 应用模态矩阵得到最终预测
Xdmd=real(Xdmd+det);% 取实部并加回之前减去的平均值
% 7. 存储预测结果和真实值
P=[P Xdmd(:,s+1:end)];% 只保留预测部分（排除训练窗口的重构数据）
Ptrue=[Ptrue Xfor];% 累积存储真实值
end % Window Sliding

Prediction{f,s}=P;% 存储当前窗口组合的完整预测结果
E=P-Data(:,s+1:s+size(P,2));% 计算误差矩阵(预测值 - 真实值)

% 计算全局误差指标（整个数据集的平均）
I=size(E,1)*size(E,2);% 误差矩阵的总元素数npx*f
MAE(f,s)=sum(sum(abs(E)))./I;% 平均绝对误差,所有预测误差的绝对值的平均值
MRE(f,s)=sum(sum(abs(E)./Data(:,s+1:s+size(P,2))))./I;% ✳平均相对误差,所有预测误差的相对值的平均值
RMSE(f,s)=sqrt(sum(sum(E.^2))./I);% 均方根误差，对异常值（大误差）更敏感，能反映误差的离散程度

% 计算细分误差指标
SAE{f,s}=abs(E)./mean(Data(:,s+1:s+size(P,2)),2);% 缩放绝对误差，每个预测误差的绝对值相对于 “空间维度均值” 的比例
TMAE{f,s}=mean(abs(E),1);% 时间平均绝对误差，
SMAE{f,s}=mean(abs(E),2);% 空间平均绝对误差，
AvgTMAE(f,s)=mean(TMAE{f,s});% Compute Avg of TMAE
AvgSMAE(f,s)=mean(SMAE{f,s});% Compute Avg of SMAE

end % End Sampling Window Loop
end % End Forecasting WIndow Loop
disp('Forecasts Generated')
toc
tic
disp('Generating Plots')
%-------------------------------------------------------------------------%
%-------------------------------------------------------------------------%
% 选择用于可视化的最优窗口 (依据误差最小，默认用 RMSE)
f_vals = min_f:min_f:max_f;
s_vals = min_s:min_s:max_s;
RMSE_grid = RMSE(f_vals, s_vals);
[~, best_linear_idx] = min(RMSE_grid(:));
[best_f_idx, best_s_idx] = ind2sub(size(RMSE_grid), best_linear_idx);
best_f = f_vals(best_f_idx);
best_s = s_vals(best_s_idx);

%% Plot Data, Forecasts and SAE
if strcmp(hwy,'day')
figure('units','normalized','outerposition',[0 0 1 1])
subplot(1,3,1)  % 激活1行3列布局中的第1个子图
contourf(Data,'Linestyle','none')  % 绘制填充轮廓图
xticks({}); yticks({});  % 隐藏x轴和y轴的刻度
h=colorbar; set(get(h,'title'),'string', {'PM2.5'});  % 设置颜色条标题
title(['True Data ' hwy])  % 设置子图标题

subplot(1,3,2)
contourf(Prediction{best_f,best_s},'Linestyle','none')
xticks({}); yticks({});
h=colorbar; set(get(h,'title'),'string', {'PM2.5'});
title(['Forecasted Data ' hwy])

subplot(1,3,3)
contourf(SAE{best_f,best_s},'Linestyle','none')
xticks({}); yticks({});
h=colorbar; set(get(h,'title'),'string', {'SAE'});
title(['Scaled Absolute Error ' hwy])

elseif ~strcmp(hwy,'day')
% 时间轴设置（根据日数据特点调整，例如按天或小时）
%d=0:minutes(5):hours(24);
%Time=datetime(1776,7,4)+d;
%Time.Format='MMMM d,yyyy HH:mm';
d=0:hours(24):(size(Data,2)-1)*hours(24);
Time=datetime(2023,1,1)+d;  % 从2023年1月1日开始的日期序列
Time.Format='yyyy-MM-dd';  % 日期格式：年-月-日
Time=timeofday(Time(1:end-1));
PlotMultiLaneNetwork(Ptrue,Prediction{best_f,best_s},Time,best_s,best_f,hwy)

end
%-------------------------------------------------------------------------%
%-------------------------------------------------------------------------%
%% Plot SMAE, TMAE and Histograms
figure('units','normalized','outerposition',[0 0 1 1])
subplot(3,1,1)
plot(SMAE{best_f,best_s})
hold on
plot(ones(1,length(SMAE{best_f,best_s})).*AvgSMAE(best_f,best_s),'m','Linewidth',2)
title([{hwy },{['SMAE for s=' num2str(best_s) ' f=' num2str(best_f)]}])
legend('SMAE','Avg of SMAE'); xticks({}); xlabel('Space'); axis('tight');

subplot(3,1,2)
plot(TMAE{best_f,best_s})
hold on
plot(ones(1,length(TMAE{best_f,best_s})).*AvgTMAE(best_f,best_s),'m','Linewidth',2)
title([{hwy },{['TMAE for s=' num2str(best_s) ' f=' num2str(best_f)]}])
legend('TMAE','Avg of TMAE'); xticks({}); xlabel('Time'); axis('tight');

% subplot(3,1,3)
% if ~strcmp(hwy,'LA Multi-lane Network')
% histogram(Data(:,min_s+1:min_s+size(Prediction{min_s,min_f},2)),...
%          'Normalization','pdf','Binwidth',.5); hold on
% histogram(Prediction{min_s,min_f},'Normalization','pdf','Binwidth',.5)
% title([{hwy },{['Histogram for s=' num2str(min_s) ' f=' num2str(min_f)]}])
% legend('PEMS','KMD'); axis('tight'); xlabel('MPH');
% 
% elseif strcmp(hwy,'LA Multi-lane Network')
% histogram(Data(:,min_s+1:min_s+size(Prediction{min_s,min_f},2)),...
%          'Normalization','pdf'); hold on
% histogram(Prediction{min_s,min_f},'Normalization','pdf')
% title([{hwy },{['Histogram for s=' num2str(min_s) ' f=' num2str(min_f)]}])
% legend('PEMS','KMD'); axis('tight'); xlabel('Vehicle Density')
% end
subplot(3,1,3)
if ~strcmp(hwy,'day')  % "非day"类型数据的直方图（保留原有逻辑，适用于其他场景）
    histogram(Data(:,best_s+1:best_s+size(Prediction{best_f,best_s},2)),...
              'Normalization','pdf','Binwidth',.5); hold on
    histogram(Prediction{best_f,best_s},'Normalization','pdf','Binwidth',.5)
    title([{hwy },{['Histogram for s=' num2str(best_s) ' f=' num2str(best_f)]}])
    legend('真实数据','预测数据');  
    axis('tight'); xlabel('数值');  % 通用数值标签，可根据数据类型调整
elseif strcmp(hwy,'day')  % 针对"day"类型数据的直方图
    % 绘制原始数据的概率密度直方图
    histogram(Data(:,best_s+1:best_s+size(Prediction{best_f,best_s},2)),...
              'Normalization','pdf', 'Binwidth', 1);  % 日数据 bins 设为1（如PM2.5浓度间隔1μg/m³）
    hold on
    % 绘制预测数据的概率密度直方图
    histogram(Prediction{best_f,best_s},'Normalization','pdf', 'Binwidth', 1);
    % 标题：包含场景和窗口参数
    title([{hwy },{['Histogram for s=' num2str(best_s) ' f=' num2str(best_f)]}])
    % 图例：明确区分真实值和预测值
    legend('真实日数据','预测日数据');
    axis('tight'); 
    xlabel('PM2.5浓度 (μg/m³)');  % 适配日均值数据的物理单位（可根据实际数据修改）
end
%-------------------------------------------------------------------------%
%-------------------------------------------------------------------------%
%% Plot MAE,MRE,RMSE
MAE=MAE(min_f:min_f:max_f,min_s:min_s:max_s);% Select NonZero Entries
MRE=MRE(min_f:min_f:max_f,min_s:min_s:max_s);% Select NonZero Entries
RMSE=RMSE(min_f:min_f:max_f,min_s:min_s:max_s);% Select NonZero Entries

if ~isscalar(MAE)%如果“MAE 不是标量（如仅计算了一组窗口组合的误差）”，执行以下代码块
figure('units','normalized','outerposition',[0 0 1 1])
subplot(1,3,1)
pcolor([[MAE MAE(:,end)];[MAE(end,:) MAE(end,end)]])
xticks([1.5 length(MAE)+.5]); xticklabels([delt*min_s,delt*max_s]);
yticks([1.5 length(MAE)+.5]); yticklabels([delt*min_s,delt*max_s]);
xlabel('Sampling Window [Min]')
ylabel('Forecasting Window [Min]')
h=colorbar;
set(get(h,'title'),'string', {'MAE'});
title(['MAE for ' hwy])

subplot(1,3,2)
pcolor([[MRE MRE(:,end)];[MRE(end,:) MRE(end,end)]])
xticks([1.5 length(MAE)+.5]); xticklabels([delt*min_s,delt*max_s]);
yticks([1.5 length(MAE)+.5]); yticklabels([delt*min_s,delt*max_s]);
xlabel('Sampling Window [Min]')
ylabel('Forecasting Window [Min]')
h=colorbar;
set(get(h,'title'),'string', {'MRE'});
title(['MRE for ' hwy])

subplot(1,3,3)
pcolor([[RMSE RMSE(:,end)];[RMSE(end,:) RMSE(end,end)]])
xticks([1.5 length(MAE)+.5]); xticklabels([delt*min_s,delt*max_s]);
yticks([1.5 length(MAE)+.5]); yticklabels([delt*min_s,delt*max_s]);
xlabel('Sampling Window [Min]')
ylabel('Forecasting Window [Min]')
h=colorbar;
set(get(h,'title'),'string', {'RMSE'});
axis('tight')
title(['RMSE for ' hwy])
else
disp('MAE, MRE & RMSE not Matrices')
disp(['MAE=' num2str(MAE) ' MRE=' num2str(MRE) ' RMSE=' num2str(RMSE)])
disp(['MAE=' num2str(MAE) ' MRE=' num2str(MRE) ' RMSE=' num2str(RMSE)])
end
end % End Function