import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import os
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'Times New Roman']  # 宋体和Time New Roman
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
class draw_data:
    def __init__(self, **kwargs):
        self.loss_list = kwargs.get('loss_list', [])
        self.opt_gap_list = kwargs.get('optgap_list', [])
        self.psnr_list = kwargs.get('psnr_list', [])
        self.ssim_list = kwargs.get('ssim_list', [])
        self.acc_train_list = kwargs.get('acc_train_list', [])
        self.acc_test_list = kwargs.get('acc_test_list', [])

def get_data(path,name=''):
    if path.split('_')[0] == 'fmnist':
        loss = np.load(path+'/loss'+name+'.npy')
        optgap = np.load(path+'/opt_gap'+name+'.npy')
        psnr = np.load(path+'/psnr'+name+'.npy')
        ssim = np.load(path+'/ssim'+name+'.npy')
        return draw_data(loss_list=loss, optgap_list=optgap, psnr_list=psnr, ssim_list=ssim)
    else:
        loss = np.load(path+'/loss'+name+'.npy')
        optgap = np.load(path+'/opt_gap'+name+'.npy')
        acc_test  = np.load(path+'/acc_test'+name+'.npy')
        acc_train = np.load(path+'/acc_train'+name+'.npy')
        return draw_data(loss_list=loss, optgap_list=optgap, acc_test_list=acc_test, acc_train_list=acc_train)

dir = 'a9a_log'
AB_VRDAGT_Instance = get_data('../draw/'+dir,'AB_VRDAGT')
AB_DAGT_Instance = get_data('../draw/'+dir,'AB_DAGT')
AB_SAGA_Instance = get_data('../draw/'+dir,'AB_SAGA')
GT_VR_Instance = get_data('../draw/'+dir,'GT_VR')

# 绘制机器学习任务
marksize = 3.5
T = 6000
step = 4
index=range(0,T+T//100,T//100)

fig,ax = plt.subplots(2,1,figsize=(10,10))
ax[0].set_xlabel('迭代次数',fontsize=20)
ax[0].set_ylabel(r'f($\hat{x}$)',fontsize=20)
ax[0].plot(index[::step],AB_DAGT_Instance.loss_list[::step],'go-',label='DAGT',markersize=marksize)
ax[0].plot(index[::step],AB_SAGA_Instance.loss_list[::step],'rv-',label='AB-SAGA',markersize=marksize)
ax[0].plot(index[::step],GT_VR_Instance.loss_list[::step],'y*-',label='GT-VR',markersize=marksize)
ax[0].plot(index[::step],AB_VRDAGT_Instance.loss_list[::step],'b^-',label='VRDAGT',markersize=marksize)
ax[0].legend(loc='upper right',fontsize=15)
# 将坐标轴的字体大小设置为15
ax[0].tick_params(labelsize=15)
begin = -10
end = -2
#局部放大图
axins_loss = ax[0].inset_axes([0.5, 0.4, 0.2, 0.2])
axins_loss.plot(index[begin:end],AB_SAGA_Instance.loss_list[begin:end],'rv-',label='AB-SAGA',markersize=marksize,markevery=5)
axins_loss.plot(index[begin:end],GT_VR_Instance.loss_list[begin:end],'y*-',label='GT-VR',markersize=marksize,markevery=5)

# 设定放大区域
min_val = min(min(AB_SAGA_Instance.loss_list[begin:end]), min(GT_VR_Instance.loss_list[begin:end]))
max_val = max(max(AB_SAGA_Instance.loss_list[begin:end]), max(GT_VR_Instance.loss_list[begin:end]))

x1, x2, y1, y2 = index[begin], index[end-1], min_val, max_val
axins_loss.set_xlim(x1, x2)
axins_loss.set_ylim(y1, y2)
# # 添加连接线
mark_inset(ax[0], axins_loss, loc1=3, loc2=1, fc="none", ec='0.5')

ax[1].set_xlabel('迭代次数',fontsize=20)
ax[1].set_ylabel('最优化间隙',fontsize=20)
ax[1].plot(index[::step],AB_DAGT_Instance.opt_gap_list[::step],'go-',label='DAGT',markersize=marksize)
ax[1].plot(index[::step],AB_SAGA_Instance.opt_gap_list[::step],'rv-',label='AB-SAGA',markersize=marksize)
ax[1].plot(index[::step],GT_VR_Instance.opt_gap_list[::step],'y*-',label='GT-VR',markersize=marksize)
ax[1].plot(index[::step],AB_VRDAGT_Instance.opt_gap_list[::step],'b^-',label='VRDAGT',markersize=marksize)
ax[1].tick_params(labelsize=15)

ax[1].legend(fontsize=15,loc='upper right')
ax[0].axis([0, T, 0.15, 0.5+0.02])
# 设置y轴只显示5个点
max_1 = max(max(AB_DAGT_Instance.loss_list),max(GT_VR_Instance.loss_list),max(AB_SAGA_Instance.loss_list),max(AB_VRDAGT_Instance.loss_list))
min_1 = min(min(AB_DAGT_Instance.loss_list),min(GT_VR_Instance.loss_list),min(AB_SAGA_Instance.loss_list),min(AB_VRDAGT_Instance.loss_list))
point = np.arange(0.15, max_1+(max_1-min_1)/4, (max_1-min_1)/4)
ax[0].set_yticks([float('%.2f' % i) for i in point])
ax[1].axis([0, T, 0,0.022])
# 设置y轴只显示5个点
point = np.arange(0, 0.022, 0.005)
ax[1].set_yticks([float('%.2f' % i) for i in point])
plt.savefig('a9a_loss_optgap.png', dpi=600, bbox_inches='tight')
plt.show()