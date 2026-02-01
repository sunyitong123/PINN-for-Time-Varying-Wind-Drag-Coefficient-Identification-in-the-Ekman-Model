# %%
import sys
sys.path.insert(0, '../../Utilities/')

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
# from plotting import newfig, savefig
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import time

np.random.seed(1234)
tf.set_random_seed(1234)

class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, X, X_b,X_b1 , u, v, u_b, v_b, layers, lb, ub):
        self.lb = lb
        self.ub = ub
        self.x = X[:,0:1]
        self.t = X[:,1:2]
        self.x_b = X_b[:,0:1]
        self.t_b = X_b[:,1:2]
        self.x_b1 = X_b1[:,0:1]
        self.t_b1 = X_b1[:,1:2]
        self.u = u
        self.v = v
        self.u_b = u_b
        self.v_b = v_b
        self.layers = layers
        self.loss_u_hist = []
        self.loss_v_hist = []
        self.loss_fu_hist = []
        self.loss_fv_hist = []
        self.loss_ub_hist = []
        self.loss_vb_hist = []
        self.loss_ub_val_hist = []
        self.loss_vb_val_hist = []
        self.total_loss_hist = []
        self.epoch=[]
        self.niter=1500000
        # Initialize NNs
        self.weights, self.biases = self.initialize_NN(layers)
        
        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        # Initialize parameters
        self.lambda_1 = tf.exp(tf.Variable([-3.0], dtype=tf.float32))
        self.lambda_2 = tf.Variable([3.5], dtype=tf.float32)

        
        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]])
        self.t_tf = tf.placeholder(tf.float32, shape=[None, self.t.shape[1]])
        self.x_b_tf = tf.placeholder(tf.float32, shape=[None, self.x_b.shape[1]])
        self.t_b_tf = tf.placeholder(tf.float32, shape=[None, self.t_b.shape[1]])
        self.x_b_tf1 = tf.placeholder(tf.float32, shape=[None, self.x_b1.shape[1]])
        self.t_b_tf1 = tf.placeholder(tf.float32, shape=[None, self.t_b1.shape[1]])
        self.u_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])
        self.v_tf = tf.placeholder(tf.float32, shape=[None, self.v.shape[1]])
        self.bu_tf = tf.placeholder(tf.float32, shape=[None, self.u_b.shape[1]])
        self.bv_tf = tf.placeholder(tf.float32, shape=[None, self.v_b.shape[1]])
    
        # self.u_pred, self.v_pred, self.f_u_pred, self.f_v_pred, self.ub_pred, self.vb_pred, self.f_ub_pred, self.f_vb_pred = \
        #     self.net_ekman(self.x_tf, self.t_tf, self.x_b_tf, self.t_b_tf)
        (self.u_pred, self.v_pred, self.f_u_pred, self.f_v_pred,self.ub_pred, self.vb_pred, self.f_ub_pred, self.f_vb_pred,
        self.u_t, self.u_x, self.u_xx, self.v_t, self.v_x, self.v_xx, self.ub_x, self.vb_x) = \
    self.net_ekman(self.x_tf, self.t_tf, self.x_b_tf, self.t_b_tf,self.x_b_tf1, self.t_b_tf1)


        self.loss_u = tf.reduce_mean(tf.square(self.u_tf - self.u_pred))
        self.loss_v = tf.reduce_mean(tf.square(self.v_tf - self.v_pred))
        self.loss_fu = tf.reduce_mean(tf.square(self.f_u_pred))
        self.loss_fv = tf.reduce_mean(tf.square(self.f_v_pred))
        self.loss_ub = tf.reduce_mean(tf.square(self.f_ub_pred))
        self.loss_vb = tf.reduce_mean(tf.square(self.f_vb_pred))
        self.loss_ub_val = tf.reduce_mean(tf.square(self.bu_tf - self.ub_pred))
        self.loss_vb_val = tf.reduce_mean(tf.square(self.bv_tf - self.vb_pred))

        self.loss = self.loss_u + self.loss_v + self.loss_fu + self.loss_fv +self.loss_vb+self.loss_ub*5 +\
                    self.loss_ub_val+self.loss_vb_val

        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 50000,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        gradients, variables = zip(*optimizer.compute_gradients(self.loss))
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        self.train_op_Adam = optimizer.apply_gradients(zip(gradients, variables))
        # self.optimizer_Adam = tf.train.AdamOptimizer()
        # self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)
        
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):        
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)        
        return weights, biases
        
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
    
    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y
            
    def net_ekman(self, x, t, xb, tb, xb1, tb1):
        f=1e-4
        sa=5e-2
        D=np.sqrt(sa/f)
        lambda_1 = self.lambda_1
        lambda_2 = self.lambda_2 
        U = self.neural_net(tf.concat([x,t],1), self.weights, self.biases)
        u=U[:,0:1]
        v=U[:,1:2]
        u_t = tf.gradients(u, t)[0]
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        
        v_t = tf.gradients(v, t)[0]
        v_x = tf.gradients(v, x)[0]
        v_xx = tf.gradients(v_x, x)[0]
        
        f_u = u_t - v - lambda_1*u_xx
        f_v = v_t + u - lambda_1*v_xx
        
        U2 = self.neural_net(tf.concat([xb, tb], 1), self.weights, self.biases)
        ub = U2[:, 0:1]
        vb = U2[:, 1:2]
        
        U3 = self.neural_net(tf.concat([xb1, tb1], 1), self.weights, self.biases)
        ub1 = U3[:, 0:1]
        vb1 = U3[:, 1:2]

        ub_x = (ub1-ub)/(-1/D)
        vb_x = (vb1-vb)/(-1/D)
        b_u = ub_x - (lambda_2/ lambda_1)
        b_v = vb_x

        return u, v, f_u, f_v, ub, vb, b_u, b_v, u_t, u_x, u_xx, v_t, v_x, v_xx, ub_x, vb_x

    
    def callback(self, loss, lambda_1,lambda_2):
        lambda_1_value = lambda_1[0]  # 记录 L-BFGS 过程中的 lambda
        lambda_2_value = lambda_2[0]    
        self.lambda_1_history.append(lambda_1_value)
        self.lambda_2_history.append(lambda_2_value)
    
        self.total_loss_hist.append(loss)
        self.epoch.append(self.niter)
        print('epoch:%.1f,Loss: %e, l1: %.5f, l2: %.5f' % (self.niter,loss, lambda_1_value,lambda_2_value))
        self.niter +=1
               
    def train(self, nIter):
        tf_dict = {
            self.x_tf: self.x, self.t_tf: self.t,
            self.u_tf: self.u, self.v_tf: self.v,
            self.x_b_tf: self.x_b, self.t_b_tf: self.t_b,
            self.bu_tf: self.u_b, self.bv_tf: self.v_b,
            self.x_b_tf1: self.x_b1, self.t_b_tf1: self.t_b1,
        }
        start_time = time.time()
        self.lambda_1_history = []
        self.lambda_2_history = []
        
        
        for it in range(nIter): 
            self.sess.run(self.train_op_Adam, tf_dict)
            # Print
            if it % 10== 0:
                loss_value = self.sess.run(self.loss, tf_dict)
                loss_u_val = self.sess.run(self.loss_u, tf_dict)
                loss_v_val = self.sess.run(self.loss_v, tf_dict)
                loss_fu_val = self.sess.run(self.loss_fu, tf_dict)
                loss_fv_val = self.sess.run(self.loss_fv, tf_dict)
                loss_ub_val = self.sess.run(self.loss_ub, tf_dict)
                loss_vb_val = self.sess.run(self.loss_vb, tf_dict)
                loss_ub_val_val = self.sess.run(self.loss_ub_val, tf_dict)
                loss_vb_val_val = self.sess.run(self.loss_vb_val, tf_dict)

                self.loss_u_hist.append(loss_u_val)
                self.loss_v_hist.append(loss_v_val)
                self.loss_fu_hist.append(loss_fu_val)
                self.loss_fv_hist.append(loss_fv_val)
                self.loss_ub_hist.append(loss_ub_val)
                self.loss_vb_hist.append(loss_vb_val)
                self.loss_ub_val_hist.append(loss_ub_val_val)
                self.loss_vb_val_hist.append(loss_vb_val_val)
                self.total_loss_hist.append(loss_value)
                
                lambda_1_value = self.sess.run(self.lambda_1)[0]
                lambda_2_value = self.sess.run(self.lambda_2)[0]
                
                self.lambda_1_history.append(lambda_1_value)
                self.lambda_2_history.append(lambda_2_value)
                
            if it % 10== 0:
                elapsed = time.time() - start_time   
                print('It: %d, Loss: %.3e, Lambda_1: %.3f, Lambda_2: %.3f,  Time: %.2f' % 
                      (it, loss_value, lambda_1_value, lambda_2_value, elapsed))
                start_time = time.time()
        
        self.optimizer.minimize(self.sess,feed_dict = tf_dict,fetches = [self.loss, self.lambda_1, self.lambda_2] 
                                ,loss_callback = self.callback)#
                               
        np.savetxt("/data/sunyitong/反问题/A&C-基础网络/1.txt", np.array(self.lambda_1_history))
        np.savetxt("/data/sunyitong/反问题/A&C-基础网络/2.txt", np.array(self.lambda_2_history))
        
        np.savez('/data/sunyitong/反问题/A&C-基础网络/loss.npz',
         loss_u=np.array(self.loss_u_hist),
         loss_v=np.array(self.loss_v_hist),
         loss_fu=np.array(self.loss_fu_hist),
         loss_fv=np.array(self.loss_fv_hist),
         loss_ub=np.array(self.loss_ub_hist),
         loss_vb=np.array(self.loss_vb_hist),
         loss_ub_val=np.array(self.loss_ub_val_hist),
         loss_vb_val=np.array(self.loss_vb_val_hist),
         total_loss=np.array(self.total_loss_hist),
         epoch=np.array(self.epoch))
        print('Training finished.')
        self.saver = tf.train.Saver()
        self.saver.save(self.sess, "/data/sunyitong/反问题/A&C-基础网络/model_checkpoint/ekman_model.ckpt")
        print("Model saved.")
    def predict(self, X_star):
        
        tf_dict = {self.x_tf: X_star[:,0:1], self.t_tf: X_star[:,1:2]}
        
        u_star = self.sess.run(self.u_pred, tf_dict)
        v_star = self.sess.run(self.v_pred, tf_dict)
        f_u_star = self.sess.run(self.f_u_pred, tf_dict)
        f_v_star = self.sess.run(self.f_v_pred, tf_dict)
        
        return u_star,v_star, f_u_star,f_v_star

# %%
f=1e-4
rhoa=1.2
rho=1025
sa=5e-2
sc=1.2e-3
D=np.sqrt(sa/f)
u=rhoa*sc*100/(rho*np.sqrt(sa*f))
A=1e-3/sa

datau = np.load('/data/sunyitong/有限差分/u2.npz')
datav = np.load('/data/sunyitong/有限差分/v2.npz')
au=datau["u"]
au=au/u
av=datav["v"]
av=av/u
x=datau["z"]
t=datau["t"]
t=t[:]
t=t*f
x=x/D

# %%
if __name__ == "__main__": 
     
    A = A
    
    N_u = 2000
    layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 2]  
    X, T = np.meshgrid(x,t)
    X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
    uu_star = au.flatten()[:,None]
    vv_star = av.flatten()[:,None]
    # Doman bounds
    lb = X_star.min(0)
    ub = X_star.max(0)    
    
    N_b =5000
    X_b,T_b = np.meshgrid(x[0],t)
    X_star_b = np.hstack((X_b.flatten()[:,None], T_b.flatten()[:,None])) 
    u0_star=au[:,0].flatten()[:,None]
    v0_star=av[:,0].flatten()[:,None]
    
    
    X_b1,T_b1 = np.meshgrid(x[1],t)
    X_star_b1 = np.hstack((X_b1.flatten()[:,None], T_b1.flatten()[:,None])) 
    u0_star1=au[:,1].flatten()[:,None]
    v0_star1=av[:,1].flatten()[:,None]
    
    idx   = np.random.choice(X_star.shape[0], N_u, replace=False)
    idx_b = np.random.choice(X_star_b.shape[0], N_b, replace=False)
    X_u_train = X_star[idx,:]
    X_b_train = X_star_b[idx_b,:]
    X_b_train1 = X_star_b1[idx_b,:]
    u_train =uu_star[idx,:]
    v_train = vv_star[idx,:]
    ub_train=u0_star[idx_b,:]
    vb_train=v0_star[idx_b,:]
    
    
    model = PhysicsInformedNN(X_u_train,X_b_train, X_b_train1, u_train, v_train, ub_train, vb_train, layers, lb, ub)
    model.train(1000000)


# %%
if __name__ == "__main__": 
    taux=0.4/1.2/100
   
    u_pred,v_pred, f_u_pred,f_v_pred = model.predict(X_star)
    error_u = np.linalg.norm(uu_star-u_pred,2)/np.linalg.norm(uu_star,2)
    error_v = np.linalg.norm(vv_star-v_pred,2)/np.linalg.norm(vv_star,2)
    
#     U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')
        
    lambda_1_value = model.sess.run(model.lambda_1)
    lambda_1_value=lambda_1_value
    lambda_2_value = model.sess.run(model.lambda_2)
    lambda_2_value=lambda_2_value*1.2e-3
    error_lambda_1 = np.abs(lambda_1_value - A)/A*100
    error_lambda_2 = np.abs(lambda_2_value - taux)/taux*100
    
    
    print('Error u: %e' % (error_u))  
    print('Error v: %e' % (error_v))  
    print('Error l1: %.5f%%' % (error_lambda_1))
    print('Error l2: %.5f%%' % (error_lambda_2))                                             

# %%
lambda_1_history = np.loadtxt("/data/sunyitong/反问题/A&C-基础网络/1.txt")
lambda_2_history = np.loadtxt("/data/sunyitong/反问题/A&C-基础网络/2.txt")
fig, axes = plt.subplots(2, 1, figsize=(15, 5), dpi=150)
ax1 = axes[0]
ax1.plot( lambda_1_history, 'b-', label="Lambda_1")
ax2 = axes[1]
ax2.plot( lambda_2_history*1.2e-3, 'r-', label="Lambda_2")

# plt.xlabel("Iterations")
# plt.ylabel("Lambda Value")
# plt.title("Evolution of Lambda during Training")
# plt.legend()
# plt.grid()
plt.show()


# %%
u_pred=u_pred
v_pred=v_pred
np.savez('A&C111111', u=u_pred,v=v_pred)

# %%
u_pred=(u_pred.reshape(len(t), len(x)))
v_pred=(v_pred.reshape(len(t), len(x)))


# %%
ru=datau["u"]
rv=datav["v"]

# %%
u1=u_pred*u
v1=v_pred*u

# %%
u1[-1, :]-ru[-1,:]

# %%
import matplotlib.pyplot as plt
import numpy as np

# 示例数据

plt.figure(figsize=(8, 6))

# 第一个图：基于 au 绘制
for i in range(0, 32, 1):  # 每隔1个深度点绘制一次
    plt.quiver(0, -i, u1[-1, i], v1[-1, i], angles='xy', scale_units='xy', scale=1, color='b',label='T')

# 第二个图：基于 u_pred 和 v_pred 绘制
for i in range(0, 32, 1):  # 每隔1个深度点绘制一次
    plt.quiver(0, -i, ru.T[i, -1], rv.T[i, -1], angles='xy', scale_units='xy', scale=1, color='r',label='P')

plt.xlabel('East-West Velocity (m/s)')
plt.ylabel('Depth (m)')
plt.title('Ekman Spiral (Northern Hemisphere)')
plt.xlim(-1.5, 1.5)   # 设置横坐标范围
plt.ylim(-32, 1.5)   # 设置纵坐标范围
plt.grid(True)
plt.show()

# %%
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
fig, axes = plt.subplots(1, 2, figsize=(15, 3), dpi=150)  # 创建1行2列的子图

# 左图
ax1 = axes[0]
h1 = ax1.imshow(np.flipud(ru.T[:, :]), interpolation='nearest', cmap='rainbow', 
                 extent=[0, 8, 32, 0], 
                 origin='lower', aspect='auto',
                 vmin=-0.6, vmax=1.0)
divider1 = make_axes_locatable(ax1)
cax1 = divider1.append_axes("right", size="5%", pad=0.10)
cbar1 = fig.colorbar(h1, cax=cax1)
cbar1.ax.tick_params(labelsize=5)
ax1.set_xlabel('$t$', size=20)
ax1.set_ylabel('$z$', size=20)
ax1.tick_params(labelsize=15)

# 右图
ax2 = axes[1]
h2 = ax2.imshow(np.flipud(u1.T), interpolation='nearest', cmap='rainbow', 
                 extent=[0, 8, 32, 0], 
                 origin='lower', aspect='auto',
                 vmin=-0.6, vmax=1.0)
divider2 = make_axes_locatable(ax2)
cax2 = divider2.append_axes("right", size="5%", pad=0.10)
cbar2 = fig.colorbar(h2, cax=cax2)
cbar2.ax.tick_params(labelsize=5)
ax2.set_xlabel('$t$', size=20)
ax2.set_ylabel('$z$', size=20)
ax2.tick_params(labelsize=15)
# 由于没有需要图例的标签，因此无需调用 ax.legend()
# 显示图像
plt.tight_layout()
plt.show()


# %%
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
fig, axes = plt.subplots(1, 2, figsize=(15, 3), dpi=50)  # 创建1行2列的子图

# 左图
ax1 = axes[0]
h1 = ax1.imshow(np.flipud(rv.T[:, :]), interpolation='nearest', cmap='rainbow', 
                 extent=[0, 8, 32, 0], 
                 origin='lower', aspect='auto',
                 vmin=-0.6, vmax=1.0)
divider1 = make_axes_locatable(ax1)
cax1 = divider1.append_axes("right", size="5%", pad=0.10)
cbar1 = fig.colorbar(h1, cax=cax1)
cbar1.ax.tick_params(labelsize=5)
ax1.set_xlabel('$t$', size=20)
ax1.set_ylabel('$z$', size=20)
ax1.tick_params(labelsize=15)

# 右图
ax2 = axes[1]
h2 = ax2.imshow(np.flipud(v1.T), interpolation='nearest', cmap='rainbow', 
                 extent=[0, 8, 32, 0], 
                 origin='lower', aspect='auto',
                 vmin=-0.6, vmax=1.0)
divider2 = make_axes_locatable(ax2)
cax2 = divider2.append_axes("right", size="5%", pad=0.10)
cbar2 = fig.colorbar(h2, cax=cax2)
cbar2.ax.tick_params(labelsize=5)
ax2.set_xlabel('$t$', size=20)
ax2.set_ylabel('$z$', size=20)
ax2.tick_params(labelsize=15)
# 由于没有需要图例的标签，因此无需调用 ax.legend()
# 显示图像
plt.tight_layout()
plt.show()


# %%

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
# 图形初始化
fig = plt.figure(figsize=(12, 8),dpi=80)
ax = fig.add_subplot(111, projection='3d')

# 颜色映射
colors = plt.cm.jet(np.linspace(0, 1, len(x)))

# 绘制原始箭头
for i in range(0, len(x), 1):  # 每隔5个深度点绘制一次
    ax.quiver(0, 0, -x[i]*D, u1.T[ i,-1], v1.T[i,-1], 0, 
              arrow_length_ratio=0.05, linewidth=1.2, color=colors[len(x)-1-i])

#添加黑色的竖线 (从 z=0 到 z=-40)
ax.plot([0, 0], [0, 0], [0, -len(x)], color='k', linestyle='-', linewidth=2)

# 添加箭头映射到 z=-40 平面，并画出映射
for i in range(0, len(x), 1):  # 每隔5个深度点映射到底面
    ax.quiver(0, 0, -len(x), u1.T[i,-1], v1.T[i,-1], 0, color=colors[len(x)-1-i], 
              arrow_length_ratio=0.05, linewidth=0.7)

# 添加虚线连接箭头顶点到 z=-40 平面的映射
# for i in range(0, len(d), 1):
#     ax.plot([u[i], u[i]], [v[i], v[i]], [-depth[i], -len(d)-1], 
#             linestyle='--', color='gray', linewidth=0.7,alpha=0.7)
# ax.quiver(0, 0, 0, 0.3, 0, 0, color='k', arrow_length_ratio=0.1, linewidth=1)
ax.xaxis._axinfo['grid']['color'] = (0.8, 0.8, 0.8, 0.3)  # X轴网格浅灰色，透明度 0.3
ax.yaxis._axinfo['grid']['color'] = (0.8, 0.8, 0.8, 0.3)  # Y轴网格浅灰色，透明度 0.3
ax.zaxis._axinfo['grid']['color'] = (0.8, 0.8, 0.8, 0.3)  # Z轴网格浅灰色，透明度 0.3


# 设置坐标范围
ax.set_xlim(1.5, -0.8)  # u轴范围
ax.set_ylim(0.8,-1.2)  # v轴范围
ax.set_zlim(-len(x), 0)      # 深度轴范围（z轴）

# 轴标签
ax.set_xlabel('u [m/s]')
ax.set_ylabel('v [m/s]')
ax.set_zlabel('h [m]')

# 设置视角
ax.view_init(elev=25, azim=35)

# 颜色条
sm = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(vmin=-len(x), vmax=0))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, shrink=0.5)
cbar.set_label('h (m)')

# 显示图形
plt.show()


# %%

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
# 图形初始化
fig = plt.figure(figsize=(12, 8),dpi=80)
ax = fig.add_subplot(111, projection='3d')

# 颜色映射
colors = plt.cm.jet(np.linspace(0, 1, len(x)))

# 绘制原始箭头
for i in range(0, len(x), 1):  # 每隔5个深度点绘制一次
    ax.quiver(0, 0, -x[i]*D, ru.T[ i,-1], rv.T[i,-1], 0, 
              arrow_length_ratio=0.05, linewidth=1.2, color=colors[len(x)-1-i])

#添加黑色的竖线 (从 z=0 到 z=-40)
ax.plot([0, 0], [0, 0], [0, -len(x)], color='k', linestyle='-', linewidth=2)

# 添加箭头映射到 z=-40 平面，并画出映射
for i in range(0, len(x), 1):  # 每隔5个深度点映射到底面
    ax.quiver(0, 0, -len(x), ru.T[i,-1], rv.T[i,-1], 0, color=colors[len(x)-1-i], 
              arrow_length_ratio=0.05, linewidth=0.7)

# 添加虚线连接箭头顶点到 z=-40 平面的映射
# for i in range(0, len(d), 1):
#     ax.plot([u[i], u[i]], [v[i], v[i]], [-depth[i], -len(d)-1], 
#             linestyle='--', color='gray', linewidth=0.7,alpha=0.7)
# ax.quiver(0, 0, 0, 0.3, 0, 0, color='k', arrow_length_ratio=0.1, linewidth=1)
ax.xaxis._axinfo['grid']['color'] = (0.8, 0.8, 0.8, 0.3)  # X轴网格浅灰色，透明度 0.3
ax.yaxis._axinfo['grid']['color'] = (0.8, 0.8, 0.8, 0.3)  # Y轴网格浅灰色，透明度 0.3
ax.zaxis._axinfo['grid']['color'] = (0.8, 0.8, 0.8, 0.3)  # Z轴网格浅灰色，透明度 0.3


# 设置坐标范围
ax.set_xlim(1.5, -0.8)  # u轴范围
ax.set_ylim(0.8,-1.2)  # v轴范围
ax.set_zlim(-len(x), 0)      # 深度轴范围（z轴）

# 轴标签
ax.set_xlabel('u [m/s]')
ax.set_ylabel('v [m/s]')
ax.set_zlabel('h [m]')

# 设置视角
ax.view_init(elev=25, azim=35)

# 颜色条
sm = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(vmin=-len(x), vmax=0))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, shrink=0.5)
cbar.set_label('h (m)')

# 显示图形
plt.show()


# %%
import matplotlib as mpl
mpl.rcParams['text.usetex'] = False
fig, axes = plt.subplots(1, 2, figsize=(15, 3), dpi=50)  # 创建1行2列的子图
# 左图
ax1 = axes[0]
ax1.plot(v1[1,:],-x*D,'r--', linewidth=2, label='Prediction')
ax1.plot(rv[1,:],-x*D,'b--', linewidth=2, label='ture')
ax1.set_title('$v:t = 0$', fontsize=15)
# plt.xlabel('U', fontsize=15)
# plt.ylabel('x', fontsize=15)
ax1.legend()  # 显示图例
#plt.xlim(-0.1, 0.5)

ax2 = axes[1]
ax2.plot(u1[1,:],-x*D,'r--', linewidth=2, label='Prediction')
ax2.plot(ru[1,:],-x*D,'b--', linewidth=2, label='ture')
ax2.set_title('$u:t = 0$', fontsize=15) 
# # plt.savefig("三阶1", bbox_inches='tight',dpi=300)
plt.show

# %%
import matplotlib as mpl
mpl.rcParams['text.usetex'] = False
fig, axes = plt.subplots(1, 2, figsize=(15, 3), dpi=50)  # 创建1行2列的子图
# 左图
ax1 = axes[0]
ax1.plot(v1[-10,:],-x*D,'r--', linewidth=2, label='Prediction')
ax1.plot(rv[-10,:],-x*D,'b--', linewidth=2, label='ture')
ax1.set_title('$v:t = -1$', fontsize=15)
# plt.xlabel('U', fontsize=15)
# plt.ylabel('x', fontsize=15)
ax1.legend()  # 显示图例
#plt.xlim(-0.1, 0.5)

ax2 = axes[1]
ax2.plot(u1[-1,:],-x*D,'r--', linewidth=2, label='Prediction')
ax2.plot(ru[-1,:],-x*D,'b--', linewidth=2, label='ture')
ax2.set_title('$u:t = -1$', fontsize=15) 
# # plt.savefig("三阶1", bbox_inches='tight',dpi=300)
plt.show

# %%
import matplotlib as mpl
mpl.rcParams['text.usetex'] = False
fig, axes = plt.subplots(1, 4, figsize=(15, 3), dpi=50)  # 创建1行2列的子图
# 左图
ax1 = axes[0]
ax1.plot(v1[:,0],'r--', linewidth=2, label='Prediction')
ax1.plot(rv[:,0],'b--', linewidth=2, label='ture')
ax1.set_title('$v:z = 0$', fontsize=15)
# plt.xlabel('U', fontsize=15)
# plt.ylabel('x', fontsize=15)
ax1.legend()  # 显示图例
#plt.xlim(-0.1, 0.5)

ax2 = axes[1]
ax2.plot(v1[:,10],'r--', linewidth=2, label='Prediction')
ax2.plot(rv[:,10],'b--', linewidth=2, label='ture')
ax2.set_title('$v:z = 10$', fontsize=15)
ax3 = axes[2]
ax3.plot(v1[:,20],'r--', linewidth=2, label='Prediction')
ax3.plot(rv[:,20],'b--', linewidth=2, label='ture')
ax3.set_title('$v:z = 20$', fontsize=15)
ax4 = axes[3]
ax4.plot(v1[:,30],'r--', linewidth=2, label='Prediction')
ax4.plot(rv[:,30],'b--', linewidth=2, label='ture')
ax4.set_title('$v:z = 30$', fontsize=15)
# # plt.savefig("三阶1", bbox_inches='tight',dpi=300)
plt.show

# %%
import matplotlib as mpl
mpl.rcParams['text.usetex'] = False
fig, axes = plt.subplots(1, 4, figsize=(15, 3), dpi=50)  # 创建1行2列的子图
# 左图
ax1 = axes[0]
ax1.plot(u1[:,0],'r--', linewidth=2, label='Prediction')
ax1.plot(ru[:,0],'b--', linewidth=2, label='ture')
ax1.set_title('$u:z = 0$', fontsize=15)
# plt.xlabel('U', fontsize=15)
# plt.ylabel('x', fontsize=15)
ax1.legend()  # 显示图例
#plt.xlim(-0.1, 0.5)

ax2 = axes[1]
ax2.plot(u1[:,10],'r--', linewidth=2, label='Prediction')
ax2.plot(ru[:,10],'b--', linewidth=2, label='ture')
ax2.set_title('$u:z = 10$', fontsize=15)
ax3 = axes[2]
ax3.plot(u1[:,20],'r--', linewidth=2, label='Prediction')
ax3.plot(ru[:,20],'b--', linewidth=2, label='ture')
ax3.set_title('$u:z = 20$', fontsize=15)
ax4 = axes[3]
ax4.plot(u1[:,30],'r--', linewidth=2, label='Prediction')
ax4.plot(ru[:,30],'b--', linewidth=2, label='ture')
ax4.set_title('$u:z = 30$', fontsize=15)
# # plt.savefig("三阶1", bbox_inches='tight',dpi=300)
plt.show

# %%


# %%



