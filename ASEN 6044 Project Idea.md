### ASEN 6044 Project Idea

#### 前言

Particle Filter， Gaussian Mixture Filter, Gaussian Sum Filter, 以及后面的各种filter，他们相较于经典的EKF/UKF， 最大的特点是能处理 non-smooth non-linear dynamics or measurements, non-Gaussian noise。 所以我们需要一个这样的动力学方程或测量方程，并添加非高斯的噪声。分工可以我们针对同一套动力学/测量方程用不同的filter，最后对比效果。

#### 背景

我有一套动力学方程是满足这个要求的，是ASEN 5010课上的旋转动力学方程。从这个方程出发，有一个可能的应用topic：**利用XXX filter估计人造卫星内部结构变化**。卫星在运行中可能遇到各种扰动，例如微小碎片撞击，太阳风等，可能会导致内部压力应力变化从而导致内部结构损坏，但地面观测站无法直接观测到卫星内部的结构变化。从地面观测站可以获得卫星的运动状态信息（姿态，range，range rate）这些，但卫星内部结构无法从观测中直接得到。但根据旋转动力学的欧拉方程，卫星内部结构/质量分布是和姿态/旋转相互耦合的，所以我们可以从姿态/旋转信息来估计卫星的内部结构/质量分布：
$$
\dot{\alpha}=\left[\begin{array}{c}
\dot{\phi} \\
\dot{\theta} \\
\dot{\psi}
\end{array}\right]=\frac{1}{\sin \theta}\left[\begin{array}{ccc}
\sin \psi & \cos \psi & 0 \\
\cos \psi \sin \theta & -\sin \psi \sin \theta & 0 \\
-\sin \psi \cos \theta & -\cos \psi \cos \theta & \sin \theta
\end{array}\right]\left[\begin{array}{c}
\omega_l \\
\omega_i \\
\omega_s
\end{array}\right] \\

\dot{\boldsymbol{\omega}}=\boldsymbol{I}^{-1}\left(-[\tilde{\omega}] \boldsymbol{I} \boldsymbol{\omega}+\boldsymbol{L}_{c m}\right)
$$
这个旋转动力学方程是高度非线性的，应该能满足这学期学的各种filter。其中$\alpha = [\phi,\theta,\psi]$ 是欧拉角，描述卫星的姿态，$\omega$是自转角速度，$L_{cm}$是地球引力带来的力矩。最后这个卫星内部结构/质量分布用刚体的惯量张量$I$来表示，如果卫星内部结构/质量分布没有变化，矩阵$I$的三个特征值会是一个常数（i.e.,无论卫星姿态如何变化，质量分布不变特征值就是固定的常数）。如果卫星内部结构发生变化，我们可以用一个阶跃函数来模拟这个变化

![function](/Users/hai/Desktop/PhD/Courses/ASEN_6044_Homework/final_project/function.jpeg)

这里既假设，在卫星受到扰动前，内部结构稳定没有变化，扰动后在短时间内进入一个新的稳定状态。所以我们可以设置状态量为，欧拉角，自转速度，质量分布，以及位置速度
$$
X = [\alpha,\omega,I,r,v]^T
$$
动力学方程用公式(1)，测量方程可以用个简单的，假设地面观测站直接能观测到欧拉角/姿态
$$
y = \alpha = [1,0,0,0,0]X
$$

#### 具体操作

1. 生成观测数据。利用上面的动力学方程，加上一些非高斯噪声，直接模拟观测数据
2. filter求解扰动前和扰动后的质量分布$I$
3. 比较不同filter的结果

