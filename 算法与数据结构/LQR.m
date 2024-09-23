//定义系统的矩阵
A = [0 1; 0 0];
// 计算A的维度
[n, m] = size(A);
// 定义输入矩阵
B = [0; 1];
// 计算B的维度
[p, q] = size(B);


// 离散化时间步长
dt = 0.1;

def c2d(A, B, dt):
    // 离散化状态矩阵
    Ad = expm(A*dt);
    // 离散化输入矩阵
    Bd = Ad * B;
    return Ad, Bd
Ad, Bd = c2d(A, B, dt);


// 系统初始化
x0 = [1; 0];
//输入初始化
u0 = 0;
// 定义系统运行步数
N = 100;
// x_history用于存储系统状态
x_history = zeros(n, N);
// u_history用于存储输入
u_history = zeros(p, N);
// 初始化系统状态
x = x0;
// 初始化输入
u = u0;
// 权重矩阵
Q = eye(n);
S = eye(n);
R = 1;


// 系统仿真
for i = 1:N
    // 计算控制输入
    u = -inv(R + B'*S*B)*B'*S*Ad*x;
    // 记录系统状态
    x_history(:, i) = x;
    // 记录输入
    u_history(:, i) = u;
    // 更新系统状态
    x = Ad*x + Bd*u;
end

