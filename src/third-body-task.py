# coding: utf-8

# In[59]:


#
# Исходные данные задачи
#

# Гравитационная постоянная
G = 6.67e-11

# Массы тел (Луна, Земля, Солнце)
m = [7.349e22, 5.792e24, 1.989e30]

# Расчитываем гравитационные параметры тел
mu = []

print("Гравитационные параметры тел")

for i, mass in enumerate(m):
    mu.append(G * mass)
    print("mu[" + str(i) + "] = " + str(mu[i]))

# Нормируем гравитационные параметры к Солнцу
kappa = []

print("Нормированные гравитационные параметры")

for i, gp in enumerate(mu):
    kappa.append(gp / mu[2])
    print("xi[" + str(i) + "] = " + str(kappa[i]))

print("\n")

# Астрономическая единица
a = 1.495978707e11

import math

# Масштаб безразмерного времени, c
T = 2 * math.pi * a * math.sqrt(a / mu[2])

print("Масштаб времени T = " + str(T) + "\n")

# Координаты NASA для Луны
xL = 5.771034756256845E-01
yL = -8.321193799697072E-01
zL = -4.855790760378579E-05

import numpy as np

xi_10 = np.array([xL, yL, zL])
print("Начальное положение Луны, а.е.: " + str(xi_10))

# Координаты NASA для Земли
xE = 5.755663665315949E-01
yE = -8.298818915224488E-01
zE = -5.366994499016168E-05

xi_20 = np.array([xE, yE, zE])
print("Начальное положение Земли, а.е.: " + str(xi_20))

# Расчитываем начальное положение Солнца, полагая что начало координат - в центре масс всей системы
xi_30 = - kappa[0] * xi_10 - kappa[1] * xi_20
print("Начальное положение Солнца, а.е.: " + str(xi_30))

# Вводим константы для вычисления безразмерных скоростей
Td = 86400.0
u = math.sqrt(mu[2] / a) / 2 / math.pi

print("\n")

# Начальная скорость Луны
vxL = 1.434571674368357E-02
vyL = 9.997686898668805E-03
vzL = -5.149408819470315E-05

vL0 = np.array([vxL, vyL, vzL])
uL0 = np.array([0.0, 0.0, 0.0])

for i, v in enumerate(vL0):
    vL0[i] = v * a / Td
    uL0[i] = vL0[i] / u

print("Начальная скорость Луны, м/с: " + str(vL0))
print(" -//- безразмерная: " + str(uL0))

# Начальная скорость Земли
vxE = 1.388633512282171E-02
vyE = 9.678934168415631E-03
vzE = 3.429889230737491E-07

vE0 = np.array([vxE, vyE, vzE])
uE0 = np.array([0.0, 0.0, 0.0])

for i, v in enumerate(vE0):
    vE0[i] = v * a / Td
    uE0[i] = vE0[i] / u

print("Начальная скорость Земли, м/с: " + str(vE0))
print(" -//- безразмерная: " + str(uE0))

# Начальная скорость Солнца
vS0 = - kappa[0] * vL0 - kappa[1] * vE0
uS0 = - kappa[0] * uL0 - kappa[1] * uE0

print("Начальная скорость Солнца, м/с: " + str(vS0))
print(" -//- безразмерная: " + str(uS0))


#
#   Вычисление векторов обобщенных ускорений
#
def calcAccels(xi):
    k = 4 * math.pi ** 2

    xi12 = xi[1] - xi[0]
    xi13 = xi[2] - xi[0]
    xi23 = xi[2] - xi[1]

    s12 = math.sqrt(np.dot(xi12, xi12))
    s13 = math.sqrt(np.dot(xi13, xi13))
    s23 = math.sqrt(np.dot(xi23, xi23))

    a1 = (k * kappa[1] / s12 ** 3) * xi12 + (k * kappa[2] / s13 ** 3) * xi13
    a2 = -(k * kappa[0] / s12 ** 3) * xi12 + (k * kappa[2] / s23 ** 3) * xi23
    a3 = -(k * kappa[0] / s13 ** 3) * xi13 - (k * kappa[1] / s23 ** 3) * xi23

    return [a1, a2, a3]


#
#   Система уравнений в нормальной форме Коши
#
def f(t, y):
    n = 9

    dydt = np.zeros((2 * n))

    for i in range(0, n):
        dydt[i] = y[i + n]

    xi1 = np.array(y[0:3])
    xi2 = np.array(y[3:6])
    xi3 = np.array(y[6:9])

    accels = calcAccels([xi1, xi2, xi3])

    i = n
    for accel in accels:
        for a in accel:
            dydt[i] = a
            i = i + 1

    return dydt

# Начальные условия задачи Коши
y0 = [xi_10[0], xi_10[1], xi_10[2],
      xi_20[0], xi_20[1], xi_20[2],
      xi_30[0], xi_30[1], xi_30[2],
      uL0[0], uL0[1], uL0[2],
      uE0[0], uE0[1], uE0[2],
      uS0[0], uS0[1], uS0[2]]

#
# Интегрируем уравнения движения
#

# Начальное время
t_begin = 0
# Конечное время
t_end = 18.0#6 * 29 * Td / T;
# Интересующее нас число точек траектории
N_plots = 1000
# Шаг времени между точкими
step = (t_end - t_begin) / N_plots

import scipy.integrate as spi

solver = spi.ode(f)

solver.set_integrator('vode', nsteps=50000, method='bdf', max_step=1e-5, rtol=1e-12)
solver.set_initial_value(y0, t_begin)

ts = []
ys = []
i = 0

while solver.successful() and solver.t <= t_end:
    solver.integrate(solver.t + step)
    ts.append(solver.t)
    ys.append(solver.y)
    print(ts[i], ys[i])
    i = i + 1

t = np.vstack(ts)

xi1_x, xi1_y, xi1_z, \
xi2_x, xi2_y, xi2_z, \
xi3_x, xi3_y, xi3_z, \
u1_x, u1_y, u1_z, \
u2_x, u2_y, u2_z, \
u3_x, u3_y, u3_z = np.vstack(ys).T

#x_L = 2.213020522725194E-03
#y_L =-1.541180833539449E-03
#z_L =-8.272746609930161E-05

#xi_L = np.array([x_L, y_L, z_L])
#xi_12 = np.array([xi1_x[-2] - xi2_x[-2], xi1_y[-2] - xi2_y[-2], xi1_z[-2] - xi2_z[-2]])

#dr = xi_L - xi_12

#print("t_end = ", t_end)
#print(a * math.sqrt(np.dot(dr, dr)) / 1000.0)

#
# Построение графиков
#
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig1 = plt.figure()

ax = fig1.add_subplot(111, projection='3d')
ax.plot((xi1_x - xi2_x) * a, (xi1_y - xi2_y) * a, (xi1_z - xi2_z) * a, color='red')
ax.axis('equal')
ax.set_xlim(-5e8, 5e8)
ax.set_xlabel('X, м')
ax.set_ylim(-5e8, 5e8)
ax.set_ylabel('Y, м')
ax.set_zlim(-5e8, 5e8)
ax.set_zlabel('Z, м')
ax.set_title('Траектория Луны в геоцентрической эклиптической системе координат')

fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='rectilinear')
ax2.plot((xi1_x - xi2_x) * a, (xi1_y - xi2_y) * a, color='red')
ax2.axis('equal')
ax2.set_xlim(-5e8, 5e8)
ax2.set_xlabel('X, м')
ax2.set_ylim(-5e8, 5e8)
ax2.set_ylabel('Y, м')
ax2.grid(True)
ax2.set_title('Проекция орбиты Луны в плоскость эклиптики')

plt.show()
