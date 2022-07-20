from firedrake import *
import numpy as np
import matplotlib.pyplot as plt

mesh = PeriodicIntervalMesh(500, 50)
V = FunctionSpace(mesh, "CG", 1)

D1 = 1000.
D2 = 0.1
c = 5.
d = 0.1

dt = 1e-1

x = SpatialCoordinate(mesh)[0]
Uinit = ( ((d**2)*(c-1))/(c**2) )**(1/(2*2 -1))
Vinit = (c*Uinit)/d
U0 = Function(V).assign(Constant(Uinit))
V0 = Function(V).assign(Constant(Vinit))
pcg = PCG64(seed=123456789)
rg = RandomGenerator(pcg)
pert = rg.normal(V)
U0 += 0.01*pert

u = Function(V)
v = Function(V)
u0 = Function(V)
v0 = Function(V)
w = TestFunction(V)
phi = TestFunction(V)

Fu = w*(u-u0)*dx - dt*(-D1*inner(grad(w), grad(u))*dx
                       + w*(c-1)*u*dx - w*u**2*v**2*dx
                       )

Fv = phi*(v-v0)*dx - dt*(-D2*inner(grad(phi), grad(v))*dx
                         + phi*u*dx + phi*u**2*v**2*dx
                         - d*phi*v*dx
                         )

uprob = NonlinearVariationalProblem(Fu, u)
usolver = NonlinearVariationalSolver(uprob)

vprob = NonlinearVariationalProblem(Fv, v)
vsolver = NonlinearVariationalSolver(vprob)

outfile = File('be_out.pvd')
u0.assign(U0)
v0.assign(V0)
u_pert = Function(V).assign(u0-Uinit)
v_pert = Function(V).assign(v0-Vinit)
outfile.write(u0, v0)
print(len(u0.dat.data[:]))

npts = 1000
points = np.linspace(0, 50, npts)

tmax = 10*dt
times = np.linspace(0, tmax, int((tmax+0.5*dt)/dt)+2)
u_arr = np.zeros((len(times), len(points)))
v_arr = np.zeros((len(times), len(points)))

u_arr[0, :] = np.asarray(u0.at(points))
v_arr[0, :] = np.asarray(v0.at(points))

print(len(times))
print(u_arr.shape)
plt.plot(u_arr[0,:])
plt.show()

t = 0
count = 0
i = 0
while t < tmax:
    t += dt
    count += 1
    vsolver.solve()
    v0.assign(v)
    usolver.solve()
    u0.assign(u)
    if count % 1 == 0:
        print(t)
        i += 1
        outfile.write(u0, v0)
        u_arr[i, :] = np.asarray(u0.at(points))
        v_arr[i, :] = np.asarray(v0.at(points))

plt.plot(u_arr[-1, :])
plt.show()

plt.contourf(points, times, u_arr)
plt.show()
