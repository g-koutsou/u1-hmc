from matplotlib import pyplot as plt
from lib import lattice, gauge
import matplotlib as mpl
from mpi4py import MPI
import numpy as np
import tqdm

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

betas = [1,2,3,4,5,6,7]
Ls = 128
L = lattice(dims=[Ls, Ls], procs=[1, 1])
Nsep = 32
pick = slice(0, 16384, Nsep)

me,uthin = dict(),dict()
for beta in tqdm.tqdm(betas):
    sbe = (lambda x: x.replace(".","p"))(f"{beta:5.3f}")
    me[beta],uthin[beta] = gauge.from_file(L, f"./data/be{sbe}.h5", top=f"beta{beta:5.3f}", pick=pick, metadata=True)

nrows = len(betas)

fig = plt.figure(1)
fig.clf()
gs = mpl.gridspec.GridSpec(nrows, 1, hspace=0.05, left=0.14)
legs = list()
for i,beta in enumerate(reversed(betas)):
    ax = fig.add_subplot(gs[i])
    plqs = np.array([x["plaquette"] for x in me[beta]])
    x = np.arange(0, len(plqs)*Nsep, Nsep)
    m, = ax.plot(x, plqs, color=colors[i])
    ax.set_ylabel("P")
    if i != nrows-1:
        ax.set_xticklabels([])
    legs.append([m, r"$\beta$={:2.1f}".format(beta)])
    c = plqs[-len(plqs)//2:].mean()
    y = np.max(np.abs(plqs[-len(plqs)//2:]-c))*1.1
    ax.set_ylim(c-y, c+y)
    ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%6.3f'))
l,m = zip(*legs)
fig.axes[0].legend(l, m, loc="lower left", frameon=False, bbox_to_anchor=(0, 1), ncol=8)
ax.set_xlabel("trajectory")
fig.canvas.draw()
fig.show()

nrows = len(betas)

fig = plt.figure(2)
fig.clf()
gs = mpl.gridspec.GridSpec(nrows, 1, hspace=0.05)
legs = list()
for i,beta in enumerate(reversed(betas)):
    ax = fig.add_subplot(gs[i])
    tps = np.array([x["topo"] for x in me[beta]])
    x = np.arange(0, len(tps)*Nsep, Nsep)
    m, = ax.plot(x, tps, color=colors[i])
    ax.set_ylabel("Q")
    if i != nrows-1:
        ax.set_xticklabels([])
    legs.append([m, r"$\beta$={:2.1f}".format(beta)])
#ylims = np.array([ax.set_ylim() for ax in fig.axes])
#yl = np.max(np.abs(ylims))
#for ax in fig.axes:
#    ax.set_ylim(-yl, yl)
l,m = zip(*legs)
fig.axes[0].legend(l, m, loc="lower left", frameon=False, bbox_to_anchor=(0, 1), ncol=8)
ax.set_xlabel("trajectory")
fig.canvas.draw()
fig.show()

usmr = dict()
alpha,nsmear = 0.1,64

fig = plt.figure(3)
fig.clf()
gs = mpl.gridspec.GridSpec(1, 1, hspace=0.05)
legs = list()
ax = fig.add_subplot(gs[0])
for i,beta in enumerate(reversed(betas)):
    usmr[beta] = gauge.copy(uthin[beta], slice=slice(0, None, 4))
    for ism in tqdm.tqdm(range(nsmear)):
        usmr[beta].apesmear(alpha)
        p = usmr[beta].plaquette()
        m = ax.errorbar(ism+i*0.05, np.mean(p), np.std(p)/np.sqrt(len(p)), marker="o", color=colors[i], ms=2)
    legs.append([m, r"$\beta$={:2.1f}".format(beta)])
l,m = zip(*legs)
ax.legend(l, m, loc="lower right", frameon=False)
ax.set_ylabel("P")
ax.set_xlabel("$N_{smear}$")
fig.canvas.draw()
fig.show()

