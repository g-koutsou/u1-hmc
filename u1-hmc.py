#########
#
# U(1), 2-dimensional, pure-gauge, Hybrid Monte Carlo
#
#########
from lib import lattice, gauge, momenta
from collections import defaultdict
from mpi4py import MPI
import numpy as np
import click
import time
import sys

def new_rng(seed):
    return np.random.Generator(np.random.PCG64(seed))

def print0(s):
    if MPI.COMM_WORLD.Get_rank() == 0:
        print(s)
        sys.stdout.flush()

rng_seeds = {"Metropolis": 7,
             "Momenta": 6,
             "Gauge": 3}

@click.command()
@click.option("--procs", "procs", default="1,1", help="process grid (nproc_x, nproc_y)")
@click.option("--beta", "beta", default=1.0, help="beta")
@click.option("--length","-L", "Ls", type=int, default=8, help="extent")
@click.option("--start", "start", default="hot", help="hot or cold")
@click.option("--n-tau", "NT", default=8, help="number of integration timesteps")
@click.option("--n-traj", "ntraj", default=1_024, help="total number of trajectories")
@click.option("--traj-length", "T", default=1, help="trajectory length")
@click.option("--print-every", default=16, help="number of trajectories between printing info to screen")
@click.option("--output-fname", "fname", default="out.h5", help="filename to write configurations")
def main(procs, beta, Ls, start, NT, ntraj, T, print_every, fname):
    tau = T/NT
    assert len(procs.split(",")) == 2, " {procs}: option `--procs` needs to be a comma separated list of two integers"
    procs = list(map(int, procs.split(",")))
    Ly,Lx = (Ls, Ls)
    assert start in ("hot", "cold"), " start should be either \"hot\" or \"cold\""
    L = lattice(dims=[Ly, Lx], procs=procs)
    u = gauge(L)
    if start == "hot":
        u.hot(seed=rng_seeds["Gauge"])
    if start == "cold":
        u.cold()
    #.. Helper functions
    def H(p, ux, masses=1):
        return p.dot(scale=1/np.sqrt(2*masses)) + beta * (1-ux.plaquette())*(Ly*Lx)
    def force(u):
        return ( - beta) * (complex(0, 1)*u.force()).real
    #..    
    print0("beta = {}, NT = {}".format(beta, NT))
    px = momenta(L, seed=rng_seeds["Momenta"])
    rng = new_rng(rng_seeds["Metropolis"])
    n_acc = 0
    twall = 0
    dH = list()
    ens = gauge(L, N=0)
    t0 = time.time()
    trajnames = list()
    metadata = list()
    for i in range(ntraj+1):
        ux = gauge.copy(u)
        px.normal()
        H0 = H(px, ux)
        px.move(force(ux), tau/2)
        for it in range(NT):
            ux.move(px, tau)
            f = force(ux)
            if it != NT-1:
                px.move(f, tau)
        px.move(f, tau/2)
        dH.append((H(px, ux) - H0)[0])
        ### Accept/reject
        if dH[-1] < 0:
            u = gauge.copy(ux)
            n_acc += 1
        else:
            r = rng.random()
            if np.exp(-dH[-1]) > r:
                u = gauge.copy(ux)
                n_acc += 1
        r_acc = n_acc/(i+1)
        p = u.plaquette()
        q = u.topo()
        ens.append(u)
        trajnames.append("traj{:08.0f}".format(i))
        metadata.append({"plaquette": p[0], "topo": q[0], "dH": dH[-1], "T": T, "tau": tau, "NT": NT})
        twall += time.time() - t0
        t0 = time.time()
        if (i % print_every == 0):
            print0("    traj. = {:8d}, plaq. = {: 7.5f}, Q = {: 5.1f}, acc. = {:3.2f}, dH = {: 12.4f}, t = {:6.4f} sec/MDU".format(
                i, p[0], q[0], r_acc, dH[-1], twall/(i+1))
            )
            groups = ["/beta{:5.3f}/{}".format(beta, n) for n in trajnames]
            ens.save(fname, groups=groups, metadata=metadata, append=(i and True))
            del ens, trajnames, metadata
            ens = gauge(L, N=0)
            trajnames = list()
            metadata = list()

if __name__ == "__main__":
    main()
