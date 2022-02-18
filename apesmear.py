#########
#
# APE Smear U(1) configurations stored in an HDF5 file
#
#########
import time
import sys

from mpi4py import MPI
import numpy as np
import gvar as gv
import click
import h5py

from lib import lattice, gauge, momenta

def print0(s):
    if MPI.COMM_WORLD.Get_rank() == 0:
        print(s)
        sys.stdout.flush()

def get_ntrajs(fname, top="/"):
    with h5py.File(fname, "r") as fp:
        assert top in fp
        trajs = [x for x in list(fp[top]) if "traj" in x]
    return len(trajs)

@click.command()
@click.argument("fname")
@click.option("--procs", "procs", default="1,1", help="process grid (nproc_x, nproc_y)")
@click.option("--length","-L", "Ls", type=int, default=8, help="lattice extent")
@click.option("--nsmear", "nsmear", default=1, help="number of smearing iterations")
@click.option("--alpha", "alpha", default=0.1, help="smearing parameter")
@click.option("--group", "group", default="/", help="HDF5 group to look under")
@click.option("--batch", "nbatch", default=None, help="smear batch configs. at a time")
@click.option("--output-fname", "output_fname", default="out.h5", help="filename to write configurations to")
def main(fname, procs, Ls, nsmear, alpha, group, nbatch, output_fname):
    procs = list(map(int, procs.split(",")))
    L = lattice(dims=[Ls, Ls], procs=procs)
    ntrajs = get_ntrajs(fname, group)
    if nbatch is None:
        nbatch = ntrajs
    else:
        nbatch = int(nbatch)
    batches = list(range(0, ntrajs, nbatch))
    t0 = time.time()
    for i,ibatch in enumerate(batches):
        u = gauge.from_file(L, fname, top=group, pick=slice(ibatch, ibatch+nbatch))
        nc = u.N
        plaq = [u.plaquette()]
        for j in range(nsmear):
            u.apesmear(alpha=alpha)
            plaq.append(u.plaquette())
        p,e = plaq[-1].mean(),plaq[-1].std()/np.sqrt(nc)
        out_group = group + "/apeN{}alpha{}/".format(nsmear, alpha)
        trajnames = [out_group + "/traj{:08.0f}".format(i) for i in range(ibatch, ibatch+nc)]
        metadata = [{"plaquette": p} for p in np.array(plaq).T.tolist()]
        u.save(output_fname, groups=trajnames, metadata=metadata, append=(i and True))
        t1 = time.time()
        print0(" Smeared {:6d} configs. ({:3d} of {:3d} batches), plaq. = {}, {:10.5f} sec".format(
            nc, i+1, len(batches), gv.gvar(p, e), t1-t0))
        t0 = time.time()
        
if __name__ == "__main__":
    main()
