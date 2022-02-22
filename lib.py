from mpi4py import MPI
import numpy as np
import h5py
import re

class lattice(object):
    def __init__(self, dims, procs=[1,1]):
        Ly,Lx = dims
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        nproc = comm.Get_size()
        assert np.prod(procs) == nproc
        assert Ly % procs[0] == 0
        assert Lx % procs[1] == 0
        self.procs = procs
        self.pcoords = [rank // procs[1], rank % procs[1]]
        self.L = [Ly, Lx]
        ly = Ly//procs[0]
        lx = Lx//procs[1]
        self.lL = [ly, lx]
        y,x = np.mgrid[:ly,:lx].reshape([2, ly*lx])
        self.idx = {
            +0: (x+1) + (y+1)*(lx+2),
            +1: (x+2) + (y+1)*(lx+2),
            +2: (x+1) + (y+2)*(lx+2),
            -1: (x+0) + (y+1)*(lx+2),
            -2: (x+1) + (y+0)*(lx+2),
            (+1,+2): (x+2) + (y+2)*(lx+2),
            (+2,+1): (x+2) + (y+2)*(lx+2),
            (+1,-2): (x+2) + (y+0)*(lx+2),
            (-2,+1): (x+2) + (y+0)*(lx+2),
            (-1,+2): (x+0) + (y+2)*(lx+2),
            (+2,-1): (x+0) + (y+2)*(lx+2),
            (-1,-2): (x+0) + (y+0)*(lx+2),
            (-2,-1): (x+0) + (y+0)*(lx+2),
        }
        ry,rx = self.pcoords
        self.x_comm = comm.Split(color=ry)
        self.y_comm = comm.Split(color=rx)
        self.comm = comm
        self.rank = rank
        self.nproc = nproc
        return
    
class field(object):
    def __init__(self, l):
        assert isinstance(l, lattice)
        self.lat = l

    def save(self, fname, groups=["/"], metadata=None, append=False):
        comm = self.lat.comm
        mode = "a" if append else "w"
        if metadata is not None:
            assert isinstance(metadata, list)
        with h5py.File(fname, mode, driver='mpio', comm=comm) as fp:
            ndof = self.ndof
            N = self.N
            ly, lx = self.lat.lL
            if isinstance(self, gauge):
                x = self.lat.idx[+0]
            else:
                x = np.arange(ly*lx)
            Ly, Lx = self.lat.L
            ry, rx = self.lat.pcoords
            sx = slice(lx*rx, lx*(rx+1))
            sy = slice(ly*ry, ly*(ry+1))
            for i,grp in enumerate(groups):
                gr = fp.require_group(grp)
                dset = gr.require_dataset("u", shape=(Ly,Lx,ndof), dtype=self.data.dtype)
                dset[sy,sx,:] = self.data[i,x,:].reshape([ly, lx, ndof])
                if metadata is not None:
                    assert isinstance(metadata[i], dict)
                    for k,v in metadata[i].items():
                        gr.attrs.create(k, v)
        return

    @classmethod
    def from_file(cls, l, fname, top="/", pick=None, metadata=False, metadata_only=False):
        comm = l.comm
        if metadata or metadata_only:
            m = list()
        with h5py.File(fname, "r", driver='mpio', comm=comm) as fp:
            Ly,Lx = l.L
            if pick is None:
                pick = slice(0, None)
            trajs = list(fp[f"{top}/"])[pick]
            if metadata_only:
                for i,tr in enumerate(trajs):
                    m.append(dict(fp[f"{top}/{tr}"].attrs))
                return m
            ly, lx = l.lL
            ry, rx = l.pcoords
            sx = slice(lx*rx, lx*(rx+1))
            sy = slice(ly*ry, ly*(ry+1))
            p = cls(l, N=len(trajs))
            if isinstance(p, gauge):
                x = l.idx[+0]
            else:
                x = np.arange(ly*lx)
            for i,tr in enumerate(trajs):
                p.data[i,x,:] = fp[f"{top}/{tr}/u"][sy, sx, :].reshape([ly*lx, p.ndof])
                if metadata:
                    m.append(dict(fp[f"{top}/{tr}"].attrs))
        if metadata:
            return m,p
        else:
            return p
    
class gauge(field):
    def __init__(self, l, data=None, N=1):
        super().__init__(l)
        self.ndof = 2
        self.N = N
        ly,lx = self.lat.lL
        if data is not None:
            self.data = np.array(data[:,:,:], order="C")
        else:
            self.data = np.zeros([self.N, (ly+2)*(lx+2), self.ndof], complex)
        return
    
    @classmethod
    def copy(cls, u, slice=slice(0, None, None)):
        data = u.data[slice, :, :]
        return cls(u.lat, data=data, N=data.shape[0])

    def cold(self):
        self.data = self.data ** 0
        return
    
    def _xchange(self):
        ly,lx = self.lat.lL
        u0 = np.array(self.data.reshape([self.N,ly+2,lx+2,self.ndof]))
        npy,npx = self.lat.procs
        rky,rkx = self.lat.pcoords
        comm = self.lat.comm
        rank = self.lat.rank
        rp0 = rkx + ((npy+rky+1)%npy)*npx
        rm0 = rkx + ((npy+rky-1)%npy)*npx
        r0p = (npx+rkx+1)%npx + rky*npx
        r0m = (npx+rkx-1)%npx + rky*npx
        u1 = np.array(u0.transpose(1, 0, 2, 3), order="C") ### n, y, x, mu -> y, n, x, mu
        comm.Sendrecv(u1[ly, :, :, :], rp0, sendtag=rp0, recvbuf=u1[   0, :, :, :], source=rm0, recvtag=rank)
        comm.Sendrecv(u1[ 1, :, :, :], rm0, sendtag=rm0, recvbuf=u1[ly+1, :, :, :], source=rp0, recvtag=rank)
        u2 = np.array(u1.transpose(2, 0, 1, 3), order="C") ### y, n, x, mu -> x, y, n, mu
        comm.Sendrecv(u2[lx, :, :, :], r0p, sendtag=r0p, recvbuf=u2[   0, :, :, :], source=r0m, recvtag=rank)
        comm.Sendrecv(u2[ 1, :, :, :], r0m, sendtag=r0m, recvbuf=u2[lx+1, :, :, :], source=r0p, recvtag=rank)
        self.data[:,:,:] = np.array(u2[:,:,:,:].transpose((2, 1, 0, 3)).reshape([self.N, (ly+2)*(lx+2), self.ndof]), order="C")
        return

    def xchange(self):
        ly,lx = self.lat.lL
        u0 = np.array(self.data.reshape([self.N,ly+2,lx+2,self.ndof]))
        npy,npx = self.lat.procs
        rky,rkx = self.lat.pcoords
        comm = self.lat.comm
        rank = self.lat.rank
        rp0 = rkx + ((npy+rky+1)%npy)*npx
        rm0 = rkx + ((npy+rky-1)%npy)*npx
        r0p = (npx+rkx+1)%npx + rky*npx
        r0m = (npx+rkx-1)%npx + rky*npx
        # u1 = np.array(u0.transpose(1, 0, 2, 3), order="C") ### n, y, x, mu -> y, n, x, mu
        # comm.Sendrecv(u1[ly, :, :, :], rp0, sendtag=rp0, recvbuf=u1[   0, :, :, :], source=rm0, recvtag=rank)
        # comm.Sendrecv(u1[ 1, :, :, :], rm0, sendtag=rm0, recvbuf=u1[ly+1, :, :, :], source=rp0, recvtag=rank)
        # u2 = np.array(u1.transpose(2, 0, 1, 3), order="C") ### y, n, x, mu -> x, y, n, mu
        # comm.Sendrecv(u2[lx, :, :, :], r0p, sendtag=r0p, recvbuf=u2[   0, :, :, :], source=r0m, recvtag=rank)
        # comm.Sendrecv(u2[ 1, :, :, :], r0m, sendtag=r0m, recvbuf=u2[lx+1, :, :, :], source=r0p, recvtag=rank)
        # self.data[:,:,:] = np.array(u2[:,:,:,:].transpose((2, 1, 0, 3)).reshape([self.N, (ly+2)*(lx+2), self.ndof]), order="C")
        u0[:,   0,   :, :] = comm.sendrecv(u0[:,ly, ...], rp0, sendtag=rp0, source=rm0, recvtag=rank)
        u0[:,ly+1,   :, :] = comm.sendrecv(u0[:, 1, ...], rm0, sendtag=rm0, source=rp0, recvtag=rank)
        u0[:,   :,   0, :] = comm.sendrecv(u0[:, :,lx,:], r0p, sendtag=r0p, source=r0m, recvtag=rank)
        u0[:,   :,lx+1, :] = comm.sendrecv(u0[:, :, 1,:], r0m, sendtag=r0m, source=r0p, recvtag=rank)
        self.data[:,:,:] = np.array(u0[:,:,:,:].reshape([self.N, (ly+2)*(lx+2), self.ndof]), order="C")
        return
    
    def hot(self, seed=None):
        Ly,Lx = self.lat.L
        ly,lx = self.lat.lL
        if seed is None:
            seed = 0
        rng = np.random.Generator(np.random.PCG64(seed))
        th = rng.random(size=[self.N, Ly, Lx, self.ndof])*np.pi*2-np.pi
        rky,rkx = self.lat.pcoords
        y = slice(rky*ly, (rky+1)*ly)
        x = slice(rkx*lx, (rkx+1)*lx)
        self.data[:,self.lat.idx[+0],:] = np.exp(complex(0, 1)*th[:,y,x,:]).reshape([self.N,lx*ly,self.ndof])
        del rng
        return
    
    def plaquette(self):
        self.xchange()
        x = self.lat.idx
        Ly,Lx = self.lat.L
        u0 = self.data[:,x[+0],0]
        u1 = self.data[:,x[+1],1]
        u2 = self.data[:,x[+2],0].conj()
        u3 = self.data[:,x[ 0],1].conj()
        pl = np.array((u0*u1*u2*u3).sum(axis=1).real, order="C")
        ps = np.zeros(shape=(self.N), dtype=float)
        self.lat.comm.Allreduce(pl, ps, op=MPI.SUM)
        return ps/(Ly*Lx)

    def topo(self):
        self.xchange()
        x = self.lat.idx
        u0 = self.data[:,x[ 0],0]
        u1 = self.data[:,x[+1],1]
        u2 = self.data[:,x[+2],0].conj()
        u3 = self.data[:,x[ 0],1].conj()
        th = (u0*u1*u2*u3)
        qx = np.array(np.arctan2(th.imag, th.real), order="C")
        qs = np.zeros(shape=(self.N), dtype=float)
        self.lat.comm.Allreduce(qx.sum(axis=1), qs, op=MPI.SUM)
        return qs/(2*np.pi)

    def rotate(self, theta):
        x = self.lat.idx
        self.data[:,x[0],:] = self.data[:,x[0],:]*np.exp(complex(0, 1)*theta)
        return

    def move(self, mom, tau):
        assert isinstance(mom, momenta)
        self.rotate(mom.data*tau)
        return
    
    def force(self):
        self.xchange()
        Ly,Lx = self.lat.L
        ly,lx = self.lat.lL
        x = self.lat.idx
        f = np.zeros([self.N,ly*lx,self.ndof], complex)
        for mu in range(2):
            nu = (mu+1)%2
            ### "Forwards" staple
            u0 = self.data[:,x[0],mu]
            u1 = self.data[:,x[+(mu+1)],nu]
            u2 = self.data[:,x[+(nu+1)],mu].conj()
            u3 = self.data[:,x[0],nu].conj()
            f0 = u0*u1*u2*u3
            #.
            ### "Backwards" staple
            u0 = self.data[:,x[-(nu+1)],mu]
            u1 = self.data[:,x[+(mu+1),-(nu+1)],nu]
            u2 = self.data[:,x[0],mu].conj()
            u3 = self.data[:,x[-(nu+1)],nu].conj()
            f1 = u0*u1*u2*u3
            #.
            f[:, :, mu] = f0 + f1.conj()
        return f

    def apesmear(self, alpha):
        self.xchange()
        Ly,Lx = self.lat.L
        ly,lx = self.lat.lL
        x = self.lat.idx
        u = np.zeros([self.N,(ly+2)*(lx+2),self.ndof], complex)
        for mu in range(2):
            nu = (mu+1)%2
            ### "Forwards" staple
            u1 = self.data[:,x[0],nu]
            u2 = self.data[:,x[+(nu+1)],mu]
            u3 = self.data[:,x[+(mu+1)],nu].conj()
            f0 = u1*u2*u3
            #.
            ### "Backwards" staple
            u1 = self.data[:,x[-(nu+1)],nu].conj()
            u2 = self.data[:,x[-(nu+1)],mu]
            u3 = self.data[:,x[+(mu+1),-(nu+1)],nu]
            f1 = u1*u2*u3
            #.
            u0 = self.data[:,x[0],mu]
            u0 = (1-alpha)*u0 + alpha*(f0 + f1)
            u[:, x[0], mu] = u0/np.sqrt((u0.conj()*u0).real)
        self.data = np.array(u, order="C")
        return
    
    def append(self, u):
        assert isinstance(u, gauge), "gauge.append() argument should be a gauge instance"
        self.data = np.concatenate((self.data, u.data))
        self.N += u.N
        return
    
class momenta(field):
    def __init__(self, l, data=None, N=1, seed=None, rng=None):
        super().__init__(l)
        self.ndof = 2
        self.N = N
        if data is not None:
            self.data = np.array(data[:,:,:], order="C")
        else:
            self.data = np.zeros(shape=[self.N,np.prod(self.lat.lL),self.ndof], dtype=float)
        self.seed = seed
        if rng is not None:
            self.rng = rng
        else:
            self.rng = np.random.Generator(np.random.PCG64(seed))        

    def normal(self):
        Ly,Lx = self.lat.L
        ly,lx = self.lat.lL
        rky,rkx = self.lat.pcoords
        y = slice(rky*ly, (rky+1)*ly)
        x = slice(rkx*lx, (rkx+1)*lx)
        r = self.rng.normal(size=[self.N,Ly,Lx,self.ndof])
        self.data[:,:] = r[:, y, x, :].reshape([self.N,ly*lx,self.ndof])
        return

    def dot(self, scale=1):
        p = self.data*scale
        pl = (p*p.conj()).sum(axis=1).sum(axis=1).real
        ps = self.lat.comm.allreduce(pl, op=MPI.SUM)
        return ps

    def move(self, f, tau):
        self.data = self.data - tau*f
                
    def scale(self, a):
        return momenta(self.lat, seed=self.seed, rng=self.rng, data=self.data*a)
    
