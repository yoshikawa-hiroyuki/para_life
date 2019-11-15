########################################################################
# paralife.py
# Description: A parallel implementation of Conway's Game of Life in Python.
########################################################################
from sys import exit
from time import sleep
import numpy as np
from mpi4py import MPI
import threading
import io
from BKserver import BKserver

g_game = None

class Life:
    '''N: length & width of the simulation in number of cells'''
    def __init__(self, N):
        self.N = N
        self.comm = MPI.COMM_WORLD
        self.mpi_size = comm.Get_size()
        self.rank = comm.Get_rank()
        self.grid = np.asfortranarray(np.random.choice(2, (N,N)))
        self.aug_grid = self.form_aug_grid(self.grid)

        self._go = True
        self._numLife = np.count_nonzero(self.grid)
        self._generation = 0

        if(self.mpi_size > 1):
            recv_buf = np.asfortranarray(np.arange(N+2))
            self.cols_per_proc = int(np.ceil(N*1.0/self.mpi_size))
            self.proc_grid = np.random.choice(2, (N+2,self.cols_per_proc+2))

            #Make the bottom/top boundaries reflective:
            self.proc_grid[0,:] = self.proc_grid[N,:]
            self.proc_grid[N+1,:] = self.proc_grid[1,:]

            #Make the array column-major in memory
            # so that columns can be scatter/gathered
            self.proc_grid = np.asfortranarray(self.proc_grid)

            #Set up all the ghost cells between the processes:
            self.comm_ghosts()


    '''Communicates the ghost cells between the processes'''
    def comm_ghosts(self):
        if(self.mpi_size > 1):
            recv_buf_ary = np.arange(N+2) #Buffer for receiving an array
            recv_buf_num = np.arange(1)   #Buffer for receiving a number
            [left_tag, right_tag, corner_tl_tag, corner_bl_tag,
                 corner_tr_tag, corner_br_tag] = list(range(6))

            #Send ghost cells & columns from the neighboring processes:
            comm.Isend(np.array(self.proc_grid[:,1]),
                           dest=(self.rank-1)%self.mpi_size, tag=left_tag)
            comm.Isend(np.array(self.proc_grid[:,self.cols_per_proc]),
                           dest=(self.rank+1)%self.mpi_size, tag=right_tag)

            #Receive ghost cells & columns from the neighboring processes:
            comm.Recv(recv_buf_ary, source=(self.rank-1)%self.mpi_size,
                          tag=right_tag)
            self.proc_grid[:,0] = recv_buf_ary
            comm.Recv(recv_buf_ary, source=(self.rank+1)%self.mpi_size,
                          tag=left_tag)
            self.proc_grid[:,self.cols_per_proc+1] = recv_buf_ary

    def comm_stat(self):
        self._go = comm.bcast(self._go, root=0)

    def __str__(self):
        return str(self.grid)

    '''Forms an augmented grid'''
    def form_aug_grid(self,a):
        return np.lib.pad(a, ((1,1),(1,1)), 'wrap')


    '''Function that sums all the neighbors of the (row,col)'th entry
    of the grid using data from the global simulation'''
    def neighbor_sum(self,row,col):
        return self.aug_grid[row+2][col+2] + self.aug_grid[row+2][col+1] + \
          self.aug_grid[row+2][col] + \
          self.aug_grid[row+1][col+2] + self.aug_grid[row+1][col] + \
          self.aug_grid[row][col+2] + self.aug_grid[row][col+1] + \
          self.aug_grid[row][col]
    

    '''Function that sums all the neighbors of the (row,col)'th entry
    of the grid using data local to this process (self.proc_grid)'''
    def neighbor_sum_local(self,row,col):
        return self.proc_grid[row+1][col+1] + self.proc_grid[row+1][col] + \
          self.proc_grid[row+1][col-1] + \
          self.proc_grid[row][col+1] + self.proc_grid[row][col-1] + \
          self.proc_grid[row-1][col+1] + self.proc_grid[row-1][col] + \
          self.proc_grid[row-1][col-1]


    '''Function that updates the grid according to the rules of the game
     of life usinga toroidal geometry (wrapping in both directions'''
    def update(self,i):
        for row,col in np.ndindex(self.grid.shape):
            if self.neighbor_sum(row,col) == 3:
                self.grid[row][col] = 1
            elif self.neighbor_sum(row,col) != 2:
                self.grid[row][col] = 0
        self.aug_grid = self.form_aug_grid(self.grid)

        self._generation = self._generation + 1
        self._numLife = np.count_nonzero(self.grid)
        return self.grid


    '''Updates the matrix by splitting things into columns and then using
    point to point communication to share boundary columns between processes.
    Each process now gets a contiguous set of columns to operate on,
    so the number of processes should not be greater than the number of columns
    (otherwise several of them will not be used).'''
    def update_para_1d_dec_point_to_point(self,i):
        temp_grid = np.copy(self.proc_grid)
        for row in range(1,N+1):
            for col in range(1,self.cols_per_proc+1):
                if self.neighbor_sum_local(row,col) == 3:
                    temp_grid[row][col] = 1
                elif self.neighbor_sum_local(row,col) != 2:
                    temp_grid[row][col] = 0
        self.proc_grid = temp_grid
        #Enforce top/bottom row boundary conditions:
        self.proc_grid[0,:] = self.proc_grid[N,:]
        self.proc_grid[N+1,:] = self.proc_grid[1,:]

        #Now send the updated columns to the necessary
        comm.barrier()
        self.comm_ghosts()

        #For testing with the animation:
        self.reconstruct_full_sim()

        self._generation = self._generation + 1
        nl = np.count_nonzero(self.grid)
        self._numLife = self.comm.reduce(nl, op=MPI.SUM, root=0)
        return self.grid


    '''Reconstructs the data for the full simulation on every processor for a
    visualization check.'''
    def reconstruct_full_sim(self):
        recv_buf_ary = np.arange(self.N*self.N)
        send_buf_ary \
        = np.asfortranarray(self.proc_grid[1:N+1, 1:(self.cols_per_proc+2)-1])
        recv_buf_ary = comm.gather(send_buf_ary,root=0)
        if(self.rank == 0):
            for proc in range(len(recv_buf_ary)):
                self.grid[:,proc*self.cols_per_proc:(proc+1)*self.cols_per_proc]\
                  = recv_buf_ary[proc]
        self.grid = comm.bcast(self.grid, root=0)

    def stop(self):
        self._go = False
        return 'STOP'


def Serve():
    ns = {}
    ns["stop"] = g_game.stop
    ns["gen"] = lambda : g_game._generation
    ns["num"] = lambda : g_game._numLife
    ns["grid"] = lambda : str(g_game.grid)

    BKserver(comm=g_game.comm, ns=ns).Serve()

    
#-------------------------------------------------------------
N = 32 #Size of grid for the game of life

comm = MPI.COMM_WORLD
mpi_size = comm.Get_size()

# Figure out MPI division of tasks (how many rows per processor)
rows_per_proc = N / mpi_size
if(rows_per_proc != int(N*1.0 / mpi_size)):
    if(comm.Get_rank() == 0):
        print("Matrix size not evenly divisible by number of processors."
              + " Use different values.")
        exit()
elif(mpi_size > 1):
    if(comm.Get_rank() == 0):
        print("Parallelizing with ",mpi_size," processors.")
else:
    print("Running in serial mode.")

np.set_printoptions(threshold=np.inf)

game = Life(N) #Initialize the game with a grid of size N
g_game = game

# Make proc 0 have the true matrix - send it to the others:
game.grid = comm.bcast(game.grid, root=0)
game.aug_grid = game.form_aug_grid(game.grid)
rank = comm.Get_rank()

# Start service
if rank == 0:
    t1 = threading.Thread(target=Serve)
    t1.start()

# Run simulation
while True:
    if mpi_size < 2:
        game.update(game._generation)
    else:
        game.update_para_1d_dec_point_to_point(game._generation)
    sleep(0.1)
    
    game.comm_stat()
    if not game._go:
        break
    continue # end of while

exit(0)
