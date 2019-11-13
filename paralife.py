########################################################################
# paralife.py
# Description: A parallel implementation of Conway's Game of Life in Python.
########################################################################

from sys import exit
from time import sleep
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpi4py import MPI
import threading
from concurrent.futures import ThreadPoolExecutor

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

        self.fig = plt.figure()

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
        if(self.mpi_size > 1):
            comm.bcast(self._go, root=0)

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
        if not self._go:
            plt.close()
            return None
        
        plt.cla()        #Clear the plot so things draw much faster
        return plt.imshow(self.grid, interpolation='nearest')


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

        self.comm_stat()
        self._generation = self._generation + 1
        nl = np.count_nonzero(self.grid)
        self._numLife = self.comm.reduce(nl, op=MPI.SUM, root=0)
        if not self._go:
            plt.close()
            return None

        plt.cla()        #Clear the plot so things draw much faster
        return plt.imshow(self.grid, interpolation='nearest')


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


    '''Updates the simulation using a 1d domain decomposition:
    every process works on a group of rows determined by that processor's rank,
    then the results are broadcast to all the other processes so that
    each process has the entire updated simulation domain after each step'''
    def update_para_1d_dec(self,i):
        for row in range(N):
            if(row % mpi_size == rank):
                for col in range(N):
                    if self.neighbor_sum(row,col) == 3:
                        self.grid[row][col] = 1
                    elif self.neighbor_sum(row,col) != 2:
                        self.grid[row][col] = 0

        for row in range(N):
            self.grid[row] = comm.bcast(self.grid[row], root=row % mpi_size)

        if(rank == 0):
            self.aug_grid = self.form_aug_grid(self.grid)
        self.aug_grid = comm.bcast(self.aug_grid,root=0)

        plt.cla()        #Clear the plot so things draw much faster
        return plt.imshow(self.grid, interpolation='nearest')


    '''Updates the simulation using a 2d domain decomposition:
    each process gets a rectangular (square?) patch to work on '''
    def update_para_2d_dec(self,i):
        for row in range(N):
            if(row % mpi_size == rank):
                for col in range(N):
                    if self.neighbor_sum(row,col) == 3:
                        self.grid[row][col] = 1
                    elif self.neighbor_sum(row,col) != 2:
                        self.grid[row][col] = 0

        for row in range(N):
            self.grid[row] = comm.bcast(self.grid[row], root=row % mpi_size)

        if(rank == 0):
            self.aug_grid = self.form_aug_grid(self.grid)
        self.aug_grid = comm.bcast(self.aug_grid,root=0)

        plt.cla()        #Clear the plot so things draw much faster
        return plt.imshow(self.grid, interpolation='nearest')


    '''Display one step of the simulation'''
    def show(self):
        plt.imshow(self.grid, interpolation='nearest')
        plt.show()


    '''Play a movie of the simulation'''
    def movie(self):
        self.anim = animation.FuncAnimation(self.fig, self.update, interval=100)
        plt.show()


    '''Play a movie of the simulation with parallelization.'''
    def movie_para(self):
        self.anim = animation.FuncAnimation(self.fig, self.update_para_1d_dec_point_to_point, interval=100)
        plt.show()

    def stop(self):
        self._go = False
        return 'STOP'

def Test():
    while True:
        x = input()
        if x == 'stop':
            g_game._go = False
            break
        elif x == 'gen':
            print("#ofGeneration={0}".format(g_game._generation))
        elif x == 'num':
            print("#ofLife={0}".format(g_game._numLife))
        sleep(0.01)
    print("Test done.")

def Serve():
    ns = {}
    ns["stop"] = g_game.stop
    ns["gen"] = lambda : g_game._generation
    ns["num"] = lambda : g_game._numLife

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


game = Life(N) #Initialize the game with a grid of size N
g_game = game

# Make proc 0 have the true matrix - send it to the others:
game.grid = comm.bcast(game.grid, root=0)
game.aug_grid = game.form_aug_grid(game.grid)
rank = comm.Get_rank()
#if(game.mpi_size > 1):
#    print(rank, game.proc_grid)

# Start service
if rank == 0:
    pool = ThreadPoolExecutor(max_workers=1)
    pool.submit(Serve)

# Run simulation
if(mpi_size == 1):
    game.movie()
else:
    game.movie_para()


