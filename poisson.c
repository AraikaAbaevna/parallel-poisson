/** 
 * poisson.c
 *
 * CY901 Assignment 6
 * Student ID: 0903448
 */

/* Include libraries */
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <mpi.h>

/* Set level of verbosity       */
/*  0 = No messages             */
/*  1 = Print parameters        */
/*  2 = Print debug information */
#define VERBOSE 1

/* Set style of file output     */
/* as required by Mathematica   */
/*  0 = Exact matrix p          */
/*  1 = (x,y,p(x,y))            */
#define PRINTXYVAL 0

/* Define pi if not in math.h */
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Macros */
#define MPI_CALL(x) do { if((x) != MPI_SUCCESS) {       \
  printf("Error in %s at line %d\n",__FILE__,__LINE__); \
  exit(x);}} while(0)

/* Function prototypes */
char *strdup( const char *str );

/* Main program logic */
int main( int argc, char **argv ) {
  
  /** 
   * Constants definitions
   *
   * Configure size of grid, simulation settings and problem parameters.
   */

  const unsigned int global_N = 2400;               /* Size of grid (global) */
    
  const unsigned int maxiter = 1000;                /* Number of iterations. Set to -1 to run to convergence */
      
  const double kappa = 100.0;                       /* kappa */
  const double rpx = 0.3, rpy = 0.3;                /* +1 charge centre r_\plus */
  const double rmx = 0.7, rmy = 0.7;                /* -1 charge centre r_\minus */
    
  const double eps = 0.000001;                      /* Convergence criterion */
    
  const double qt = 0.25;                           /* Four-point stencil factor */
  const double kpi = kappa / M_PI;                  /* Constant factor in rhs */

  char *potential = strdup("potential.dat");        /* Potential output filename */
  char *field     = strdup("field.dat");            /* Field output filename */

  const int interval = 100;                         /* Debug interval */

  const int x = 0, y = 1;                           /* Semantic variables: directions */
  const int up = 0, down = 1, left = 2, right = 3;  /* Semantic variables: neighbours */

  /** 
   * Variable definitions
   *
   * Simulation variables, including grids and errors
   */
          
  double **p, **p_new, **rho, **swap, **global_p;   /* Grids */
  double h, h2, invtwoh;                            /* Grid spacing, and its square */

  unsigned int N;                                   /* Local grid dimensions */
    
  unsigned int ix, iy, ixg, iyg, ip;                /* Counters */
  int iter, enditer;

  double coord_x, coord_y;                          /* Cartesian coordinates */
    
  double error, maxerror;                           /* Temporary error variables */
  
  double cdx, cdy;                                  /* Central difference variables */
   
  FILE *fp;                                         /* File pointer */

  /* MPI variables */
  int nprocs;                                       /* Number of processors */
  int proc;                                         /* Current processor rank */

  MPI_Comm cart_comm;                               /* Cartesian communicator */
  int global_coords[2];                             /* Current processor coordinates in grid */
  int global_start[2];                              /* Current processor global coordinates in grid */
  int global_neighbours[4];                         /* Current processor neighbour ranks (up,down,left,right) */

  MPI_Status status;                                /* MPI status structure for receives */
  int proot;                                        /* Square root of number of processors */

  int ndims = 2;                                    /* Number of dimensions */
  int reorder = 0;                                  /* Rank reorder flag */
  int pbc [2] = {1,1};                              /* Periodic boundary conditions */
  int dims[2];                                      /* Number of processes in each dimension */

  double **halo_buffers;                            /* Arrays to store local halo data */
  double *sendbuf, *recvbuf;                        /* Send and receive buffers */
  double stencil_left, stencil_right;               /* Temporary variables for five-point stencil [left/right] */
  double stencil_up, stencil_down;                  /* Temporary variables for five-point stencil [up/down] */
                                                    /* Required for program to run on one processor */

  double t1, t2;                                    /* MPI timers */

  /** 
   * MPI Setup
   *
   * Initiation and setup of 2D
   * Cartesian coordinates.
   */
  
  MPI_CALL( MPI_Init( &argc, &argv ) );                         /* Initialise MPI */
  MPI_CALL( MPI_Comm_size( MPI_COMM_WORLD, &nprocs ) );         /* Get number of processors */
  MPI_CALL( MPI_Comm_rank( MPI_COMM_WORLD, &proc ) );           /* Get current processor rank */

  /* Check we have a square number of processors */
  proot = (int) sqrt( (double) nprocs + 0.5 );
  if( proot * proot != nprocs ) {
    if( proc == 0 ) {
      printf("Number of processors must be an exact square!\n");
    }

    /* All shutdown MPI */
    MPI_CALL( MPI_Finalize() );

    /* Exit program */
    exit(EXIT_FAILURE);
  }

  /* Dimensions of Cartesian communicator */
  dims[x] = proot; dims[y] = proot;

  /* Create a Cartesian communicator from MPI_COMM_WORLD */
  MPI_CALL( MPI_Cart_create( MPI_COMM_WORLD, ndims, dims, pbc, reorder, &cart_comm) );

  /* Store rank of processor in new communicator. No reordering. */
  MPI_CALL( MPI_Comm_rank( cart_comm, &proc ) );

  /* Store coordinates of MPI tasks */
  MPI_CALL( MPI_Cart_coords( cart_comm, proc, ndims, global_coords ) );

  /* Set left/right neighbouring tasks */
  MPI_CALL( MPI_Cart_shift( cart_comm, 0, 1, &global_neighbours[left], &global_neighbours[right] ) );

  /* Set down/up neighbouring tasks */
  MPI_CALL( MPI_Cart_shift( cart_comm, 1, 1, &global_neighbours[up], &global_neighbours[down] ) );

  /** 
   * Grid setup
   *
   * Calculate grid-related variables
   * Allocate memory for grid arrays
   */

  /* Set grid spacing in both directions */
  h = 1.0 / (double) global_N;

  /* Cache h^2 */
  h2 = h * h;
  
  /* Cache 1/(2h) */
  invtwoh = (double) global_N * 0.5;

  /* Calcalate local grid spacing */
  N = global_N / (int) sqrt( (double) nprocs + 0.5 );

  /* Calculate local grid start coordinates */
  global_start[x] = global_coords[x] * N; 
  global_start[y] = global_coords[y] * N;

#if VERBOSE>0
  if( proc == 0 ) {
    printf("\nParameters:\n");
    printf("====================\n");
    printf("P       = %d\n",nprocs);
    printf("N       = %d\n",global_N);
    printf("maxiter = %d\n",maxiter);
    printf("kappa   = %10.6lf\n",kappa);
    printf("rpx     = %10.6lf\n",rpx);
    printf("rmx     = %10.6lf\n",rmx);
    printf("kpi     = %10.6lf\n",kpi);
    printf("eps     = %10.6lf\n",eps);
    printf("h       = %10.6lf\n\n",h);
  }
  
  /* Testing neighbours correct */
  /* printf("Rank %d: l:%d r:%d u:%d d:%d\n",proc,
                  global_neighbours[left],global_neighbours[right],
                  global_neighbours[up],global_neighbours[down]); */

  /* Testing coordinates */
  /* printf("Rank %d: (%d,%d)\n",proc,
                  global_coords[x],global_coords[y]); */

  /* Testing local grid width/heights */
  /* printf("Rank %d: N = %d\n",proc,N); */

  /* Testing local grid start coordinates */
  /* printf("Rank %d: global_start coordinates (%d,%d)\n",proc,
                  global_start[x],global_start[y]); */
#endif

  /* Start timer */
  t1 = MPI_Wtime();

  /* Allocate columns of grids */
  p = (double **) malloc( N * sizeof(double *) );
  if( p == NULL ) printf("Error allocating p\n");

  p_new = (double **) malloc( N * sizeof(double *) );
  if( p_new == NULL ) printf("Error allocating p_new\n"); 

  rho = (double **) malloc( N * sizeof(double *) );
  if( rho == NULL ) printf("Error allocating rho\n"); 
  
  /* Allocate rows of grids */
  for( ix = 0; ix < N; ix++ ) {
    p[ix] = (double *) malloc( N * sizeof(double) );
    if( p[ix] == NULL ) printf("Error allocating row %d of %d in p\n",ix,N);

    p_new[ix] = (double *) malloc( N * sizeof(double) );
    if( p_new[ix] == NULL ) printf("Error allocating row %d of %d in p_new\n",ix,N);

    rho[ix] = (double *) malloc( N * sizeof(double) );
    if( rho[ix] == NULL ) printf("Error allocating row %d of %d in rho\n",ix,N);

    if( p == NULL || p_new == NULL || rho == NULL ) {
      /* Shut down MPI */
      MPI_CALL( MPI_Finalize() );

      /* Exit program */
      exit(EXIT_FAILURE);
    }
  }

  /* Set initial condition */
  for( ix = 0; ix < N; ix++ ) {
    for( iy = 0; iy < N; iy++ ) {
      p[ix][iy] = 0.0;
    }
  }

  /* Set up rhs */
  for( ix = 0; ix < N; ix++ ) {
    for( iy = 0; iy < N; iy++ ) {
      /* Calculate coordinates */
      coord_x = h * (double) ( ix + global_start[x] );
      coord_y = h * (double) ( iy + global_start[y] );

      rho[ix][iy] = kpi * ( exp( -1 * kappa * ( (coord_x - rpx)*(coord_x - rpx) 
                                              + (coord_y - rpy)*(coord_y - rpy) ) ) 
                          - exp( -1 * kappa * ( (coord_x - rmx)*(coord_x - rmx) 
                                              + (coord_y - rmy)*(coord_y - rmy) ) ) );
    }
  }

  /* Allocate halo buffers */
  halo_buffers = (double **) malloc( 4 * sizeof(double *) );
  if( halo_buffers == NULL ) printf("Error allocating halo_buffers\n");

  for( ix = 0; ix < 4; ix++ ) {
    halo_buffers[ix] = (double *) malloc( N * sizeof(double) );
    if( halo_buffers[ix] == NULL ) {
      printf("Error allocating row %d in halo_buffers\n",ix);
      
       /* Shut down MPI */
      MPI_CALL( MPI_Finalize() );

      /* Exit program */
      exit(EXIT_FAILURE);
    }
  }

  /* Allocate send and receive buffers */
  sendbuf = (double *) malloc( N * sizeof(double) );
  if( sendbuf == NULL ) printf("Error allocating sendbuf\n");

  recvbuf = (double *) malloc( N * sizeof(double) );
  if( recvbuf == NULL ) printf("Error allocating recvbuf\n");

  /* Check if any allocation failed */
  if( p == NULL            || p_new == NULL   || rho == NULL     || 
      halo_buffers == NULL || sendbuf == NULL || recvbuf == NULL ) {
    /* Shut down MPI */
    MPI_CALL( MPI_Finalize() );

    /* Exit program */
    exit(EXIT_FAILURE);
  }

#if VERBOSE>1   
  if( proc == 0 ) printf("rho[0][0] = %g\n",rho[0][0]);
#endif

  /** 
   * Main simulation loop
   *
   * The main loop is configured depending on whether
   * we have a fixed number of iterations, or 
   * we run the simulation to convergence
   *
   * We unroll loops to reduce redundancies and
   * reduce cache thrashing. The grid is computed on
   * in the following order
   *
   *             4
   *     1  ___________  7
   *       |           |
   *       |           |
   *     2 |     5     | 8
   *       |           |
   *       |           |
   *     3  -----------  9
   *             6
   *
   */
  
  /* Initiate iteration counter */
  iter = maxiter;

  /* Start main loop */
  while( iter != 0 ) {
#if VERBOSE>1
    if( proc == 0 && iter > 0 &&     iter  % interval == 0 ) printf("iter = %d  \t",maxiter-iter);
    if( proc == 0 && iter < 0 && abs(iter) % interval == 0 ) printf("iter = %d  \t",abs(iter));
#endif

    /* Perform halo swaps           */
    /* Only one processor available */
    if( nprocs == 1 ) {
      /* No need to perform halo swaps */
    } else {
      /* Copy left edge of p into send buffer */
      for( iy = 0; iy < N; iy++ ) {
        sendbuf[iy] = p[0][iy];
      }

      /* Send left edge to left neighbour. Receive left column of right neighbour */
      MPI_CALL( MPI_Sendrecv(sendbuf, N, MPI_DOUBLE,
                     global_neighbours[left],  proc + 10*global_neighbours[left],
                   recvbuf, N, MPI_DOUBLE,
                     global_neighbours[right], global_neighbours[right] + 10*proc,
                   cart_comm, &status) );

      /* Copy receive buffer into halo_buffers[right] */
      for( iy = 0; iy < N; iy++ ) {
        halo_buffers[right][iy] = recvbuf[iy];
      }

      /* Copy right edge of p into send buffer */
      for( iy = 0; iy < N; iy++ ) {
        sendbuf[iy] = p[N-1][iy];
      }

      /* Send right edge to right neighbour. Receive right column of left neighbour */
      MPI_CALL( MPI_Sendrecv(sendbuf, N, MPI_DOUBLE,
                     global_neighbours[right], proc + 10*global_neighbours[right],
                   recvbuf, N, MPI_DOUBLE,
                     global_neighbours[left],  global_neighbours[left] + 10*proc,
                   cart_comm, &status) );

      /* Copy receive buffer into halo_buffers[left] */
      for( iy = 0; iy < N; iy++ ) {
        halo_buffers[left][iy] = recvbuf[iy];
      }

      /* Copy bottom row of p into send buffer */
      for( ix = 0; ix < N; ix++ ) {
        sendbuf[ix] = p[ix][N-1];
      }

      /* Send bottom edge to bottom neighbour. Receive bottom edge of top neighbour */
      MPI_CALL( MPI_Sendrecv(sendbuf, N, MPI_DOUBLE,
                     global_neighbours[down], proc + 10*global_neighbours[down],
                   recvbuf, N, MPI_DOUBLE,
                     global_neighbours[up],   global_neighbours[up] + 10*proc,
                   cart_comm, &status) );

      /* Copy receive buffer into halo_buffers[up] */
      for( iy = 0; iy < N; iy++ ) {
        halo_buffers[up][iy] = recvbuf[iy];
      }

      /* Copy top row of p into send buffer */
      for( ix = 0; ix < N; ix++ ) {
        sendbuf[ix] = p[ix][0];
      }

      /* Send top edge to top neighbour. Receive top edge of bottom neighbour */
      MPI_CALL( MPI_Sendrecv(sendbuf, N, MPI_DOUBLE,
                     global_neighbours[up],   proc + 10*global_neighbours[up],
                   recvbuf, N, MPI_DOUBLE,
                     global_neighbours[down], global_neighbours[down] + 10*proc,
                   cart_comm, &status) );

      /* Copy receive buffer into halo_buffers[down] */
      for( iy = 0; iy < N; iy++ ) {
        halo_buffers[down][iy] = recvbuf[iy];
      }
    }

    /* Start computing new values */
    /* 1. Top-left corner */
    stencil_left = ( nprocs == 1 ) ? p[N-1][0] : halo_buffers[left][0];
    stencil_up   = ( nprocs == 1 ) ? p[0][N-1] : halo_buffers[up][0];
    p_new[0][0] = qt * ( p[1][0] + stencil_left + p[0][1] + stencil_up - h2 * rho[0][0] );

#if VERBOSE>1
    if( proc == 0 && abs(iter) % interval == 0 ) printf("p[0][0] = %g     \t",p[0][0]);
#endif

    /* 2. Left column */
    for( iy = 1; iy < N-1; iy++ ) {
      stencil_left = ( nprocs == 1 ) ? p[N-1][iy] : halo_buffers[left][iy];
      p_new[0][iy] = qt * ( p[1][iy] + stencil_left + p[0][iy+1] + p[0][iy-1] - h2 * rho[0][iy] );
    }

    /* 3. Bottom-left corner */
    stencil_left = ( nprocs == 1 ) ? p[N-1][N-1] : halo_buffers[left][N-1];
    stencil_down = ( nprocs == 1 ) ? p[0][0]     : halo_buffers[down][0];
    p_new[0][N-1] = qt * ( p[1][N-1] + stencil_left + stencil_down + p[0][N-2] - h2 * rho[0][N-1] );

    /* 4. Top row */
    for( ix = 1; ix < N-1; ix++ ) {
      stencil_up = ( nprocs == 1 ) ? p[ix][N-1] : halo_buffers[up][ix];
      p_new[ix][0] = qt * ( p[ix+1][0] + p[ix-1][0]  + p[ix][1] + stencil_up - h2 * rho[ix][0] );
    }
    
    /* 5. Internal grid */
    for( ix = 1; ix < N-1; ix++ ) {
      for( iy = 1; iy < N-1; iy++ ) {
        p_new[ix][iy] = qt * ( p[ix+1][iy] + p[ix-1][iy] + p[ix][iy+1] + p[ix][iy-1] - h2 * rho[ix][iy] );
      }
    }

    /* 6. Bottom row */
    for( ix = 1; ix < N-1; ix++ ) {
      stencil_down = ( nprocs == 1 ) ? p[ix][0] : halo_buffers[down][ix];
      p_new[ix][N-1] = qt * ( p[ix+1][N-1] + p[ix-1][N-1] + stencil_down + p[ix][N-2] - h2 * rho[ix][N-1] );
    }

    /* 7. Top-right corner */
    stencil_right = ( nprocs == 1 ) ? p[0][0]     : halo_buffers[right][0];
    stencil_up    = ( nprocs == 1 ) ? p[N-1][N-1] : halo_buffers[up][N-1];
    p_new[N-1][0] = qt * ( stencil_right + p[N-2][0] + p[N-1][1] + stencil_up - h2 * rho[N-1][0] );

    /* 8. Right column */
    for( iy = 1; iy < N-1; iy++ ) {
      stencil_right = ( nprocs == 1 ) ? p[0][iy] : halo_buffers[right][iy];
      p_new[N-1][iy] = qt * ( stencil_right + p[N-2][iy] + p[N-1][iy+1] + p[N-1][iy-1] - h2 * rho[N-1][iy] );
    }

    /* 9. Bottom-right corner */
    stencil_right = ( nprocs == 1 ) ? p[0][N-1] : halo_buffers[right][N-1];
    stencil_down  = ( nprocs == 1 ) ? p[N-1][0] : halo_buffers[down][N-1];
    p_new[N-1][N-1] = qt * ( stencil_right + p[N-2][N-1] + stencil_down + p[N-1][N-2] - h2 * rho[N-1][N-1] );

    /* Calculate error */
    maxerror = -1;
    for( ix = 0; ix < N; ix++ ) {
      for( iy = 0; iy < N; iy++ ) {
        error = fabs( p_new[ix][iy] - p[ix][iy] );
        if( error > maxerror ) maxerror = error;
      }
    }

    /* Collect global error */
    MPI_CALL( MPI_Allreduce( MPI_IN_PLACE, &maxerror, 1, MPI_DOUBLE, MPI_MAX, cart_comm ) );

#if VERBOSE>1
    if( proc == 0 && abs(iter) % interval == 0 ) printf("maxerror = %g\n",maxerror);
#endif

    /* Swap matrices */
    swap  = p;
    p     = p_new;
    p_new = swap;

    /* Exit loop if convergence criterion is met, else decrement counter */
    if( maxerror <= eps ) { enditer = iter; iter = 0; }
    else --iter;
  }

#if VERBOSE>0
    if( proc == 0 ) {
      printf("iter = %d  \t",(iter>0)?enditer:abs(enditer));
      printf("p[0][0] = %g     \t",p[0][0]);
      printf("maxerror = %g\n",maxerror);
    }
#endif

  /* Send buffer no longer required */
  free(sendbuf);

  /* Processor 0 still needs receive buffer */
  if( proc != 0 ) free(recvbuf);

  /** 
   * Simulation finalisation
   *
   * Processor 0 allocates memory for global grid 
   * and gathers all data from other processors.
   * Data is written out to files.
   */
  
  /* Only one processor so global grid is local grid */
  if( nprocs == 1 ) {
    /* Define global pointer to be local pointer */
    global_p = p;
  }
  else {
    if( proc == 0 ) {
      /* Allocate memory for global grid */
      global_p = (double **) malloc( global_N * sizeof(double *) );
      if( global_p == NULL ) printf("Error allocating global_p\n");

      for( ix = 0; ix < global_N; ix++ ) {
        global_p[ix] = (double *) malloc( global_N * sizeof(double) );
        if( global_p[ix] == NULL ) printf("Error allocating row %d of global_p\n",ix);
      }

      /* Copy processor 0's local grid to global grid */
      for( ix = 0; ix < N; ix++ ) {
        for( iy = 0; iy < N; iy++ ) {
          /* Calculate global indices */
          ixg = ix + global_start[x];
          iyg = iy + global_start[y];

          /* Perform copy */
          global_p[ixg][iyg] = p[ix][iy];
        }
      }
    }

    /* All processors wait for processor 0 */
    MPI_CALL( MPI_Barrier( cart_comm ) );

    /* Processor 0 collects data */
    if( proc == 0 ) {
      /* Loop over other processors */
      for( ip = 1; ip < nprocs; ip++ ) {
        /* Grab starting grid coordinates */ 
        MPI_CALL( MPI_Recv( &global_start, 2, MPI_INT, ip, ip, cart_comm, &status ) );

        /* Loop over columns of local grid */
        for( ix = 0; ix < N; ix++ ) {
          MPI_CALL( MPI_Recv( recvbuf, N, MPI_DOUBLE, ip, ix+(N*ip), cart_comm, &status ) );
        
          /* Copy column to global grid */
          for( iy = 0; iy < N; iy++ ) {
            /* Calculate global indices */
            ixg = ix + global_start[x];
            iyg = iy + global_start[y];

            /* Perform copy */
            global_p[ixg][iyg] = recvbuf[iy];
          }
        }
      }
    } else {          
      /* All other processors send data */
      
      /* Send starting grid coordinates */
      MPI_CALL( MPI_Send( &global_start, 2, MPI_INT, 0, proc, cart_comm ) );

      /* Loop over columns of local grid */
      for( ix = 0; ix < N; ix++ ) {
        /* Send columns (contiguous in memory) */
        MPI_CALL( MPI_Send( p[ix], N, MPI_DOUBLE, 0, ix+(N*proc), cart_comm) );
      }
    }
  }

  /* Stop timer */
  t2 = MPI_Wtime();

  /* Print out time */
  if( proc == 0 ) printf("Total simulation time took %12.6f seconds\n",t2-t1);

  /* Begin writing data to files */
  if( proc == 0 ) {
    /* Free processor 0's receive buffer */
    free( recvbuf );

    /** 
     * Write potential to file
     *
     * Structure of output file is:
     *   p_{0,0}   ... p_{0,N-1}
     *       .              .
     *       .              .
     *       .              .
     *   p_{N-1,0} ... p_{N-1,N-1}
     */
    fp = fopen( potential, "w" );
    if( fp == NULL ) printf("Error opening potential.dat for output\n");

    /* Write potential values to file */
 #if PRINTXYVAL==1
    for( ix = 0; ix < global_N; ix++ ) {
      for( iy = 0; iy < global_N; iy++ ) {
        coord_x = h * (double) ix;
        coord_y = h * (double) iy;
        fprintf(fp,"%g %g %g\n",coord_x,coord_y,global_p[ix][iy]);
      }
      fprintf(fp,"\n");
    }
 #else   
    for( ix = 0; ix < global_N; ix++ ) {
      for( iy = 0; iy < global_N; iy++ ) {
        fprintf(fp,"%g ",global_p[ix][iy]);
      }
      fprintf(fp,"\n");
    }
 #endif
    fclose(fp);

    /** 
     * Write vector field to file
     *
     * Structure of output file is:
     *   E_{0,0}[x]   E_{0,0}[y]   ... E_{0,N-1}[x]   E_{0,N-1}[y]
     *        .            .                 .              .
     *        .            .                 .              .
     *        .            .                 .              .
     *   E_{N-1,0}[x] E_{N-1,0}[y] ... E_{N-1,N-1}[x] E_{N-1,N-1}[y]
     */
    fp = fopen( field, "w" );
    if( fp == NULL ) printf("Error opening field.dat for output\n");

    /* Write field values to file */
    for( ix = 0; ix < global_N; ix++ ) {
      for( iy = 0; iy < global_N; iy++ ) {
        /* Calculate central difference (unoptimised) */
        if( ix == 0 ) {
          cdx = invtwoh * ( global_p[ix+1][iy] - global_p[global_N-1][iy] );
        } else if( ix == global_N-1 ) {
          cdx = invtwoh * ( global_p[0][iy] - global_p[ix-1][iy] );
        } else {
          cdx = invtwoh * ( global_p[ix+1][iy] - global_p[ix-1][iy] );
        }

        if( iy == 0 ) {
          cdy = invtwoh * ( global_p[ix][iy+1] - global_p[ix][global_N-1] );
        } else if( iy == global_N-1 ) {
          cdy = invtwoh * ( global_p[ix][0] - global_p[ix][iy-1] );
        } else {
          cdy = invtwoh * ( global_p[ix][iy+1] - global_p[ix][iy-1] );
        }

        /* Write to file */
        fprintf(fp,"%g %g ",cdx,cdy);
      }
      fprintf(fp,"\n");
    }
    fclose(fp);
   
    /* Free global grid */
    if( nprocs != 1 ) free( global_p );
  }

  /* Release grids */
  for( ix = 0; ix < N; ix++ ) {
    free( p[ix]     );
    free( p_new[ix] );
  }
  free( p     );
  free( p_new );

  free( potential );
  free( field     );

  /* Shutdown MPI */
  MPI_CALL( MPI_Finalize() );

  return EXIT_SUCCESS;

}

/**
 * strdup()
 *
 * Returns duplicate of string
 */
char *strdup( const char *str ) {
  int   n   = strlen(str) + 1;
  char *dup = malloc(n);
  if( dup ) strcpy(dup, str);
  return dup;
}
