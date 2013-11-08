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
#include <time.h>

/* Set level of verbosity       */
/*  0 = No messages             */
/*  1 = Print parameters        */
/*  2 = Print debug information */
#define VERBOSE 2

/* Set style of file output     */
/*  0 = Exact matrix p          */
/*  1 = (x,y,p(x,y))            */
#define PRINTXYVAL 0

/* Define pi if not in math.h */
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Function prototypes */
char *strdup( const char *str );

/* Main program logic */
int main( int argc, char const *argv[] ) {
  
  /** 
   * Constants definitions
   *
   * Configure size of grid, simulation settings and problem parameters.
   */

  const unsigned int N = 2400;                      /* Size of grid */
    
  const unsigned int maxiter = 1000;                /* Number of iterations. Set to -1 to run to convergence */
      
  const double kappa = 100.0;                       /* kappa */
  const double rpx = 0.3, rpy = 0.3;                /* +1 charge centre r_\plus */
  const double rmx = 0.7, rmy = 0.7;                /* -1 charge centre r_\minus */
    
  const double eps = 0.000001;                      /* Convergence criterion */
    
  const double qt = 0.25;                           /* Four-point stencil factor */
  const double kpi = kappa / M_PI;                  /* Constant factor in rhs */

  const int interval = 100;                         /* Print out interval */

  char *potential = strdup("potential.dat");        /* Potential output filename */
  char *field = strdup("field.dat");                /* Field output filename */

  /** 
   * Variable definitions
   *
   * Simulation variables, including grids and errors
   */
          
  double **p, **p_new, **rho, **temp;               /* Grids */
  double h, h2, invtwoh;                            /* Grid spacing, and its square */
    
  unsigned int ix, iy, iter, enditer;               /* Counters */

  double coord_x, coord_y;                          /* Cartesian coordinates */
    
  double error, maxerror;                           /* Temporary error variables */

  double cdx, cdy;                                  /* Central difference variables */
    
  FILE *fp;                                         /* File pointer */

  clock_t t1, t2;                                   /* Clocks */

#if VERBOSE>0
  printf("\nParameters:\n");
  printf("====================\n");
  printf("N       = %d\n",N);
  printf("maxiter = %d\n",maxiter);
  printf("kappa   = %10.6lf\n",kappa);
  printf("rpx     = %10.6lf\n",rpx);
  printf("rmx     = %10.6lf\n",rmx);
  printf("kpi     = %10.6lf\n",kpi);
  printf("eps     = %10.6lf\n",eps);
#endif

  /* Start timer */
  t1 = clock();

  /** 
   * Grid setup
   *
   * Calculate grid-related variables
   * Allocate memory for grid arrays
   */

  /* Set grid spacing in both directions */
  h = 1.0 / (double) N;

#if VERBOSE>0
  printf("h       = %10.6lf\n\n",h);
#endif

  /* Cache h^2 */
  h2 = h * h;

  /* Cache 1/(2h) */
  invtwoh = (double) N * 0.5;

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
      coord_x = h * (double) ix;
      coord_y = h * (double) iy;

      rho[ix][iy] = kpi * ( exp( -1 * kappa * ( (coord_x - rpx)*(coord_x - rpx) 
                                              + (coord_y - rpy)*(coord_y - rpy) ) ) 
                          - exp( -1 * kappa * ( (coord_x - rmx)*(coord_x - rmx) 
                                              + (coord_y - rmy)*(coord_y - rmy) ) ) );
    }
  }

#if VERBOSE>1   
  printf("rho[0][0] = %g\n",rho[0][0]);
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
    if( iter > 0 &&     iter  % interval == 0 ) printf("iter = %d  \t",maxiter-iter);
    if( iter < 0 && abs(iter) % interval == 0 ) printf("iter = %d  \t",abs(iter));
#endif
  
    /* 1. Top-left corner */
    p_new[0][0] = qt * ( p[1][0] + p[N-1][0] + p[0][1] + p[0][N-1] - h2 * rho[0][0] );

#if VERBOSE>1
    if( abs(iter) % interval == 0 ) printf("p[0][0] = %g     \t",p[0][0]);
#endif

    /* 2. Left column */
    for( iy = 1; iy < N-1; iy++ ) {
      p_new[0][iy] = qt * ( p[1][iy] + p[N-1][iy] + p[0][iy+1] + p[0][iy-1] - h2 * rho[0][iy] );
    }

    /* 3. Bottom-left corner */
    p_new[0][N-1] = qt * ( p[1][N-1] + p[N-1][N-1] + p[0][0] + p[0][N-2] - h2 * rho[0][N-1] );

    /* 4. Top row */
    for( ix = 1; ix < N-1; ix++ ) {
      p_new[ix][0] = qt * ( p[ix+1][0] + p[ix-1][0] + p[ix][1] + p[ix][N-1] - h2 * rho[ix][0] );
    }
    
    /* 5. Internal grid */
    for( ix = 1; ix < N-1; ix++ ) {
      for( iy = 1; iy < N-1; iy++ ) {
        p_new[ix][iy] = qt * ( p[ix+1][iy] + p[ix-1][iy] + p[ix][iy+1] + p[ix][iy-1] - h2 * rho[ix][iy] );
      }
    }

    /* 6. Bottom row */
    for( ix = 1; ix < N-1; ix++ ) {
      p_new[ix][N-1] = qt * ( p[ix+1][N-1] + p[ix-1][N-1] + p[ix][0] + p[ix][N-2] - h2 * rho[ix][N-1] );
    }

    /* 7. Top-right corner */
    p_new[N-1][0] = qt * ( p[0][0] + p[N-2][0] + p[N-1][1] + p[N-1][N-1] - h2 * rho[N-1][0] );

    /* 8. Right column */
    for( iy = 1; iy < N-1; iy++ ) {
      p_new[N-1][iy] = qt * ( p[0][iy] + p[N-2][iy] + p[N-1][iy+1] + p[N-1][iy-1] - h2 * rho[N-1][iy] );
    }

    /* 9. Bottom-right corner */
    p_new[N-1][N-1] = qt * ( p[0][N-1] + p[N-2][N-1] + p[N-1][0] + p[N-1][N-2] - h2 * rho[N-1][N-1] );

    /* Calculate error */
    maxerror = -1;
    for( ix = 0; ix < N; ix++ ) {
      for( iy = 0; iy < N; iy++ ) {
        error = fabs( p_new[ix][iy] - p[ix][iy] );
        if( error > maxerror ) maxerror = error;
      }
    }

#if VERBOSE>1
    if( abs(iter) % interval == 0 ) printf("maxerror = %g\n",maxerror);
#endif

    /* Swap matrices */
    temp  = p;
    p     = p_new;
    p_new = temp;

    /* Exit loop if convergence criterion is met, else decrement counter */
    if( maxerror <= eps ) { enditer = iter; iter = 0; }
    else --iter;
  }

#if VERBOSE>0
  printf("iter = %d  \t",(iter>0)?enditer:abs(enditer));
  printf("p[0][0] = %g     \t",p[0][0]);
  printf("maxerror = %g\n",maxerror);
#endif

  /** 
   * Simulation finalisation
   *
   * Processor 0 allocates memory for global grid 
   * and gathers all data from other processors.
   * Data is written out to files.
   */
  
  t2 = clock();

  /* Print out time */
  printf("Total simulation time took %12.6f seconds\n",(double)(t2-t1)/CLOCKS_PER_SEC);

  /** 
   * Write potential to file
   *
   * Structure of output file is:
   * p_{0,0}   ... p_{0,N-1}
   *     .              .
   *     .              .
   *     .              .
   * p_{N-1,0} ... p_{N-1,N-1}
   */
  fp = fopen( potential, "w" );
  if( fp == NULL ) printf("Error opening potential.dat for output\n");

  /* Write potential values to file */
#if PRINTXYVAL==1
  for( ix = 0; ix < N; ix++ ) {
    for( iy = 0; iy < N; iy++ ) {
      coord_x = h * (double) ix;
      coord_y = h * (double) iy;
      fprintf(fp,"%g %g %g\n",coord_x,coord_y,p[ix][iy]);
    }
    fprintf(fp,"\n");
  }
#else   
  for( ix = 0; ix < N; ix++ ) {
    for( iy = 0; iy < N; iy++ ) {
      fprintf(fp,"%g ",p[ix][iy]);
    }
    fprintf(fp,"\n");
  }
#endif
  fclose(fp);

  /** 
   * Write vector field to file
   *
   * Structure of output file is:
   * E_{0,0}[x]   E_{0,0}[y]   ... E_{0,N-1}[x]   E_{0,N-1}[y]
   *      .            .                 .              .
   *      .            .                 .              .
   *      .            .                 .              .
   * E_{N-1,0}[x] E_{N-1,0}[y] ... E_{N-1,N-1}[x] E_{N-1,N-1}[y]
   */
  fp = fopen( field, "w" );
  if( fp == NULL ) printf("Error opening field.dat for output\n");

  /* Write field values to file */
  for( ix = 0; ix < N; ix++ ) {
    for( iy = 0; iy < N; iy++ ) {
      /* Calculate central difference (unoptimised) */
      if( ix == 0 ) {
        cdx = invtwoh * ( p[ix+1][iy] - p[N-1][iy] );
      } else if( ix == N-1 ) {
        cdx = invtwoh * ( p[0][iy] - p[ix-1][iy] );
      } else {
        cdx = invtwoh * ( p[ix+1][iy] - p[ix-1][iy] );
      }

      if( iy == 0 ) {
        cdy = invtwoh * ( p[ix][iy+1] - p[ix][N-1] );
      } else if( iy == N-1 ) {
        cdy = invtwoh * ( p[ix][0] - p[ix][iy-1] );
      } else {
        cdy = invtwoh * ( p[ix][iy+1] - p[ix][iy-1] );
      }

      /* Write to file */
      fprintf(fp,"%g %g ",cdx,cdy);
    }
    fprintf(fp,"\n");
  }
  fclose(fp);

  /* Release dynamically allocated memory */
  for( ix = 0; ix < N; ix++ ) {
    free( p[ix]     );
    free( p_new[ix] );
  }
  free( p     );
  free( p_new );

  free( potential );
  free( field     );

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
