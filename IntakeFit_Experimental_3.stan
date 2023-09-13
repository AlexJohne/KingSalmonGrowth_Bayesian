// IntakeFit_Experimental_3.stan
// Shane.Richards@utas.edu.au
// 06/09/2023

data {
  int<lower=1>         I;       // number of observations
  int<lower=1>         J;       // number of individuals
  int<lower=1>         K;       // number of samples (time x tank)
  int<lower=1>         D;       // number of days
  int<lower=1>         S;       // number of sample times
  real<lower=0>    w_std;       // standard weight 
  int<lower=1,upper=J> j[I];    // fish ID
  int<lower=1,upper=K> k[I];    // sample ID
  real<lower=0>        w[I];    // fish weight
  real<lower=-1>   w_rel[I];   // relative fish weights (0 = of average weight)
  real<lower=0>   Intake[I];   // fish intake
  real<lower=-1>     m_r[J,D]; // weight relative to cohort
  int<lower=0>       m_d[J,S]; // times
  int<lower=0>       v_w[J];   // number of weights per individual
  real<lower=0.0>    m_w[J,S]; // weights
}

parameters {
  // normal parameters
  real<lower=  1.00,upper= 2.0>  ln_a1;     // log-intake (at standard weight)
  real<lower= -0.75,upper= 0.05> ln_b1;     // intake-weight power
  real<lower= -1.0, upper= 1.0>  c1;        // relative weight effect on intake
  real<lower= -4.0, upper= 1.0>  ln_a2;     // log-cov (at standard weigth)
  real<lower= -1.0, upper= 1.0>  b2;        // weight effect on cov
  real<lower=  0.01,upper= 1.0>  sigma_k;   // sd of sample variation
  real<lower=  0.5, upper= 1.0>  ln_eta;    // log-intake-weight conversion
  real<lower=  1.0, upper= 2.0>  ln_a3;     // log-cost (at standard weight)
  real<lower= -0.5, upper= 1.0>  ln_b3;     // cost-weight power
  real<lower= -1.5, upper= 1.0>  c2;        // relative weight effect on cost
  real<lower= -4.0, upper=-2.0>  ln_a4;     // log-cov (at standard weigth)
  
  vector<lower=0>[2] sigma_u;  // sigma_I, sigma_C
  cholesky_factor_corr[2] L_u; // correlation 
  matrix[2,J] z_u;             // unscaled individual I,C deviations
  vector[K] RE_k;              // sample deviations
}

transformed parameters { 
  matrix[2,J] u; // scaled individual I,C deviations

  u = diag_pre_multiply(sigma_u, L_u) * z_u;  // subj random effects
} 

model {
  // local variables
  real al; real be;
  real mu; real cov;
  real eta;
  real a1; real b1;
  real a2;
  real a3; real b3;
  real a4;
  vector[2] muRE;
  vector[D] wP; // predicted weights (day-to-day)
  
  muRE[1] = 0.0; muRE[2] = 0.0;

  // priors on model parameters
  // intake parameters
  ln_a1 ~ normal( 1.50, 0.25); //  
  ln_b1 ~ normal(-0.35, 0.20); //  
  c1    ~ normal( 0.00, 0.50); //  
  ln_a2 ~ normal(-1.00, 2.00); //  
  b2    ~ normal( 0.00, 0.50); //  
  // growth parameters
  ln_eta ~ normal( 0.3, 0.4); //  
  ln_a3  ~ normal( 1.5, 0.2); //  
  ln_b3  ~ normal( 0.0, 0.4); //  
  c2     ~ normal( 0.0, 0.5); //  
  ln_a4  ~ normal(-3.0, 1.0); //  

  L_u ~ lkj_corr_cholesky(2.0); // prior on correlation
  to_vector(z_u) ~ normal(0,1); // prior on individual deviations

  RE_k ~ normal(0.0, sigma_k);  // random effect across samples 

  // calculate positive model parameters
  a1  = exp(ln_a1);  // base-line intake
  b1  = exp(ln_b1);  // intake-weight power
  a2  = exp(ln_a2);  // intake observation error (gamma dist)
  a3  = exp(ln_a3);  // base-line metabolism 
  b3  = exp(ln_b3);  // metabolism-weight power
  a4  = exp(ln_a4);  // weight observation error (gamma dist)
  eta = exp(ln_eta); // intake -> mass conversion constant

  for (i in 1:I) { // for each observation
    // expected intake for observation i
    mu = a1*exp(c1*w_rel[i] + u[1,j[i]] + RE_k[k[i]])*pow(w[i]/w_std, b1);
    
    // coefficient of variation in intake for observation i
    cov = a2*exp(b2*(w[i] - w_std)/w_std); // coefficient of variation
    
    // gamma parameters
    al = 1.0/(cov*cov);
    be = al/mu;
    // add log-likelihood term associated with observation i
    target += gamma_lpdf(Intake[i] | al, be); // add species log-likelihood term
  }
  
  for (jj in 1:J) {
    if (v_w[jj] > 1) { // more than 1 weight so growth to predict
      for (ss in 1:(v_w[jj] - 1)) { // for each growth interval
        // predict weight change for indivual jj from sample time ss -> ss+1
        wP[m_d[jj,ss]] = m_w[jj,ss]; // initial weight
        for (dd in (m_d[jj,ss]):(m_d[jj,ss+1]-1)) { // update each day
          wP[dd+1] = wP[dd] + 
            eta*a1*exp(c1*m_r[jj,dd] + u[1,jj])*pow(wP[dd]/w_std, b1) -
                a3*exp(c2*m_r[jj,dd] + u[2,jj])*pow(wP[dd]/w_std, b3);
        }
        al = 1.0/(a4*a4);         // gamma - alpha parameter
        be = al/wP[m_d[jj,ss+1]]; // gamma - beta parameter
        target += gamma_lpdf(m_w[jj,ss+1] | al, be);
      }
    }
  }
}

generated quantities {
  // calculate the positive model parameters and return them
  matrix[2,2] CM; // scaled individual I,C deviations
  
  real a1; real b1;
  real a2;
  real a3; real b3;
  real a4;
  real eta;
  real rho; // correlatio coefficient

  a1 = exp(ln_a1);
  b1 = exp(ln_b1);
  a2 = exp(ln_a2);
  a3 = exp(ln_a3);
  b3 = exp(ln_b3);
  a4 = exp(ln_a4);
  eta = exp(ln_eta);
  //extract correlation parameters
  CM = diag_pre_multiply(sigma_u, L_u); 
  // correlation between intake and cost deviations
  rho = CM[1,1]*CM[2,1] / pow(CM[1,1]*CM[1,1]*(CM[2,1]*CM[2,1] + CM[2,2]*CM[2,1]), 0.5);
}

