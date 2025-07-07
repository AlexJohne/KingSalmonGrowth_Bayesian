// IntakeFit_Experimental_6.stan
// Shane.Richards@utas.edu.au
// alexandra.johne@utas.edu.au
// 01/07/2025

data {
  int<lower=1>         I;      // number of observations
  int<lower=1>         J;      // number of individuals
  int<lower=1>         K;      // number of samples (time x tank)
  int<lower=1>         D;      // number of days
  int<lower=1>         S;      // number of sample times
  real<lower=0>    w_std;      // standard weight 
  int<lower=1,upper=J> j[I];   // fish ID
  int<lower=1,upper=K> k[I];   // (time x tank) ID
  real<lower=0>        w[I];   // fish weight
  real<lower=-1>   w_rel[I];   // relative fish weights (0 = of average weight)
  real<lower=-1>    w_mu[I];   // relative fish weights (0 = of average weight)
  real<lower=0>   Intake[I];   // fish intake
  real<lower=-1>     m_r[J,D]; // weight relative to cohort
  real<lower=0>     m_mu[J,D]; // mean weight to cohort
  int<lower=0>       m_d[J,S]; // times
  int<lower=0>       v_w[J];   // number of weights per individual
  real<lower=0.0>    m_w[J,S]; // weights
}

parameters {
  // intake
  real<lower=  0.75,  upper= 2.0>     ln_a1;  // log-intake (at standard weight)
  real<lower= -0.75,  upper= 0.5>     ln_b1;   // intake-weight power
  real<lower= -0.75,  upper= 0.75>    c10;    // relative weight effect on intake
  real<lower= -0.005, upper= 0.005>   c11;    // relative weight effect on intake
  real<lower= -4.0,   upper=-0.25>    ln_cv1; // log-cov intake (at standard weigth)
  real<lower=  0.25,  upper= 1.0>     ln_eta; // log-intake-weight conversion
  real<lower=  0.5,   upper= 2.0>     ln_a2;  // log-cost (at standard weight)
  real<lower= -0.5,   upper= 1.0>     ln_b2;  // cost-weight power
  real<lower= -1.5,   upper= 1.0>     c20;    // relative weight effect on cost
  real<lower= -0.005, upper= 0.005>   c21;    // relative weight effect on cost
  real<lower= -3.5,   upper=-1.25>    ln_cv2; // log-cov cost (at standard weigth)
  real<lower= -1.0,   upper= 2.0>     g1;
  real<lower= -1.0,   upper= 2.0>     g2;

  vector<lower=0>[2] sigma_u;  // sigma_I, sigma_C
  cholesky_factor_corr[2] L_u; // correlation
  matrix[2,J] z_u;             // unscaled individual I,C deviations
  
  real  <lower=-3.0, upper=  1.0>    ln_sigma_RE;
  vector<lower=-3.0, upper=  3.0>[K] sample_RE;   // estimated skipper variation (random effect)
}

model {
  // local variables
  real al; real be;
  real mu; real cov;
  real gamm;
  real eta; real sigma_RE;
  real a1; real b1; real cv1;
  real a2; real b2; real cv2;
  vector[2] muRE;
  matrix[J,D] wP; // predicted weights (individual, day-to-day)
  
  matrix[2,J] z_u2; // scaled individual I,C deviations
  matrix[2,J] u; // predicted weights (individual, day-to-day)
  
  muRE[1] = 0.0; muRE[2] = 0.0;

  // priors on model parameters
  // intake parameters
  ln_a1  ~ normal( 1.60, 0.20);  //  log-intake at standard weight
  ln_b1  ~ normal(-0.35, 0.10);  //  log-power intake
  c10    ~ normal( 0.00, 0.50);  //  relative weight effect on intake
  c11    ~ normal( 0.00, 0.50);  //  relative weight effect on intake
  ln_cv1 ~ normal(-1.00, 1.00);  //  log-cov intake
  // growth parameters
  ln_eta ~ normal( 0.3, 0.2); // log conversion intake -> weight gain
  ln_a2  ~ normal( 1.5, 0.2); // log-cost at standard weight
  ln_b2  ~ normal( 0.0, 0.2); // log-power cost
  c20    ~ normal( 0.0, 0.1); // relative weight effect on cost
  c21    ~ normal( 0.0, 0.1); // relative weight effect on cost
  ln_cv2 ~ normal(-2.0, 1.0); // log-cov cost
  
  g1 ~ normal(0.0, 0.5);
  g2 ~ normal(0.0, 0.5);

  L_u ~ lkj_corr_cholesky(2.0); // prior on correlation
  to_vector(z_u) ~ normal(0,1); // prior on individual deviations
  
  ln_sigma_RE  ~ normal(-2.0, 1.0);
  sample_RE    ~ normal( 0.0, 1.0); // sample random effect

  // calculate positive model parameters
  a1   = exp(ln_a1);   // intake at standard weight
  b1   = exp(ln_b1);   // power intake
  cv1  = exp(ln_cv1); // cov intake
  eta  = exp(ln_eta);  // conversion intake -> weight gain
  a2   = exp(ln_a2);   // base-line metabolism 
  b2   = exp(ln_b2);   // metabolism-weight power
  cv2  = exp(ln_cv2); // weight observation error (gamma dist)
  sigma_RE  = exp(ln_sigma_RE);
  
  for (i in 1:J) {
  	z_u2[1,i] = z_u[1,i]*(1.0 + g1*z_u[1,i]);
  	z_u2[2,i] = z_u[2,i]*(1.0 + g2*z_u[2,i]);
  	u[1,i] = sigma_u[1]*z_u2[1,i];
  	u[2,i] = sigma_u[2]*(L_u[2,1]*z_u2[1,i] + L_u[2,2]*z_u2[2,i]);
  }

  for (i in 1:I) { // for each observation
    // expected intake for observation i
    mu = a1 * exp(sigma_RE*sample_RE[k[i]]) * pow(w[i]/w_std, b1) *
      exp((c10 + c11*w_mu[i])*w_rel[i] + u[1,j[i]]);

    // gamma parameters
    al = 1.0/(cv1*cv1); // cov = cv1
    be = al/mu;
    
    // add log-likelihood term associated with observation i
    target += gamma_lpdf(Intake[i] | al, be); // add species log-likelihood term
  }
  
  for (jj in 1:J) { // for each individual
    if (v_w[jj] > 1) { // more than 1 weight so growth to predict
      for (ss in 1:(v_w[jj] - 1)) { // for each growth interval
        // predict weight change for indivual jj from sample time ss -> ss+1
        wP[jj,m_d[jj,ss]] = m_w[jj,ss]; // initial weight
        for (dd in (m_d[jj,ss]):(m_d[jj,ss+1]-1)) { // update each day
          wP[jj,dd+1] = wP[jj,dd] +
            eta*a1*exp((c10 + c11*m_mu[jj,dd])*m_r[jj,dd] + u[1,jj])*pow(wP[jj,dd]/w_std, b1) -
                a2*exp((c20 + c21*m_mu[jj,dd])*m_r[jj,dd] + u[2,jj])*pow(wP[jj,dd]/w_std, b2);
        }

        al = 1.0/(cv2*cv2);          // gamma - alpha parameter
        be = al/wP[jj,m_d[jj,ss+1]]; // gamma - beta parameter

        target += gamma_lpdf(m_w[jj,ss+1] | al, be);
      }
    }
  }
}

generated quantities {
  // calculate the positive model parameters and return them
  real a1; real b1; real cv1;
  real a2; real b2; real cv2;
  real eta;
  real sigma_RE;
  real rho; // correlation coefficient

  matrix[2,J] z_u2; // scaled individual I,C deviations
  matrix[2,J] u; // predicted weights (individual, day-to-day)

  a1    = exp(ln_a1);
  b1    = exp(ln_b1);
  cv1   = exp(ln_cv1);
  a2    = exp(ln_a2);
  b2    = exp(ln_b2);
  cv2   = exp(ln_cv2);
  eta   = exp(ln_eta);
  sigma_RE = exp(ln_sigma_RE);

  // correlation between intake and cost deviations
  rho = L_u[2,1];
  
  for (i in 1:J) {
    z_u2[1,i] = z_u[1,i]*(1.0 + g1*z_u[1,i]);;
    z_u2[2,i] = z_u[2,i]*(1.0 + g2*z_u[2,i]);
    u[1,i]    = sigma_u[1]*z_u2[1,i];
    u[2,i]    = sigma_u[2]*(L_u[2,1]*z_u2[1,i] + L_u[2,2]*z_u2[2,i]);
  }
}

