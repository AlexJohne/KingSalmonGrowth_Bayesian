// IntakeFit.stan
// Shane.Richards@utas.edu.au
// 27/04/2023

data {
  int<lower=1>         I;     // number of observations
  int<lower=1>         J;     // number of individuals
  int<lower=1>         K;     // number of samples
  real<lower=0>    w_std;     // standard weight 
  int<lower=1,upper=J> j[I];  // fish ID
  int<lower=1,upper=K> k[I];  // sample ID
  real<lower=0>        w[I];  // fish weight
  real<lower=-1>    w_rel[I]; // mean fish weights
  real<lower=0>    Intake[I]; // fish intake
}
 
parameters {
	// normal parameters
  real<lower=  1.2,upper=  1.8>   ln_a1; // log-intake (at standard weight)
  real<lower= -1.5,upper= -0.4>   ln_b1; // intake-weight power
  real<lower= -1.0,upper=  1.0>   c1;    // relative weight effect
  real<lower= -1.0,upper= -0.5>   ln_a2; // log-cov (at standard weigth)
  real<lower= -0.01,upper=  0.01> b2;    // weight effect on cov
  real<lower=  0.01,upper=1.0> sigma_j;  // sd of individual variation
  real<lower=  0.01,upper=1.0> sigma_k;  // sd of sample variation
  vector[J] RE_j;                        // fish deviations
  vector[K] RE_k;                        // sample deviations
}

model {
	// local variables
	real al;
	real be;
	real mu;
	real cov;
	real a1;
	real a2;
	real b1;

	// priors on model parameters
	ln_a1 ~ normal( 1.5, 0.1); //  
	ln_b1 ~ normal(-0.75, 0.25); //  
	c1    ~ normal( 0.0, 0.25); //  
	ln_a2 ~ normal(-0.7, 1.00); //  
	b2    ~ normal( 0.0, 0.005); //  
  sigma_j ~ exponential(2.0); // sd of individual variation
  sigma_k ~ exponential(2.0); // sd of sample variation
	RE_j  ~ normal(0, sigma_j); // random effect across individuals 
	RE_k  ~ normal(0, sigma_k); // random effect across samples 

  // calculate positive model parameters
	a1 = exp(ln_a1);
	b1 = exp(ln_b1);
	a2 = exp(ln_a2);

  for (i in 1:I) { // for each species
    // expected intake for obseravtion i
    mu = a1*exp(c1*w_rel[i] + RE_j[j[i]] + RE_k[k[i]])*pow(w[i]/w_std, b1);
    // coefficient of variation in intake for observation i
    cov = a2*exp(b2*(w[i] - w_std)); // 
    // gamma parameters
    al = 1.0/(cov*cov);
    be = al/mu;
    // add log-likelihood term associated with observation i
    target += gamma_lpdf(Intake[i] | al, be); // add species log-likelihood term
  }
}

generated quantities {
	 // calculate the positive modle parameters and return them
   real a1;
   real b1;
   real a2;

   a1 = exp(ln_a1);
   b1 = exp(ln_b1);
   a2 = exp(ln_a2);
}

