functions {
  // Funzione base senza xc
  real lambda_direct(array[] real theta,
                     array[] real x_r,
                     real t) {

    real c1 = x_r[1];
    real c2 = x_r[2];
    real c3 = x_r[3];
    real c4 = x_r[4];
    real c5 = x_r[5];
    real c6 = x_r[6];
    real c7 = x_r[7];

    array[7] real q;
    array[4] real a;
    array[4] real b;

    for (i in 1:7) q[i] = theta[i];
    for (i in 1:4) a[i] = theta[7+i];
    for (i in 1:4) b[i] = theta[11+i];

    real omega = 2 * pi() / 7 / 24;

    real linpred = q[1]*c1 + q[2]*c2 + q[3]*c3 + q[4]*c4 +
      q[5]*c5 + q[6]*c6 + q[7]*c7 +
      a[1]*cos(omega*t) + b[1]*sin(omega*t) +
      a[2]*cos(omega*7*t) + b[2]*sin(omega*7*t) +
      a[3]*cos(omega*14*t) + b[3]*sin(omega*14*t) +
      a[4]*cos(omega*21*t) + b[4]*sin(omega*21*t);

    linpred = fmin(linpred, 20.0);
    linpred = fmax(linpred, -20.0);
    linpred = exp(linpred);
    linpred = fmax(linpred, 1e-10);

    return linpred;
  }


  // a, b: estremi integrazione
  // passo fisso = 0.4
  real trap_integrate(real a, real b,
                      array[] real theta,
                      array[] real x_r,
                      array[] int x_i) {

    real h = 0.4;
    real integral = 0;
    real x = a;

    while (x < b) {
      real next_x = fmin(x + h, b);
      integral += 0.5 * (lambda_direct(theta, x_r, x) + lambda_direct(theta, x_r, next_x)) * (next_x - x);
      x = next_x;
    }
    return integral;

}
}

data {
  int<lower=0> N;
  array[N] real t;
  array[N] real<lower=0> y;
  array[N] real<lower=0, upper=1> censored;
  array[N] real<lower=0, upper=1> CR;
  array[N] real<lower=0, upper=1> MT;
  array[N] real<lower=0, upper=1> MLB;
  array[N] real<lower=0, upper=1> MLR;
  array[N] real<lower=0, upper=1> SI;
  array[N] real<lower=0, upper=1> SIU;
  array[N] real<lower=0, upper=1> T;
}

transformed data {
  array[N, 7] real x_r_data;
  for (i in 1:N) {
    x_r_data[i, 1] = CR[i];
    x_r_data[i, 2] = MT[i];
    x_r_data[i, 3] = MLB[i];
    x_r_data[i, 4] = MLR[i];
    x_r_data[i, 5] = SI[i];
    x_r_data[i, 6] = SIU[i];
    x_r_data[i, 7] = T[i];
  }

  array[N, 0] int x_i_data;
}

parameters {
  vector[7] raw_q;
  vector[4] raw_a;
  vector[4] raw_b;
}

transformed parameters {
  real sd = 0.28;

  vector[7] q = sd * raw_q;
  vector[4] a = sd * raw_a;
  vector[4] b = sd * raw_b;

  array[15] real theta;
  for (i in 1:7) theta[i] = q[i];
  for (i in 1:4) theta[7+i] = a[i];
  for (i in 1:4) theta[11+i] = b[i];

  vector[N] lambda;
  vector[N] Lambda;

  for (i in 1:N) {
    // lambda diretto
    lambda[i] = lambda_direct(theta, x_r_data[i, ], t[i]);

    // lambda integrato
    if (y[i] < 0.4) {
      Lambda[i] = lambda[i] * y[i];
    } else {
      Lambda[i] = trap_integrate(t[i], t[i] + y[i],
                               theta, x_r_data[i, ],
                               x_i_data[i, ]);
    }
  }
}

model {
  raw_q ~ std_normal();
  raw_a ~ std_normal();
  raw_b ~ std_normal();

  for (i in 1:N) {
    if (censored[i] == 1) {
      target += -Lambda[i];
    } else {
      target += log(lambda[i]) - Lambda[i];
    }
  }
}
