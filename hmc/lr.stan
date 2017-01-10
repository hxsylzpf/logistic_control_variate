data {
    int N;
    int D;
    int y[N];
    vector[D] X[N];
}
parameters {
    row_vector[D] beta;
}
model {
    for (n in 1:N)
        y[n] ~ bernoulli_logit(beta * X[n]);
}
