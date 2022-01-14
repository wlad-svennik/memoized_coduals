type Codual = (Real, Real -> Real)


fwd_sin (x, dx) = (sin x, \k -> cos x * dx k)

rev_sin (x, dx) = (sin x, \k -> dx (cos x * k))


fwd_mul (u, du) (v, dv) = (u * v, \k -> u * dv k + du k * v)

rev_mul (u, du) (v, dv) = (u * v, \k -> dv (u * k) + dv (k * v))


fwd_add (u, du) (v, dv) = (u + v, \k -> du k + dv k)

rev_add (u, du) (v, dv) = (u + v, \k -> du k + dv k)
