

float sign(float x) {
  if (fabs(x) < 1e-6) {
    return 0;
  } else if (x > 0) {
    return 1;
  }
  return -1;
}

__kernel void rprop_update(__global float *weights, __global float *gradient,
                           __global float *old_gradient, __global float *w_update,
                           __global float *old_w_update) {
  const int i = get_global_id(0);
  float weight_change = 0.0;
  float change = sign(gradient[i] * old_gradient[i]);

  // The gradient is converging in the same direction as the previous update
  // Increase the delta to converge faster
  if (change > 0.0) {
    float delta = fmin(weights_update[i] * eta_plus, update_max);
    weight_change = delta * sign(gradient[i]);
    weights_update[i] = delta;
    old_gradient[i]= gradient[i];
  }
  // The gradient as changed direction
  // Rollback and reduce the delta to converge slower
  else if (change < 0) {
    float delta = fmax(weights_update[i] * eta_minus, update_min);
    weights_update[i] = delta;
    weight_change = -last_weights_change[i];
    old_gradient[i] = 0.0;
  }
  // No need to change the delta
  else if (change == 0) {
    float delta = weights_update[i];
    weight_change = sign(gradient[i]) * delta;
    old_gradient[i] = gradient[i];
  }
  weights[i] -= weight_change;
  last_weights_change[i] = weight_change;
}