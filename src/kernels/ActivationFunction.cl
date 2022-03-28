

__kernel void identity(__global float *ptr) {}

__kernel void didentity(__global float *ptr) {}

float _sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }

__kernel void sigmoid(__global float *ptr) {
  int id = get_global_id(0);
  ptr[id] = _sigmoid(ptr[id]);
}

__kernel void dsigmoid(__global float *ptr) {
  int id = get_global_id(0);
  ptr[id] = _sigmoid(ptr[id]) * (1 - _sigmoid(ptr[id]));
}

__kernel void relu(__global float *ptr) {
  int id = get_global_id(0);
  ptr[id] = fmax(0.f, ptr[id]);
}

__kernel void drelu(__global float *ptr) {
  int id = get_global_id(0);
  ptr[id] = ptr[id] > 0 ? 1 : 0;
}

__kernel void leakyRelu(__global float *ptr) {
  int id = get_global_id(0);
  ptr[id] = ptr[id] > 0 ? ptr[id] : 0.01f * ptr[id];
}

__kernel void dleakyRelu(__global float *ptr) {
  int id = get_global_id(0);
  ptr[id] = ptr[id] > 0 ? 1 : 0.01f;
}

__kernel void square(__global float *ptr) {
  int id = get_global_id(0);
  ptr[id] = ptr[id] * ptr[id];
}

__kernel void dsquare(__global float *ptr) {
  int id = get_global_id(0);
  ptr[id] = 2 * ptr[id];
}