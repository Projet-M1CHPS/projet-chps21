
__kernel void normalizeCharToFloat(__global unsigned char *input, __global float *output,
                                   ulong offset, float factor) {
  const int row = get_global_id(0);
  output[row + offset] = (float) input[row];
  output[row + offset] /= factor;
}