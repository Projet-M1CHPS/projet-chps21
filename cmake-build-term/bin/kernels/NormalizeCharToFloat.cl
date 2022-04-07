
__kernel void normalizeCharToFloat(__global unsigned char *input, __global float *output,
                                   float factor, ulong width, ulong height) {
  const int row = get_global_id(0);
  // const int col = get_global_id(1);
  output[row] = (float) input[row] * factor;
}