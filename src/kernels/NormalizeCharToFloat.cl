
__kernel
void normalizeCharToFloat(__global unsigned char *input, __global float *output, float factor, ulong size)
{
    const int i = get_global_id(0);
    output[i] = (float)input[i] / factor;
}