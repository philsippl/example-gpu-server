// This is obviously a stupid sum implementation, but it's just for testing.
extern "C" __global__ void sum_buffer(unsigned char* buffer, int size, int* result) {
    for (int i = 0; i < size; i++) {
        result[0] += buffer[i];
    }
}
