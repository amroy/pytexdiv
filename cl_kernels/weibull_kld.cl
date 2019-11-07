#define EPS 0.0f

__kernel void weibull_kld(__global const float *models, __global float *divs, int M, int D) {
 
    // Get the index of the current element to be processed
    //int idx = get_global_id(0);
    //int i = idx / M; // The current row
    //int j = idx % M; // The current column
    int i = get_global_id(0);
    int j = get_global_id(1);
    float a1, b1, a2, b2, div = 0.0f, lambda = 0.5772f;

    /**
     * The weibull model is coded as follows {a00 b00 a01 b01 a02 b02 ... aij bij ... aLJ bLJ}
     * with i the scale index and j the subband index.
     */
    for (int k=0; k<D; k+=2) {
    	a1 = models[i*D + k] + EPS; b1 = models[i*D + k+1] + EPS;
    	a2 = models[j*D + k] + EPS; b2 = models[j*D + k+1] + EPS;
        div += log(b1 / (pow(a1, b1))) - log(b2 / pow(a2, b2)) + (log(a1) - lambda/b1)*(b1-b2) + pow((a1/a2), b2) * tgamma(b2/b1+1) - 1.0f;
    }
 
    divs[i*M+j] = div;
}
