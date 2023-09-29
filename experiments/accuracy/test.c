#include <stdio.h>
#include<stdlib.h>
#include<string.h>

#ifdef __BAMBU_SIM__
#include <mdpi/mdpi_user.h>
#endif

float *a;
float *b;
float *c;

float *c_expected;
int *labels;
int nodes;
int classes = 7;

extern void forward_kernel(float *a, float *b, float *c);

int load_data(FILE *logfile) {
    FILE *file = fopen("data.bin", "rb");
    if (file == NULL) {
        perror("Error opening file");
        return 1;
    }

    // labels data
    int labels_dim;
    fread(&labels_dim, sizeof(int), 1, file);
    labels = (int *) malloc(labels_dim * sizeof(int));
    fread(labels, sizeof(int), labels_dim, file);
    fprintf(logfile, "Labels dimension: %d\n", labels_dim);
    for (int i = 0; i < labels_dim; i++)
        fprintf(logfile, "%d ", labels[i]);
    fprintf(logfile, "\n");
    nodes = labels_dim;

    // features data
    int features_rows, features_cols;
    fread(&features_rows, sizeof(int), 1, file);
    fread(&features_cols, sizeof(int), 1, file);

    a = (float *) malloc(features_rows * features_cols * sizeof(float));
    fread(a, sizeof(float), features_rows * features_cols, file);

    fprintf(logfile, "Features dimension: %dx%d\n", features_rows, features_cols);
    for (int i = 0; i < features_rows; i++) {
        for (int j = 0; j < features_cols; j++) {
            fprintf(logfile, "%f ", a[i * features_cols + j]);
        }
        fprintf(logfile, "\n");
    }


    // adjacency matrix
    int adj_rows, adj_cols;
    fread(&adj_rows, sizeof(int), 1, file);
    fread(&adj_cols, sizeof(int), 1, file);
    b = (float *) malloc(adj_rows * adj_cols * sizeof(float));
    fread(b, sizeof(float), adj_rows * adj_cols, file);

    fprintf(logfile, "Adjacency matrix dimension: %dx%d\n", adj_rows, adj_cols);
    for (int i = 0; i < adj_rows; i++) {
        for (int j = 0; j < adj_cols; j++) {
            fprintf(logfile, "%f ", b[i * adj_cols + j]);
        }
        fprintf(logfile, "\n");
    }

    // output data
    int out_rows, out_cols;
    fread(&out_rows, sizeof(int), 1, file);
    fread(&out_cols, sizeof(int), 1, file);
    c_expected = (float *) malloc(out_rows * out_cols * sizeof(float));
    fread(c_expected, sizeof(float), out_rows * out_cols, file);

    fprintf(logfile, "Expected output dimension: %dx%d\n", out_rows, out_cols);
    for (int i = 0; i < out_rows; i++) {
        for (int j = 0; j < out_cols; j++) {
            fprintf(logfile, "%f ", c_expected[i * out_cols + j]);
        }
        fprintf(logfile, "\n");
    }


    // malloc for c result
    c = (float *) malloc(out_rows * out_cols * sizeof(float));

    memset(c, 0, sizeof(float) * out_rows * out_cols);
#ifdef __BAMBU_SIM__
    m_param_alloc(0, sizeof(a)*features_rows*features_cols);
    m_param_alloc(1, sizeof(b)*adj_rows*adj_cols);
    m_param_alloc(2, sizeof(c)*out_rows*out_cols);
#endif

    fclose(file);
    return 0;
}

float accuracy() {
    // assumption: test set = 20% of the dataset
    // take the row of the output matrix which correspond to the test set
    // pick the column index of the maximum element of those line
    int test_set_dim = (int) nodes * 0.2;
    float acc = 0.0;
    int gold_classified = 0;

    for (int i = nodes - test_set_dim; i < nodes; i++) {
        int tmp_class = -1;
        float tmp_value = -1000000.0;
        for (int j = 0; j < classes; j++) {
            if (c[i * classes + j] > tmp_value) {
                tmp_value = c[i * classes + j];
                tmp_class = j;
            }
        }
        if (tmp_class == labels[i])
            gold_classified++;
    }

    acc = (float) gold_classified / test_set_dim;
    return acc;
}

// total error computed as the sum of the difference of the output matrix only for test nodes (last test set dim rows)
float error() {
    float total_error = 0.0;
    int test_set_dim = (int) nodes * 0.2;
    for (int i = nodes - test_set_dim; i < nodes; i++) {
        for (int j = 0; j < classes; j++) {
            float diff = c[i * classes + j] - c_expected[i * classes + j];
            if (diff < 0)
                diff = 0 - diff;
            total_error += diff;
        }
    }

    return total_error;
}

int diff_cells() {
    int count = 0;

    int test_set_dim = (int) nodes * 0.2;
    for (int i = nodes - test_set_dim; i < nodes; i++) {
        for (int j = 0; j < classes; j++) {
            float diff = c[i * classes + j] - c_expected[i * classes + j];
            if (diff != 0)
                count++;
        }
    }
    return count;
}

int main() {
    FILE *logfile = fopen("cora_FPGA_data.log", "w");

    load_data(logfile);

    forward_kernel(a, b, c);

    fprintf(logfile, "*****************\n");

    fprintf(logfile, "Computed output dimension: %dx%d\n", nodes, classes);
    for (int i = 0; i < nodes; i++) {
        for (int j = 0; j < classes; j++) {
            fprintf(logfile, "%f ", c[i * 7 + j]);
        }
        fprintf(logfile, "\n");
    }

    fprintf(logfile, "Model accuracy: %f\n", accuracy());

    float total_error = error();
    int diff_values = diff_cells();
    fprintf(logfile, "Floating total error: %.10f\n", total_error);
    fprintf(logfile, "Floating values with error: %d\n", diff_values);
    fprintf(logfile, "Floating average error: %.10f\n", (float) total_error / diff_values);


    return 0;

}
