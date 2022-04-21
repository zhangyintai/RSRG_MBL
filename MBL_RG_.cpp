#include <stdio.h>
#include <stdlib.h>
#include <mathimf.h>
#include <memory.h>
#include <mpi.h>
#include "mkl_lapacke.h"

#define Max(a, b) ((a) < (b) ? (b) : (a))
#define Min(a, b) ((a) > (b) ? (b) : (a))
#define Abs(a) ((a) > (0) ? (a) : -(a))

const long int m_random = 2147483647; //set m=a*q+r;  randomz: 100000001; 16807: 2147483647
const long int a_random = 16807;      //set a; rand	omz: 329; 16807: 16807
const long int q_random = 127773;     // m_random / a_random; //set q
const long int r_random = 2836;       //m_random % a_random; //set r
const long int b_random = 0;          //set b

const double pi = 3.141592653589793;
const double e = 2.718281828459045;

const double epsilon = 1E-9; // if absolute value smaller than epsilon, set to zero.
long int SEED_random = 20200720;

const int N_h = 10001;

int StringToInt(char *a);
long int StringToLongInt(char *a);
double StringToDouble(char *a);

void ExpRandom(int L, double *random, double W, double alpha);
void MultiPointRandom(int L, double *random, double W, double n);

struct stack_int // Stack structure for integer variables
{
    int *a;
    int len;

    struct stack_int *prev;
    struct stack_int *next;
    struct stack_int *end;
};
typedef struct stack_int stack_int;

struct stack_double // Stack structure for double variables
{
    double *a;
    int len;

    struct stack_double *prev;
    struct stack_double *next;
    struct stack_double *end;
};
typedef struct stack_double stack_double;

struct Cluster // Cluster structure
{
    stack_int *spins;

    double bandwidth;
    double level_spacing;
    int m_spins;
    int num;
    int combined;
};
typedef struct Cluster Cluster;

struct stack_Cluster // Stack structure for integer variables
{
    Cluster *a;

    int len;

    struct stack_Cluster *prev;
    struct stack_Cluster *next;
    struct stack_Cluster *end;
};
typedef struct stack_Cluster stack_Cluster;

// struct Sample_QP
// {
//     int N_sample;
//     double *A; //amplitude
//     double *k;
//     double *phi;
// };
// typedef struct Sample_QP Sample_QP;

struct Sample
{
    int N_sample;
    double W;
};
typedef struct Sample Sample;

struct Single_particle
{
    double energy;
    double pos;
    int wave_function;
};
typedef struct Single_particle Single_particle;

// These functions are basic stack operations
void init_stack_int(stack_int *S);
void push_stack_int(int b, stack_int *S);
int pop_stack_int(stack_int *S);
void destroy_stack_int(stack_int **S);
void combine_stack_int(stack_int **a, stack_int **b);
void print_stack_int(stack_int *S);

void init_stack_double(stack_double *S);
void push_stack_double(double b, stack_double *S);
double pop_stack_double(stack_double *S);
void destroy_stack_double(stack_double **S);
void combine_stack_double(stack_double **a, stack_double **b);
void print_stack_double(stack_double *S);

void init_stack_Cluster(stack_Cluster *S);
void push_stack_Cluster(Cluster *b, stack_Cluster *S);
Cluster *pop_stack_Cluster(stack_Cluster *S);
void destroy_stack_Cluster(stack_Cluster **S, int clear_cluster);
void combine_stack_Cluster(stack_Cluster **a, stack_Cluster **b);
void print_stack_Cluster(stack_Cluster *S);

Single_particle *Anderson(int L, double V, double *random, double *coupling, double t, int print_H, double W, double uncert, int rank);
int cmp_func_double(const void *a, const void *b);
int cmp_func_Single_particle(const void *a, const void *b);

double Uncert(int L, int N_random, double t, double W, char random_mode, double para);

// Combine two clusters
int combine_cluster(stack_Cluster *merge, Cluster **a, Cluster **b,
                    double *coupling, int num_of_clusters);

Cluster *Merge_cluster(stack_Cluster *merge, int num,
                       int num_of_clusters, double *coupling);

void update_coupling(stack_Cluster *Clusters, double **coupling,
                     int num_of_clusters, int num_of_clusters_new,
                     stack_int **merged_m_spin_list, stack_int **merged_num_list);

double mismatch(Cluster *a, Cluster *b); // Calculate mismatch

//Build a spin chain with interactions and local random
void build_sample(int N_spin, double W, double t, double V,
                  char random_mode, double para, int rank,
                  stack_Cluster *Clusters, double **coupling, double uncert);

double calculation(int L, stack_Cluster **Clusters, double **coupling); // Calculate entropy

// These two functions are for generating random numbers
long int Schrage(long int z);
double Random();

void gen_int_table(double *int_table, double *h_table, double alpha, int N_h, double W);
int Bsearch(double key, double *a, int N_a, int i_min, int i_max);
void Inv_table_random(double *potential, int N, int N_h, double *int_table, double *h_table);
void random_pm(double *potential, int N);

void Print_Result(double *RESULT, int num_of_procs);

extern void print_matrix(char *desc, MKL_INT m, MKL_INT n, double *a, MKL_INT lda);
void DSYEVD(int L, double *a, double *w, int print);

int main(int argc, char *argv[])
{
    //Stack test

    // stack_int *a, *b;
    // a = (stack_int *)malloc(sizeof(stack_int));
    // b = (stack_int *)malloc(sizeof(stack_int));

    // init_stack_int(a);
    // init_stack_int(b);

    // push_stack_int(1, a);
    // push_stack_int(2, a);
    // push_stack_int(2, b);
    // combine_stack_int(&a, &b);
    // printf("%d\n", *((a->next)->a));
    // printf("%d\n", *(((a->next)->next)->a));
    // printf("%d\n", *(((a->end))->a));
    // printf("%d\n", pop_stack_int(a));
    // destroy_stack_int(&a);

    // return 0;

    MPI_Init(&argc, &argv);

    double t, V, W_max, W_min;
    int L, N_random, N_w;
    double para;
    char random_mode;
    L = StringToInt(argv[1]);
    t = StringToDouble(argv[2]);
    V = StringToDouble(argv[3]);
    W_min = StringToDouble(argv[4]);
    W_max = StringToDouble(argv[5]);
    N_w = StringToDouble(argv[6]);
    N_random = StringToInt(argv[7]);
    random_mode = *(argv[8]);
    para = StringToDouble(argv[9]);

    int ii, jj;

    int num_of_procs = 0, rank = 0;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_of_procs);

    long int i, j;
    double sum[1] = {0};

    SEED_random = SEED_random + rank;

    stack_Cluster **Clusters = NULL;
    double **coupling = NULL;

    if (rank == 0)
    {
        printf("Number of cores: %d\n\n", num_of_procs);
        printf("L = %d, W_min = %lf, W_max = %lf\n", L, W_min, W_max);
        printf("t = %lf, V = %lf\n", t, V);
        printf("N_group = %d, random_mode = %c, para = %lf\n", num_of_procs / N_w, random_mode, para);
    }

    double uncert =  1.2;// Uncert(L, N_random, t, (W_max - W_min) / (double)(N_w - 1) * (double)(rank % N_w) + W_min, random_mode, para);
    // Uncert = Uncert(L, t, (W_max - W_min) / (double)(N_w - 1) * (double)(rank % N_w) + W_min);

    for (i = 0; i < N_random; i++)
    {
        Clusters = (stack_Cluster **)malloc(sizeof(stack_Cluster *));
        *Clusters = (stack_Cluster *)malloc(sizeof(stack_Cluster));

        init_stack_Cluster(*Clusters);

        coupling = (double **)malloc(sizeof(double *));
        *coupling = (double *)malloc(sizeof(double) * L * L);

        memset(*Clusters, 0, sizeof(stack_Cluster *));
        memset(*coupling, 0, sizeof(double) * L * L);

        if (N_w != 1)
        {
            build_sample(L, (W_max - W_min) / (double)(N_w - 1) * (double)(rank % N_w) + W_min,
                         t, V, random_mode, para, rank, *Clusters, coupling, uncert);
        }
        else
        {

            build_sample(L, W_max,
                         t, V, random_mode, para, rank, *Clusters, coupling, uncert);
        }

        double temp_result = calculation(L, Clusters, coupling);
        sum[0] = sum[0] + temp_result;

        destroy_stack_Cluster(Clusters, 1);

        free(*Clusters);
        *Clusters = NULL;

        free(Clusters);
        Clusters = NULL;

        free(*coupling);
        *coupling = NULL;

        free(coupling);
        coupling = NULL;
    }

    sum[0] = sum[0] / ((double)N_random);

    double RESULT[num_of_procs];
    memset(RESULT, 0, sizeof(double) * num_of_procs);

    int displ[num_of_procs], receive_count[num_of_procs];

    for (i = 0; i < num_of_procs; i++)
    {
        displ[i] = i;
        receive_count[i] = 1;
    }

    MPI_Gatherv(sum, 1, MPI_DOUBLE,
                RESULT, receive_count, displ,
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        Print_Result(RESULT, num_of_procs);
    }

    MPI_Finalize();
    exit(0);
    return 0;
    // printf("------------------------------------------------\n");
}

void init_stack_int(stack_int *S)
{
    S->a = NULL;
    S->prev = NULL;
    S->next = NULL;
    S->end = S;
    S->len = 0;
}

void push_stack_int(int b, stack_int *S)
{
    if (S == NULL)
    {
        return;
    }

    (S->end)->next = (stack_int *)malloc(sizeof(stack_int));
    ((S->end)->next)->prev = S->end;

    (S->end) = (S->end)->next;

    (S->end)->a = (int *)malloc(sizeof(int));
    *((S->end)->a) = b;

    (S->end)->next = NULL;
    (S->len) = (S->len) + 1;

    (S->end->len) = 0;
}

int pop_stack_int(stack_int *S)
{
    if (S == NULL)
    {
        printf("Exit in pop_stack_int\n");

        exit(1);
    }

    if (S->len == 0)
    {
        printf("Exit in pop_stack_int\n");

        exit(1);
    }

    int result;

    result = *((S->end)->a);

    stack_int *new_end = (S->end)->prev;

    free((S->end)->a);
    (S->end)->a = NULL;

    free(S->end);
    S->end = NULL;

    S->end = new_end;

    (S->end)->next = NULL;
    S->len = (S->len) - 1;

    return result;
}

void destroy_stack_int(stack_int **S)
{

    if (*S == NULL)
    {
        return;
    }

    stack_int *temp;
    temp = (*S)->end;

    while ((temp->a) != NULL)
    {
        pop_stack_int(*S);
        temp = (*S)->end;
    }

    free(*S);
    *S = NULL;
}

void combine_stack_int(stack_int **a, stack_int **b)
{
    if (*a == NULL)
    {
        printf("Exit in combine_stack_int");

        exit(1);
    }

    if (*b == NULL)
    {
        return;
    }

    ((*a)->end)->next = (*b)->next;
    ((*b)->next)->prev = (*a)->end;

    (*a)->end = (*b)->end;
    (*a)->len = (*a)->len + (*b)->len;

    free(*b);
    *b = NULL;
}

void print_stack_int(stack_int *S)
{
    if (S == NULL)
    {
        printf("Not initialized!");
        exit(1);
    }

    stack_int *temp = S->next;

    if (temp == NULL)
    {
        printf("Empty!");
    }

    while (temp != NULL)
    {
        printf("%d ", *(temp->a));
        temp = temp->next;
    }

    printf("\n");
}

void init_stack_double(stack_double *S)
{
    S->a = NULL;
    S->prev = NULL;
    S->next = NULL;
    S->end = S;
    S->len = 0;
}

void push_stack_double(double b, stack_double *S)
{
    if (S == NULL)
    {
        return;
    }

    (S->end)->next = (stack_double *)malloc(sizeof(stack_double));
    ((S->end)->next)->prev = S->end;

    (S->end) = (S->end)->next;

    (S->end)->a = (double *)malloc(sizeof(double));
    *((S->end)->a) = b;

    (S->end)->next = NULL;
    (S->len) = (S->len) + 1;

    (S->end->len) = 0;
}

double pop_stack_double(stack_double *S)
{
    if (S == NULL)
    {
        printf("Exit in pop_stack_double\n");

        exit(1);
    }

    if (S->len == 0)
    {
        printf("Exit in pop_stack_double\n");

        exit(1);
    }

    double result;

    result = *((S->end)->a);

    stack_double *new_end = (S->end)->prev;

    free((S->end)->a);
    (S->end)->a = NULL;

    free(S->end);
    S->end = NULL;

    S->end = new_end;

    (S->end)->next = NULL;
    S->len = (S->len) - 1;

    return result;
}

void destroy_stack_double(stack_double **S)
{

    if (*S == NULL)
    {
        return;
    }

    stack_double *temp;
    temp = (*S)->end;

    while ((temp->a) != NULL)
    {
        pop_stack_double(*S);
        temp = (*S)->end;
    }

    free(*S);
    *S = NULL;
}

void combine_stack_double(stack_double **a, stack_double **b)
{
    if (*a == NULL)
    {
        printf("Exit in combine_stack_double");

        exit(1);
    }

    if (*b == NULL)
    {
        return;
    }

    ((*a)->end)->next = (*b)->next;
    ((*b)->next)->prev = (*a)->end;

    (*a)->end = (*b)->end;
    (*a)->len = (*a)->len + (*b)->len;

    free(*b);
    *b = NULL;
}

void print_stack_double(stack_double *S)
{
    if (S == NULL)
    {
        printf("Not initialized!");
        exit(1);
    }

    stack_double *temp = S->next;

    if (temp == NULL)
    {
        printf("Empty!");
    }

    while (temp != NULL)
    {
        printf("%lf ", *(temp->a));
        temp = temp->next;
    }

    printf("\n");
}

void init_stack_Cluster(stack_Cluster *S)
{
    S->a = NULL;
    S->prev = NULL;
    S->next = NULL;
    S->end = S;
    S->len = 0;
}

void push_stack_Cluster(Cluster *b, stack_Cluster *S)
{
    if (S == NULL)
    {
        printf("Exit in push_stack_Cluster\n");

        exit(1);
    }

    (S->end)->next = (stack_Cluster *)malloc(sizeof(stack_Cluster));
    ((S->end)->next)->prev = S->end;

    (S->end) = (S->end)->next;

    (S->end)->a = b;

    (S->end)->next = NULL;
    (S->len) = (S->len) + 1;
    (S->end->len) = 0;
}

Cluster *pop_stack_Cluster(stack_Cluster *S)
{
    if (S == NULL)
    {
        printf("Exit in pop_stack_Cluster\n");

        exit(1);
    }

    if (S->len == 0)
    {
        printf("Exit in pop_stack_Cluster\n");

        exit(1);
    }

    Cluster *result;

    result = (S->end)->a;

    stack_Cluster *new_end = (S->end)->prev;

    free(S->end);
    S->end = NULL;

    S->end = new_end;
    (S->end)->next = NULL;
    S->len = (S->len) - 1;

    return result;
}

void destroy_stack_Cluster(stack_Cluster **S, int destroy_cluster)
{

    if (S == NULL)
    {
        return;
    }

    Cluster *temp;

    while ((*S)->len > 0)
    {
        temp = pop_stack_Cluster(*S);
        if (temp != NULL)
        {
            destroy_stack_int(&(temp->spins));
        }
        else
        {
            continue;
        }
        if (destroy_cluster == 1)
        {
            free(temp);
            temp = NULL;
        }
    }

    free(*S);
    *S = NULL;
}

void combine_stack_Cluster(stack_Cluster **a, stack_Cluster **b)
{
    if (*a == NULL)
    {
        printf("Exit in combine_stack_Cluster\n");

        exit(1);
    }

    if (*b == NULL)
    {
        return;
    }

    ((*a)->end)->next = (*b)->next;
    ((*b)->next)->prev = (*a)->end;

    (*a)->end = (*b)->end;
    (*a)->len = (*a)->len + (*b)->len;

    free(*b);
    *b = NULL;
}

void print_stack_Cluster(stack_Cluster *S)
{
    if (S == NULL)
    {
        printf("Exit in print_stack_Cluster\n");

        exit(1);
    }

    stack_Cluster *temp;
    temp = S->next;

    while (temp != NULL)
    {
        printf("Cluster #: %d, number of spins: %d\n",
               (temp->a)->num, (temp->a)->m_spins);
        printf("bandwidth: %lf, level spacing: %lf\n",
               (temp->a)->bandwidth, (temp->a)->level_spacing);
        printf("Spins:\n");
        print_stack_int((temp->a)->spins);
        printf("---------------------------------------\n");

        temp = temp->next;
    }
    printf("\n");
}

int cmp_func_double(const void *a, const void *b)
{
    if ((*(double *)a - *(double *)b) > 0)
    {
        return 1;
    }
    else
    {
        return -1;
    }
}

int cmp_func_Single_particle(const void *a, const void *b)
{
    if (((*(Single_particle *)a).pos - (*(Single_particle *)b).pos) > 0)
    {
        return 1;
    }
    else
    {
        return -1;
    }
}

Single_particle *Anderson(int L, double V, double *random, double *coupling, double t, int print_H, double W, double uncert, int rank)
{
    double *H;
    H = (double *)malloc(sizeof(double) * L * L);

    Single_particle *spin_chain;
    spin_chain = (Single_particle *)malloc(sizeof(Single_particle) * L);

    memset(spin_chain, 0, sizeof(spin_chain) * L);

    memset(H, 0, sizeof(double) * L * L);

    int i, j, k;

    *(H + 0 * L + 0) = *(random + 0);

    for (i = 1; i < L; i++)
    {
        *(H + i * L + (i - 1)) = t;
        *(H + i * L + i) = *(random + i);
    }

    if (print_H == 1 && rank == 0)
    {
        for (i = 0; i < L; i++)
        {
            printf("[");
            for (j = 0; j < L; j++)
            {
                if (i >= j)
                {
                    printf("% 6.2lf,", *(H + i * L + j));
                }
                else
                {
                    printf("% 6.2lf,", *((H + j * L + i)));
                }
            }
            printf("],\n");
        }
    }

    memset(random, 0, sizeof(double) * L);

    DSYEVD(L, H, random, print_H);
    double temp_pos;
    double x0 = uncert; 
    // double temp_x0 = 2.0 / log(1.0 + 4.0 * W * W / t / t);
    // printf("%lf, %lf\n", x0, temp_x0);
    // double x0 = 2.0 / log(1.0 + 4.0 * W * W / t / t);

    // double uncert = 0;
    // double pospos;

    for (i = 0; i < L; i++) //Enum eigenstates
    {
        temp_pos = 0;
        // pospos = 0;

        for (j = 0; j < L; j++) //Enum sites
        {
            // \tilde{c_i} = \sum_{j} a_{ij} c_j, a_{ij}=*(H+i*L + j)
            // c_j = \sum_{i}a_{ji}c_i
            temp_pos = temp_pos + ((double)j) * (*(H + i * L + j)) * (*(H + i * L + j));

            // pospos = pospos + ((double)j) * ((double)j) * (*(H + i * L + j)) * (*(H + i * L + j));
        }
        // uncert = uncert + sqrt(pospos - temp_pos * temp_pos);

        (spin_chain + i)->energy = *(random + i) - *(random + 0);
        (spin_chain + i)->pos = temp_pos;
        (spin_chain + i)->wave_function = i;

        // if (rank == 0)
        // {
        //     printf("(%lf, %lf), ", (spin_chain+i)->pos, (spin_chain+i)->energy);
        // }
    }

    // if (rank == 0)
    // {
    // printf("\n\n");
    // }

    // uncert = uncert / ((double)L);

    // printf("%lf, %lf\n", uncert, Uncert);

    qsort(spin_chain, L, sizeof(Single_particle), cmp_func_Single_particle);

    // if (rank == 0)
    // {
    //     printf("After sort\n");
    // }

    double temp_coup = 0;
    // double max_coup = -1;

    int ii;
    for (i = 0; i < L; i++)
    {
        /*
        if (rank == 0)
        {
            printf("(%lf, %lf), ", (spin_chain+i)->pos, (spin_chain+i)->energy);
        }
        */

        for (j = i + 1; j < L; j++)
        {
            /*
            temp_coup = 0;

            for (ii = 0; ii < L; ii++)
            {
                break;
                if ((ii == i) || (ii == j))
                {
                    continue;
                }
                for (k = 0; k < L - 1; k++)
                {
                    temp_coup = temp_coup - (*(H + ((spin_chain + i)->wave_function) * L + k + 1)) *
                                                (*(H + ((spin_chain + ii)->wave_function) * L + k)) *
                                                (*(H + ((spin_chain + ii)->wave_function) * L + k + 1)) *
                                                (*(H + ((spin_chain + j)->wave_function) * L + k));

                    temp_coup = temp_coup + (*(H + ((spin_chain + i)->wave_function) * L + k + 1)) *
                                                (*(H + ((spin_chain + ii)->wave_function) * L + k)) *
                                                (*(H + ((spin_chain + ii)->wave_function) * L + k)) *
                                                (*(H + ((spin_chain + j)->wave_function) * L + k + 1));

                    temp_coup = temp_coup + (*(H + ((spin_chain + i)->wave_function) * L + k)) *
                                                (*(H + ((spin_chain + ii)->wave_function) * L + k + 1)) *
                                                (*(H + ((spin_chain + ii)->wave_function) * L + k + 1)) *
                                                (*(H + ((spin_chain + j)->wave_function) * L + k));

                    temp_coup = temp_coup - (*(H + ((spin_chain + i)->wave_function) * L + k)) *
                                                (*(H + ((spin_chain + ii)->wave_function) * L + k + 1)) *
                                                (*(H + ((spin_chain + ii)->wave_function) * L + k)) *
                                                (*(H + ((spin_chain + j)->wave_function) * L + k + 1));
                }
            }
            
        
            for (k = 0; k < L - 1; k++)
            {
                temp_coup = temp_coup - (*(H + ((spin_chain + i)->wave_function) * L + k + 1)) *
                                            (*(H + ((spin_chain + j)->wave_function) * L + k)) *
                                            (*(H + ((spin_chain + i)->wave_function) * L + k + 1)) *
                                            (*(H + ((spin_chain + j)->wave_function) * L + k));

                temp_coup = temp_coup + (*(H + ((spin_chain + i)->wave_function) * L + k + 1)) *
                                            (*(H + ((spin_chain + j)->wave_function) * L + k)) *
                                            (*(H + ((spin_chain + i)->wave_function) * L + k)) *
                                            (*(H + ((spin_chain + j)->wave_function) * L + k + 1));

                temp_coup = temp_coup + (*(H + ((spin_chain + i)->wave_function) * L + k)) *
                                            (*(H + ((spin_chain + j)->wave_function) * L + k + 1)) *
                                            (*(H + ((spin_chain + i)->wave_function) * L + k + 1)) *
                                            (*(H + ((spin_chain + j)->wave_function) * L + k));

                temp_coup = temp_coup - (*(H + ((spin_chain + i)->wave_function) * L + k)) *
                                            (*(H + ((spin_chain + j)->wave_function) * L + k + 1)) *
                                            (*(H + ((spin_chain + i)->wave_function) * L + k)) *
                                            (*(H + ((spin_chain + j)->wave_function) * L + k + 1));
            }
            */
            *(coupling + i + L * j) = V * pow(e, -Abs((spin_chain + i)->pos - (spin_chain + j)->pos) / x0);
            /*
            if (max_coup < V * Abs(temp_coup))
            {
               max_coup = V * Abs(temp_coup);
            }
            */

            *(coupling + i * L + j) = *(coupling + i + L * j);
        }
    }
    /*
    if (rank < 40)
    {
        printf("max_coup: %lf, W: %lf, rank: %d\n", max_coup, W, rank);
    }
    */
    /*
    if (rank==0)
    {
        printf("\n\n");
    }
    */

    // qsort(spin_chain, L, sizeof(spin_chain), cmp_func_Single_particle);

    free(H);
    H = NULL;

    return spin_chain;
}

double Uncert(int L, int N_random, double t, double W, char random_mode, double para)
{
    double *H;

    double *w;

    H = (double *)malloc(sizeof(double) * L * L);
    w = (double *)malloc(sizeof(double) * L);

    double temp_pos;

    double uncert = 0;
    double pospos;

    int i, j, k;

    for (k = 0; k < N_random; k++)
    {
        memset(H, 0, sizeof(double) * L * L);

        if (random_mode == 'E')
        {
            ExpRandom(L, w, W, para);
        }
        else if (random_mode == 'M')
        {
            MultiPointRandom(L, w, W, para);
        }

        for (i = 1; i < L; i++)
        {
            *(H + i * L + (i - 1)) = t;
            *(H + i * L + i) = *(w + i);
        }

        memset(w, 0, sizeof(double) * L);

        DSYEVD(L, H, w, 0);

        for (i = 0; i < L; i++) //Enum eigenstates
        {
            temp_pos = 0;
            pospos = 0;

            for (j = 0; j < L; j++) //Enum sites
            {
                // \tilde{c_i} = \sum_{j} a_{ij} c_j, a_{ij}=*(H+i*L + j)
                // c_j = \sum_{i}a_{ji}c_i
                temp_pos = temp_pos + ((double)j) * (*(H + i * L + j)) * (*(H + i * L + j));

                pospos = pospos + ((double)j) * ((double)j) * (*(H + i * L + j)) * (*(H + i * L + j));
            }
            uncert = uncert + sqrt(pospos - temp_pos * temp_pos);
        }
    }

    free(H);
    free(w);

    return uncert / (((double)L) * ((double)N_random));
}

int StringToInt(char *a)
{
    int s = 0, len = strlen(a);
    for (int i = 0; i < strlen(a); i++)
    {
        if (int(a[i]) >= 48 && int(a[i]) <= 57)
        {
            s = s * 10;
            s = s + a[i] - 48;
        }
    }

    return s;
}

long int StringToLongInt(char *a)
{
    long int s = 0;
    int len = strlen(a);

    for (int i = 0; i < strlen(a); i++)
    {
        if (int(a[i]) >= 48 && int(a[i]) <= 57)
        {
            s = s * 10;
            s = s + a[i] - 48;
        }
    }

    return s;
}

double StringToDouble(char *a)
{
    double s = 0;
    int len = strlen(a);

    double decimal = 1;

    for (int i = 0; i < strlen(a); i++)
    {
        if (int(a[i]) >= 48 && int(a[i]) <= 57)
        {
            if (decimal == 1)
                s = s * 10.0;

            s = s + (a[i] - 48) * decimal;

            if (decimal != 1)
            {
                decimal = decimal / 10;
            }
        }
        if (a[i] == '.')
        {
            decimal = decimal / 10.0;
        }
    }

    return s;
}

void build_sample(int L, double W, double t, double V,
                  char random_mode, double para,
                  int rank, stack_Cluster *Clusters, double **coupling, double uncert)
{
    double *random = (double *)malloc(sizeof(double) * L);

    Cluster *new_cluster;
    Single_particle *spin_chain;

    memset(random, 0, sizeof(double) * L);

    int i, j;

    SEED_random = SEED_random + rank;

    //exponential random
    if (random_mode == 'E')
    {
        ExpRandom(L, random, W, para);
    }
    else if (random_mode == 'M')
    {
        MultiPointRandom(L, random, W, para);
    }

    //multipoit random
    // multi_point_random(random, W, para_list[para_i]);

    //inverse power random
    //double *h_table, *int_table;
    //int_table = (double *)malloc(sizeof(double) * N_h);
    //h_table = (double *)malloc(sizeof(double) * N_h);
    //memset(int_table, 0, sizeof(int_table));
    //memset(h_table, 0, sizeof(h_table));
    //gen_int_table(int_table, h_table, para_list[para_i], N_h, W);
    //Inv_table_random(random, L, N_h, int_table, h_table);
    //free(h_table);
    //h_table = NULL;
    //free(int_table);
    //int_table = NULL;

    spin_chain = Anderson(L, V, random, *coupling, t, 0, W, uncert, rank);

    for (i = 0; i < L; i++)
    {
        new_cluster = (Cluster *)malloc(sizeof(Cluster));
        memset(new_cluster, 0, sizeof(Cluster));

        new_cluster->bandwidth = (spin_chain + i)->energy;
        new_cluster->level_spacing = (spin_chain + i)->energy;
        new_cluster->m_spins = 1;
        new_cluster->num = i;
        new_cluster->combined = 0;

        new_cluster->spins = (stack_int *)malloc(sizeof(stack_int));

        memset(new_cluster->spins, 0, sizeof(stack_int));

        init_stack_int(new_cluster->spins);

        push_stack_int(i, new_cluster->spins);

        push_stack_Cluster(new_cluster, Clusters);
    }

    free(random);
    random = NULL;

    free(spin_chain);
    spin_chain = NULL;
}

double mismatch(Cluster *a, Cluster *b) //Philipp T. Dumitrescu, et al., PRL, 119, 110604 (2017)
{
    double mis;

    // printf("%lf, %lf\n", a->bandwidth, a->level_spacing);
    // print_stack_int(a->spins);

    // printf("%lf, %lf\n", b->bandwidth, b->level_spacing);
    // print_stack_int(b->spins);

    if ((a->bandwidth == a->level_spacing) && (b->bandwidth == b->level_spacing))
    {
        mis = Abs((a->bandwidth - b->bandwidth));
        return mis;
    }

    if (a->level_spacing >= b->bandwidth)
    {
        mis = Max((a->level_spacing - b->bandwidth), (b->level_spacing));
    }
    else if (b->level_spacing >= a->bandwidth)
    {
        mis = Max((b->level_spacing - a->bandwidth), (a->level_spacing));
    }
    else
    {
        mis = (a->level_spacing) * (b->level_spacing) / Min((a->bandwidth), (b->bandwidth));
    }

    return mis;
}

int combine_cluster(stack_Cluster *merge, Cluster **a, Cluster **b,
                    double *coupling, int num_of_clusters)
{
    // printf("%lf", *(coupling + (*a)->num * num_of_clusters + (*b)->num));
    if (mismatch(*a, *b) < *(coupling + (*a)->num * num_of_clusters + (*b)->num))
    {
        // printf("%d, %lf, %lf, %d\n", (*a)->num, (*a)->bandwidth, (*a)->level_spacing, (*a)->m_spins);
        // print_stack_int((*a)->spins);
        // printf("********************************************\n");

        // printf("%d, %lf, %lf, %d\n", (*b)->num, (*b)->bandwidth, (*b)->level_spacing, (*b)->m_spins);
        // print_stack_int((*b)->spins);
        // printf("********************************************\n");

        // printf("coupling: %lf, mismatch: %lf\n", *(coupling + (*a)->num * num_of_clusters + (*b)->num), mismatch(*a, *b));
        // printf("********************************************\n");

        push_stack_Cluster(*b, merge);

        (*b)->combined = 1;

        return 1;
    }

    return 0;
}

Cluster *Merge_cluster(stack_Cluster *merge, int num, int num_of_clusters, double *coupling)
{
    stack_Cluster *temp, *tempp;
    Cluster *merged_cluster = (Cluster *)malloc(sizeof(Cluster));

    merged_cluster->bandwidth = 0.0;
    merged_cluster->level_spacing = 0.0;
    merged_cluster->m_spins = 0;
    merged_cluster->num = num;
    merged_cluster->combined = 0;
    merged_cluster->spins = (stack_int *)malloc(sizeof(stack_int));

    init_stack_int(merged_cluster->spins);

    temp = merge->next;

    while (temp != NULL)
    {
        merged_cluster->bandwidth = merged_cluster->bandwidth +
                                    pow((temp->a)->bandwidth, 2.0);

        tempp = temp->next;

        while (tempp != NULL)
        {

            merged_cluster->bandwidth = merged_cluster->bandwidth +
                                        pow(*(coupling + (temp->a)->num +
                                              (tempp->a)->num * num_of_clusters),
                                            2.0) *
                                            2.0;

            tempp = tempp->next;
        }

        temp = temp->next;
    }

    merged_cluster->bandwidth = pow(merged_cluster->bandwidth, 0.5);

    temp = merge->next;

    while (temp != NULL)
    {
        merged_cluster->m_spins = merged_cluster->m_spins + (temp->a)->m_spins;
        combine_stack_int(&(merged_cluster->spins), &((temp->a)->spins));

        temp = temp->next;
    }

    if ((merged_cluster->m_spins) < 40)
    {
        merged_cluster->level_spacing = (merged_cluster->bandwidth) /
                                        (pow(2.0, merged_cluster->m_spins) - 1.0);
    }
    else
    {
        merged_cluster->level_spacing = (merged_cluster->bandwidth) *
                                        pow(2.0, -(merged_cluster->m_spins));
    }

    return merged_cluster;
}

void update_coupling(stack_Cluster *Clusters, double **coupling,
                     int num_of_clusters, int num_of_clusters_new,
                     stack_int **merged_m_spin_list, stack_int **merged_num_list)
{

    //Philipp T. Dumitrescu, et al., PRL, 119, 110604 (2017)

    int i, j, n_i, n_j, n_ii = 0, n_jj = 0;
    double **new_coupling;

    stack_Cluster *temp_i, *temp_j;

    stack_int *merge_m_spin_i, *merge_m_spin_j;
    stack_int *merge_m_spin_ii, *merge_m_spin_jj;
    stack_int *merge_num_i, *merge_num_j;
    stack_int *merge_num_ii, *merge_num_jj;

    Cluster *cluster_i, *cluster_j;

    double max_coupling = 0.0;

    new_coupling = (double **)malloc(sizeof(double *));

    *new_coupling = (double *)malloc(sizeof(double) * num_of_clusters_new * num_of_clusters_new);
    memset(*new_coupling, 0, sizeof(double) * num_of_clusters_new * num_of_clusters_new);

    temp_i = Clusters->next;

    for (i = 0; i < num_of_clusters_new; i++)
    {
        temp_j = temp_i->next;

        n_i = (temp_i->a)->m_spins;

        merge_m_spin_i = *(merged_m_spin_list + i);
        merge_num_i = *(merged_num_list + i);

        for (j = i + 1; j < num_of_clusters_new; j++)
        {
            n_j = (temp_j->a)->m_spins;

            merge_m_spin_j = *(merged_m_spin_list + j);
            merge_num_j = *(merged_num_list + j);

            if ((merge_m_spin_i->len == 1) && (merge_m_spin_j->len == 1))
            {
                *(*new_coupling + i + num_of_clusters_new * j) = 0.0;
                *(*new_coupling + i * num_of_clusters_new + j) = 0.0;

                temp_j = temp_j->next;
                continue;
            }

            max_coupling = 0.0;
            n_ii = 0;
            n_jj = 0;

            merge_num_ii = merge_num_i->next;
            merge_m_spin_ii = merge_m_spin_i->next;

            while (merge_num_ii != NULL)
            {
                merge_num_jj = merge_num_j->next;
                merge_m_spin_jj = merge_m_spin_j->next;

                while (merge_num_jj != NULL)
                {
                    if (*(*coupling + *(merge_num_ii->a) +
                          num_of_clusters * (*(merge_num_jj->a))) >= max_coupling)
                    {
                        max_coupling = *(*coupling + *(merge_num_ii->a) +
                                         num_of_clusters * (*(merge_num_jj->a)));

                        n_ii = *(merge_m_spin_ii->a);
                        n_jj = *(merge_m_spin_jj->a);
                    }

                    merge_num_jj = merge_num_jj->next;
                    merge_m_spin_jj = merge_m_spin_jj->next;
                }

                merge_num_ii = merge_num_ii->next;
                merge_m_spin_ii = merge_m_spin_ii->next;
            }

            *(*new_coupling + i + num_of_clusters_new * j) = max_coupling *
                                                             pow(2.0, (double)(-n_i - n_j + n_ii + n_jj) * 0.5);
            *(*new_coupling + i * num_of_clusters_new + j) = *(*new_coupling + i + num_of_clusters_new * j);

            // if ((n_i < n_ii) || (n_j < n_jj))
            // {
            //     printf("haha\n");
            //     printf("n_i:%d, n_j:%d, n_ii:%d, n_jj:%d\n", n_i, n_j, n_ii, n_jj);
            //     // print_stack_int(temp_i->a->spins);
            //     // print_stack_int(temp_j->a->spins);
            //     print_stack_int(merge_m_spin_i);
            //     print_stack_int(merge_m_spin_j);
            // }

            temp_j = temp_j->next;
        }

        temp_i = temp_i->next;
    }

    free(*coupling);
    *coupling = NULL;

    // *coupling = (double *)realloc(*new_coupling,
    //                               sizeof(double) * num_of_clusters_new * num_of_clusters_new);
    *coupling = *new_coupling;

    free(new_coupling);
    new_coupling = NULL;
}

double calculation(int L, stack_Cluster **Clusters, double **coupling)
{
    int num_of_clusters = L;
    int changed = 1, temp_changed;
    int i, j, temp_j;
    int *resonate = NULL;

    stack_Cluster *temp, **merge = NULL, *tempp, *new_Clusters = NULL;
    stack_int **merged_m_spin_list = NULL, **merged_num_list = NULL;

    Cluster *a, *b;

    while (changed != 0)
    {

        new_Clusters = (stack_Cluster *)malloc(sizeof(stack_Cluster));
        init_stack_Cluster(new_Clusters);

        resonate = (int *)malloc(sizeof(int) * num_of_clusters * num_of_clusters);
        memset(resonate, 0, sizeof(int) * num_of_clusters * num_of_clusters);

        changed = 0;
        temp_j = 0;
        temp = (*Clusters)->next;

        temp = (*Clusters)->next;

        while (temp != NULL)
        {
            tempp = temp->next;
            while (tempp != NULL)
            {
                if (mismatch(temp->a, tempp->a) < *(*coupling + (temp->a)->num * num_of_clusters + (tempp->a)->num))
                {
                    *(resonate + (temp->a)->num * num_of_clusters + (tempp->a)->num) = 1;
                    *(resonate + (temp->a)->num + num_of_clusters * (tempp->a)->num) = 1;
                }
                tempp = tempp->next;
            }
            temp = temp->next;
        }

        stack_Cluster *head, *tail;

        temp = (*Clusters)->next;
        while (temp != NULL)
        {
            if ((temp->a)->combined == 1)
            {
                temp = temp->next;
                continue;
            }

            merged_m_spin_list = (stack_int **)realloc(merged_m_spin_list, sizeof(stack_int *) * (temp_j + 1));
            merged_num_list = (stack_int **)realloc(merged_num_list, sizeof(stack_int *) * (temp_j + 1));
            merge = (stack_Cluster **)realloc(merge, sizeof(stack_Cluster *) * (temp_j + 1));

            *(merged_m_spin_list + temp_j) = (stack_int *)malloc(sizeof(stack_int));
            *(merged_num_list + temp_j) = (stack_int *)malloc(sizeof(stack_int));
            *(merge + temp_j) = (stack_Cluster *)malloc(sizeof(stack_Cluster));

            memset(*(merged_m_spin_list + temp_j), 0, sizeof(stack_int));
            memset(*(merged_num_list + temp_j), 0, sizeof(stack_int));
            memset(*(merge + temp_j), 0, sizeof(stack_Cluster));

            init_stack_int(*(merged_m_spin_list + temp_j));
            init_stack_int(*(merged_num_list + temp_j));
            init_stack_Cluster(*(merge + temp_j));

            push_stack_Cluster(temp->a, *(merge + temp_j));
            push_stack_int((temp->a)->m_spins, *(merged_m_spin_list + temp_j));
            push_stack_int((temp->a)->num, *(merged_num_list + temp_j));

            (temp->a)->combined = 1;

            head = (*(merge + temp_j))->next;
            tail = (*(merge + temp_j))->end;

            do
            {
                tempp = (*Clusters)->next;
                while (tempp != NULL)
                {
                    if (*(resonate + (head->a)->num * num_of_clusters + (tempp->a)->num) == 1 &&
                        ((tempp->a)->combined == 0))
                    {
                        push_stack_Cluster(tempp->a, *(merge + temp_j));
                        push_stack_int((tempp->a)->m_spins, *(merged_m_spin_list + temp_j));
                        push_stack_int((tempp->a)->num, *(merged_num_list + temp_j));

                        (tempp->a)->combined = 1;
                        tail = (*(merge + temp_j))->end;
                    }

                    tempp = tempp->next;
                }

                head = head->next;

            } while (head != NULL);

            temp_j = temp_j + 1;
            temp = temp->next;
        }

        for (i = 0; i < temp_j; i++)
        {
            if (changed == 0)
            {
                if ((*(merge + i))->len > 1)
                {
                    changed = 1;
                }
            }

            push_stack_Cluster(Merge_cluster(*(merge + i), i, num_of_clusters, *coupling), new_Clusters);
        }

        // printf("New clusters:\n");
        // print_stack_Cluster(new_Clusters);
        // printf("changed!\n");

        update_coupling(new_Clusters, coupling,
                        num_of_clusters, temp_j,
                        merged_m_spin_list, merged_num_list);

        num_of_clusters = temp_j;

        for (i = 0; i < temp_j; i++)
        {
            destroy_stack_int((merged_m_spin_list + i));
            destroy_stack_int((merged_num_list + i));
            destroy_stack_Cluster((merge + i), 0);
        }

        destroy_stack_Cluster(Clusters, 1);

        *Clusters = new_Clusters;

        // for (i = 0; i < num_of_clusters; i++)
        // {
        //     for (j = 0; j < num_of_clusters; j++)
        //     {
        //         printf("%lf\t", *(*coupling + i + num_of_clusters * j));
        //     }
        //     printf("\n");
        // }

        free(merged_m_spin_list);
        merged_m_spin_list = NULL;

        free(merged_num_list);
        merged_num_list = NULL;

        free(merge);
        merge = NULL;

        free(resonate);
        resonate = NULL;
    }

    int left, right;
    double total = 0;

    // printf("num_of_clusters = %d\n", num_of_clusters);

    stack_int *temp_int;

    // Calculate entropy
    for (i = 0; i < 1; i++)
    {
        temp = (*Clusters)->next;

        while (temp != NULL)
        {
            temp_int = ((temp->a)->spins)->next;

            left = 0;
            right = 0;

            while (temp_int != NULL)
            {

                if (*(temp_int->a) < L / 2)
                {
                    left++;
                }
                else
                {
                    right++;
                }

                temp_int = temp_int->next;
            }
            total = total + (double)(Min(left, right));
            temp = temp->next;
        }
    }

    return total * 2.0 / (double)L;
}

void print_matrix(char *desc, MKL_INT m, MKL_INT n, double *a, MKL_INT lda)
{
    MKL_INT i, j;
    printf("\n %s\n", desc);
    for (i = 0; i < m; i++)
    {
        printf("[");
        for (j = 0; j < n; j++)
            printf(" %6.2f,", a[i + j * lda]);
        printf("],\n");
    }
}

void DSYEVD(int L, double *a, double *w, int print)
{

    MKL_INT n = L, lda = L, info;

    // double w[N];
    // double a[LDA * N] = {
    //     6.39, 0.00, 0.00, 0.00, 0.00,
    //     0.13, 8.37, 0.00, 0.00, 0.00,
    //     -8.23, -4.46, -9.58, 0.00, 0.00,
    //     5.71, -6.10, -9.25, 3.72, 0.00,
    //     -3.18, 7.21, -7.42, 8.54, 2.51};

    // Solve eigenproblem
    info = LAPACKE_dsyevd(LAPACK_COL_MAJOR, 'V', 'U', n, a, lda, w);

    /* Check for convergence */
    if (info > 0)
    {
        printf("The algorithm failed to compute eigenvalues.\n");
        exit(1);
    }

    if (print == 1)
    {
        /* Print eigenvalues */
        print_matrix("Eigenvalues", 1, n, w, 1);
        /* Print eigenvectors */
        print_matrix("Eigenvectors (stored columnwise)", n, n, a, lda);
    }
}

void ExpRandom(int L, double *random, double W, double alpha)
{
    int i;
    double temp = Random();

    for (i = 0; i < L; i++)
    {
        temp = Random();
        if (alpha == 0)
        {
            *(random + i) = (temp * W - W / 2.0) * 2.0;
        }

        else
        {
            if (temp == 0.5)
            {
                *(random + i) = 0;
            }
            else if (temp < 0.5)
            {
                temp = pow(e, alpha) - 2.0 * temp * (pow(e, alpha) - 1.0);
                *(random + i) = -W / alpha * log(temp);
            }
            else
            {
                temp = temp - 0.5;
                temp = 2.0 * temp * (pow(e, alpha) - 1.0) + 1;
                *(random + i) = W / alpha * log(temp);
            }
        }
    }
}

void MultiPointRandom(int L, double *random, double W, double n)
{
    int i, j;
    double temp = Random();

    for (i = 0; i < L; i++)
    {
        temp = Random();
        j = (int)(temp * n);
        *(random + i) = W * 2.0 * (double)j / ((double)n - 1.0) - W;
    }
}

double Random()
{
    int temp;

    temp = Schrage(SEED_random) + b_random;

    if (temp > m_random)
        temp = temp - m_random;

    SEED_random = temp;

    return (double)(temp) / 2147483647.0;
}

long int Schrage(long int z)
{
    long int s;

    s = a_random * (z % q_random) - r_random * (z / q_random);

    if (s > 0)
    {
        return s;
    }
    else
    {
        return s + m_random;
    }
}

void Print_Result(double *RESULT, int num_of_procs)
{
    int i;

    printf("[");
    for (i = 0; i < num_of_procs; i++)
    {
        printf("%.6lf, ", *(RESULT + i));
    }
    printf("]\n\n");
}

void random_pm(double *potential, int N)
{
    int i;
    double temp = Random();

    for (i = 0; i < N; i++)
    {
        temp = Random();

        if (temp > 0.5)
        {
            *(potential + i) = -*(potential + i);
        }
        else
        {
        }
    }
}

void gen_int_table(double *int_table, double *h_table, double alpha, int N_h, double W)
{
    int i;

    double g_0 = sqrt(W), h = 0, g_1 = 0, jifen = 0;
    double dg = (g_1 - g_0) / N_h, g = g_0;

    *(int_table + 0) = 0;

    for (i = 1; i < N_h; i++)
    {
        g = g + dg;
        h = W - g * g;
        jifen = jifen - 2 * pow(Abs(g), 1 - 2 * alpha) * pow(2 * W - g * g, -alpha) * dg;

        *(int_table + i) = jifen;
        *(h_table + i) = h;
    }
}

int Bsearch(double key, double *a, int N_a, int i_min, int i_max)
{
    int i_mid = 0;

    if (key == 0)
    {
        return 0;
    }
    else if (key == 1)
    {
        return N_a - 1;
    }
    else
    {
        if (i_min == i_max || (i_max - i_min == 1))
        {
            return i_min;
        }
        i_mid = (i_min + i_max) >> 1;
        if (Abs(a[i_mid] - key) <= epsilon)
        {
            return i_mid;
        }
        else if (a[i_mid] > key)
        {
            return Bsearch(key, a, N_a, i_min, i_mid);
        }
        else
        {
            return Bsearch(key, a, N_a, i_mid, i_max);
        }
    }
}

void Inv_table_random(double *potential, int N, int N_h, double *int_table, double *h_table)
{
    int i, j;
    double temp = Random();

    for (i = 0; i < N; i++)
    {
        j = Bsearch(temp * (*(int_table + N_h - 1)), int_table, N_h, 0, N_h - 1);
        *(potential + i) = *(h_table + j);
        temp = Random();
    }
}
