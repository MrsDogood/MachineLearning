# HessianFree
I'm trying to understand NeuralNetworks and particularly HF optimization, so I'm creating a repo to experiment in.

## Building
This project depends on Maven and Make and only builds in Linux at the moment.
Run the `build.sh` script.

## Simple Example
You'll find a simple, useless example for this project in 
`src/main/java/com/github/mrsdogood/example/SimpleExample.java`.
I know that's a long file path; I hate maven conventions too.
The example can be compiled and executed with the script `runExample.sh`.

## MNIST Training Example
I'm just learning how to recognize digits using these things.
The script `runMNISTTraining.sh` run without arguments will launch the example and create
a file `dat/mnist_0.dat` that will hold training records and a neural network for recognizing
the handwriten digit 0. The script takes the following parameters:

    Usage: runMNISTTraining.sh <nn_file> <digit> \
        (<training_iterations>) (<learning_rate>) (<negative_sample_rate>) \
        (<rand_seed>)
    nn_file is the file to store the neural net weights in
    digit is an integer 0-9
    training_iterations default is 100
    learning_rate default is 1.0
    negative_sample_rate default is 0.2
    rand_seed will default to a randomly generated value
