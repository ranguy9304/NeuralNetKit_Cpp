#ifndef MODEL_H // Include guards to prevent multiple inclusion

#define MODEL_H

#include <stdio.h>
#include <string>
#include <thread>
#include <Eigen/Dense>
#include <iostream>
#include <cmath>
#include <random>
using namespace std;

namespace model
{

    double lossFunction(Eigen::MatrixXd y, Eigen::MatrixXd yHat);

    class ActivationFunction
    {
    public:
        static Eigen::MatrixXd sigmoid(Eigen::MatrixXd x);
        static Eigen::MatrixXd relu(Eigen::MatrixXd x);
        static Eigen::MatrixXd softmax(Eigen::MatrixXd x);
    };

    class Layer
    {
    public:
        
        Eigen::MatrixXd weights;
        Eigen::VectorXd baises;
        Eigen::MatrixXd outputs;
        Layer(int in_nodes, int out_nodes, bool randomize);
        Eigen::MatrixXd (*activation)(Eigen::MatrixXd);
        void setWeights(Eigen::MatrixXd weights);
        void setBaises(Eigen::VectorXd baises);
        void setActivation(Eigen::MatrixXd (*function)(Eigen::MatrixXd));
        void setOutputs(Eigen::MatrixXd outputs);
    };
    class Model
    {
    public:
        // vector<Eigen::MatrixXd> weights;
        // vector<Eigen::VectorXd> baises;
        // vector<Eigen::MatrixXd> outputs;
        vector<Layer> layers;
        vector<int> nodes;
        vector<Eigen::MatrixXd (*)(Eigen::MatrixXd)> activations;

        // ActivationFunction activation;

        int inNodes;
        int outNodes = 0;

        void setInput(int inNodes);
        void addLayer(int noNodes, Eigen::MatrixXd (*activationFunc)(Eigen::MatrixXd));
        void setOutput(int outNodes, Eigen::MatrixXd (*activationFunc)(Eigen::MatrixXd));
        void compile();
        void forwardSingle(Eigen::MatrixXd input);
        void backwardSingle(Eigen::MatrixXd outputs, Eigen::MatrixXd expected);
    };

}

#endif