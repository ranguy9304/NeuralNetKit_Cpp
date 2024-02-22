
#include "model.hpp"

namespace model
{

    double lossFunction(Eigen::MatrixXd y, Eigen::MatrixXd yHat)
    {
        // Compute the mean squared loss function
        return (y - yHat).array().square().mean();
    }

    Eigen::MatrixXd ActivationFunction::sigmoid(Eigen::MatrixXd x)
    {
        return 1 / (1 + exp(-1 * (x.array())));
    }
    Eigen::MatrixXd ActivationFunction::relu(Eigen::MatrixXd x)
    {
        return x.cwiseMax(0);
    }
    Eigen::MatrixXd ActivationFunction::softmax(Eigen::MatrixXd x)
    {
        return exp(x.array() - x.maxCoeff()).matrix() / exp(x.array() - x.maxCoeff()).sum();
    }

    Layer::Layer(int in_nodes, int out_nodes, bool randomize = true)
    {
        if (randomize)
        {
            random_device rd;
            mt19937 gen(rd()); // here you could set the seed, but random_device already does that
            uniform_real_distribution<float> dis(-0.1, 0.1);

            this->weights = Eigen::MatrixXd::NullaryExpr(in_nodes, out_nodes, [&]()
                                                         { return dis(gen); });
            // multiply all elements in this->weights by 10
            // this->weights *= 10;
            // this->weights = Eigen::MatrixXd::Random(in_nodes, out_nodes);

            // this->baises = Eigen::VectorXd::Random(out_nodes);
            this->baises = Eigen::VectorXd::Zero(out_nodes); // Use Zero() for efficiency

            // Generate random values
            for (int i = 0; i < out_nodes; ++i)
            {
                this->baises(i) = dis(gen);
            }
            this->baises *= 10;
        }
        else
        {
            // Initialize weight and bais with different values if not randomizing
        }
    }

    void Layer::setWeights(Eigen::MatrixXd weights)
    {
        this->weights = weights;
    }
    void Layer::setBaises(Eigen::VectorXd baises)
    {
        this->baises = baises;
    }
    void Layer::setOutputs(Eigen::MatrixXd outputs)
    {
        this->outputs = outputs;
    }
    void Layer::setActivation(Eigen::MatrixXd (*function)(Eigen::MatrixXd))
    {
        this->activation = function;
    }

    void Model::setInput(int inNodes)
    {
        this->inNodes = inNodes;
        this->nodes.push_back(inNodes);
    }
    void Model::addLayer(int noNodes, Eigen::MatrixXd (*activationFunc)(Eigen::MatrixXd))
    {
        if (this->nodes.size() == 0)
        {
            cout << "define input first" << endl;
            exit(EXIT_FAILURE);
        }
        if (this->outNodes != 0)
        {
            cout << "Out layers already defined cant add more" << endl;
            exit(EXIT_FAILURE);
        }
        this->nodes.push_back(noNodes);
        this->activations.push_back(activationFunc);
    }
    void Model::setOutput(int outNodes, Eigen::MatrixXd (*activationFunc)(Eigen::MatrixXd))
    {
        this->outNodes = inNodes;
        this->nodes.push_back(outNodes);
        this->activations.push_back(activationFunc);
    }

    void Model::compile()
    {
        if (this->nodes.size() <= 2)
        {
            cout << "no layers added" << endl;
            exit(EXIT_FAILURE);
        }
        for (int i = 1; i < this->nodes.size(); i++)
        {

            // Eigen::MatrixXd weight(this->nodes[i - 1], this->nodes[i]);
            // weight.setRandom();
            // Eigen::VectorXd bais(this->nodes[i]);
            // bais.setRandom();
            Layer l(this->nodes[i - 1], this->nodes[i]);
            // l.setWeights(weight);
            // l.setBaises(bais);
            // cout << l.baises << "\n\n";

            l.setActivation(this->activations[i - 1]);
            this->layers.push_back(l);
        }
    }

    void Model::forwardSingle(Eigen::MatrixXd input)
    {
        // cout << input << endl;
        // this->layers[0].setOutputs(input);
        Eigen::MatrixXd temp = input;
        // cout << "helo" << endl;
        for (int i = 0; i < this->layers.size(); i++)
        {
            // cout << this->layers[i].outputs << "\n-----------\n" << this->layers[i].weights;
            // cout << "\n%%%%%\n" << this->layers[i].outputs * this->layers[i].weights <<"\n11111111111111111111111\n"<< endl;
            this->layers[i].setOutputs(this->layers[i].activation(temp * this->layers[i].weights + this->layers[i].baises.transpose()));
            temp = this->layers[i].outputs;
            // input.transpose();
        }
        // cout << "nt g" << endl;
    }

    void Model::backwardSingle(Eigen::MatrixXd outputs, Eigen::MatrixXd expected)
    {
        // Learning rate
        double learningRate = 0.01;
        // cout << outputs << endl;
        // cout << "----" << endl;
        // cout << expected << endl;
        // Compute error at output layer
        Eigen::MatrixXd error = outputs - expected;
        // cout << "----\n"<< endl;
        // cout << error << endl;
        // cout << "----\n"<< endl;

        for (int i = this->layers.size() - 1; i >= 0; i--)
        {
            // Compute derivative of the activation function
            Eigen::MatrixXd derivative;
            if (this->activations[i] == ActivationFunction::sigmoid)
            {
                // For sigmoid, derivative is output * (1 - output)
                derivative = this->layers[i].outputs.array() * (1 - this->layers[i].outputs.array());
            }
            else if (this->activations[i] == ActivationFunction::relu)
            {
                // For ReLU, derivative is 1 for positive values, 0 otherwise
                derivative = (this->layers[i].outputs.array() > 0).cast<double>();
            }
            else if (this->activations[i] == ActivationFunction::softmax)
            {
                // For softmax, derivative is 1 for positive values, 0 otherwise
                derivative = this->layers[i].outputs.array() * (1 - this->layers[i].outputs.array());
            }
            

            // print the derivatives for debugging
            // cout << derivative << endl;
            // Other activation functions can be added here

            // Calculate gradient
            Eigen::MatrixXd gradient = error.cwiseProduct(derivative);
            // cout << "-----------GRAD ----------\n"<< gradient << "\n----------"<< endl;
            // // Calculate deltas for weights and biases
            Eigen::MatrixXd deltaWeights;
            Eigen::MatrixXd deltaBiases;
            if (i > 0)
            {
                deltaWeights = this->layers[i - 1].outputs.transpose() * gradient;
                deltaBiases = gradient.colwise().sum();
                // cout << "-----------delta biase ----------\n"<< deltaBiases << "\n----------"<< endl;
                this->layers[i].weights -= learningRate * deltaWeights;
                this->layers[i].baises -= learningRate * deltaBiases.colwise().mean();
                error = gradient * this->layers[i].weights.transpose();
            }
            // else
            // {
            //     deltaWeights = inputs.transpose() * gradient; // 'inputs' should be stored from the forward pass
            //     deltaBiases = gradient.colwise().sum();
            // }

            // Update weights and biases
            // this->layers[i].weights -= learningRate * deltaWeights;
            // this->layers[i].baises -= learningRate * deltaBiases.rowwise().mean();

            // Update error for next layer
            // if (i > 0)
            // {
            //     error = gradient * this->layers[i].weights.transpose();
            // }
        }
    }
}