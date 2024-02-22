SimpleNNCpp - Simplified Neural Network Library in C++

Welcome to SimpleNNCpp, a streamlined C++ library designed for building basic neural networks. This library is tailored for educational purposes, offering an approachable introduction to neural network concepts and implementations in C++.
Overview

SimpleNNCpp is a lightweight library that provides fundamental components to construct and operate neural networks. Its simplicity and minimalistic design make it ideal for learning and small-scale projects.
Features

    Customizable Neural Network Layers: Easily define and manage layers of a neural network.
    Basic Activation Functions: Includes common activation functions like Sigmoid, ReLU, and Softmax.
    Loss Function: Integrated loss function to evaluate network performance.
    Eigen Library Integration: Leverages the Eigen library for efficient matrix operations.

Model Structure

The core of SimpleNNCpp is the model namespace, which contains the essential components to build neural networks:

    ActivationFunction: Static methods for Sigmoid, ReLU, and Softmax functions.
    Layer: Represents a single layer in a neural network, with customizable weights, biases, and activation functions.
    Model: The neural network model, allowing the addition of layers and setting input/output nodes.

Getting Started

    Clone the Repository: Get the latest version of SimpleNNCpp.
    Prerequisites: Ensure you have the Eigen library set up in your environment.
    Explore Examples: Check out the provided examples to understand the library usage.

Usage

    Define a Model: Instantiate a model::Model object.
    Set Input Layer: Define the input nodes using setInput.
    Add Layers: Add layers with addLayer, specifying the number of nodes and activation function.
    Set Output Layer: Define the output layer with setOutput.
    Compile the Model: Prepare the model for training with compile.
    Train the Model: Use forwardSingle and backwardSingle for training on data.

Contributions

Contributions are welcome! Whether it's bug fixes, improvements, or documentation - feel free to fork the repository and submit pull requests.
License

This project is licensed under the MIT License - see the LICENSE file for details.
