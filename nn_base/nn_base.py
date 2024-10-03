import numpy as np
import copy


class NeuralNetworkModel:
    def __init__(self) -> None:
        self.layers = []
        self.Ws = []
        self.Bs = []
        self.dWs = []
        self.dBs = []
        self.VdWs = []
        self.VdBs = []
        self.cost_history = []


    def sequential(self, layers: list) -> None:
        """
        Takes a list of layers and stores them within the object.
        
        Example:
        self.sequential([self.layer(64, "relu"), self.layer(32, "relu"), self.layer(10, "softmax")])
        
        """
        
        self.layers = layers


    def layer(self, units: int, activation: str) -> tuple[int, str]:
        """
        Defines a layer with a specified number of units and an activation function.
        
        Example:
        self.layer(64, "relu") -> (64, "relu")

        """

        return units, activation.lower()


    def initialize_params(self, num_features: int, layers: list) -> None:
        """
        Initializes weights and biases for a neural network using He initialization.
        
        This method initializes:
        - Weights (Ws): Randomly initialized using He init to help prevent vanishing/exploding gradients.
        - Biases (Bs): Initialized as zeros.
        - Weight gradients (dWs): Initialized as zeros, matching the shape of the weights.
        - Bias gradients (dBs): Initialized as zeros, matching the shape of the biases.
        - Weight velocities (VdWs): Initialized as zeros, used for momentum, \
            matching the shape of the weights.
        - Bias velocities (VdBs): Initialized as zeros, used for momentum, \
            matching the shape of the biases.

        """

        n, k = num_features, layers[0][0]

        for i in range(layers.__len__()):
            self.Ws.append(np.random.randn(n, k) * np.sqrt(2 / n))            
            self.Bs.append(np.zeros((1, k)))
            
            self.dWs.append(np.zeros((n, k)))
            self.dBs.append(np.zeros((1, k)))

            self.VdWs.append(np.zeros((n, k)))
            self.VdBs.append(np.zeros((1, k)))

            if i+1 != layers.__len__():
                n, k = layers[i][0], layers[i+1][0]

   
    def forward_prop(self, X: np.ndarray, layers: list) -> tuple[list, list, np.ndarray]:
        """
        Performs the forward propagation step through the network layers.

        This method:
        - Computes the linear transformation Z = data @ self.Ws[i] + self.Bs[i] for each layer.
        - Applies ReLU activation function to intermediate results.
        - Appends the computed Z values (pre-activations) to the list `Zs`.
        - Appends the activation values (ReLU outputs) to the list `As` for intermediate layers.
        - For the output layer:
            - Applies softmax activation if the layer's activation function is "softmax", \
                which includes normalization for numerical stability.
            - Applies sigmoid activation if the layer's activation function is "sigmoid".

        """
        
        Zs = []
        As = []

        data = X.copy()
        for i in range(layers.__len__()):
            Z = data @ self.Ws[i] + self.Bs[i]
            Zs.append(Z)
            
            if i+1 != layers.__len__():
                data = np.maximum(0, Z)
                As.append(data)

        if layers[-1][1] == "softmax":
            Z_shifted = Z - np.max(Z, axis=1, keepdims=True)
            exp_Z = np.exp(Z_shifted)
            data = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

        else:
            data = 1 / (1 + np.exp(-Z))
        

        return Zs, As, data
    

    def compute_cost(self, Y: np.ndarray, Y_hat: np.ndarray, Y_one_hot: np.ndarray) -> np.float64:
        """
        Computes the cost for the given predictions and true labels based on the output layer activation.

        This method:
        - Computes the cost based on the activation function of the last layer:
            - **Sigmoid activation**: For binary or multi-label classification, \
                                      applies binary cross-entropy cost.
            - **Softmax activation**: For multi-class classification, \
                                      applies categorical cross-entropy cost.

        """
        
        cost = - np.sum(Y * np.log(Y_hat + 1e-10) + (1 - Y) * np.log(1 - Y_hat + 1e-10)) \
            if Y_one_hot is None else - np.sum(Y_one_hot * np.log(Y_hat + 1e-10))

        return cost


    def backward_prop(self, dWs_: list, dBs_: list, X: np.ndarray, Y: np.ndarray, \
                      Zs: list, As: list, Y_hat: np.ndarray, Y_one_hot: np.ndarray) -> None:
        """
        Performs the backward propagation step through the network to compute gradients.

        This method:
        - Computes the error term for the output layer based on the activation function used.
        - Updates the weight gradients (dWs_) and bias gradients (dBs_) using the chain rule \
            of differentiation.
        - Uses the ReLU derivative to propagate the error backward through the network.

        """
            
        error = Y_hat - Y if Y_one_hot is None else Y_hat - Y_one_hot
        
        for i in range(dWs_.__len__() - 1, -1, -1):
            if i != 0:
                dWs_[i] += As[i-1].T @ error
                dBs_[i] += np.sum(error, axis=0, keepdims=True)
            
            else:
                dWs_[i] += X.T @ error
                dBs_[i] += np.sum(error, axis=0, keepdims=True)
                break
            
            error = error @ self.Ws[i].T * np.where((Zs[i-1] >= 0), 1, 0)


    def fit(self, X: np.ndarray, Y: np.ndarray, batch_size: int=1000, epochs: int=1000, \
            alpha: float=0.1) -> None:
        """
        Trains the neural network model.

        This method:
        - Initializes the network parameters (weights and biases), their gradients and velocities.
        - Iterates over the specified number of epochs.
        - Processes the data in mini-batches, if the user specifies a samaller batch size.
        - Computes forward propagation.
        - Computes the cost for the current batch and accumulates it.
        - Performs backward propagation to compute gradients and accumulates it.
        - Prints the average cost for each epoch.
        - Updates weights and biases for each epoch

        """
        
        if self.Ws == []:
            self.initialize_params(X.shape[1], self.layers)
        
        cost_history = self.cost_history.__len__()
        m = Y.shape[0]

        for epoch in range(epochs):
            print(f"Epoch: {epoch+1}/{epochs}")

            total_cost = 0.0
            dWs_ = copy.deepcopy(self.dWs)
            dBs_ = copy.deepcopy(self.dBs)
            
            for i in range(0, X.shape[0], batch_size):
                batch_X = X[i:i+batch_size]
                batch_Y = Y[i:i+batch_size]

                Zs, As, Y_hat = self.forward_prop(batch_X, self.layers)

                Y_one_hot = np.eye(len(np.unique(Y)))[batch_Y.flatten()] \
                    if self.layers[-1][1] == "softmax" else None
                
                total_cost += self.compute_cost(batch_Y, Y_hat, Y_one_hot)

                self.backward_prop(dWs_, dBs_, batch_X, batch_Y, Zs, As, Y_hat, Y_one_hot)
            
            cost = total_cost / m
            self.cost_history.append((cost_history + (epoch+1), cost))
            print(f"cost = {cost}")

            for i in range(dWs_.__len__()):
                self.VdWs[i] = 0.9 * self.VdWs[i] + (1 - 0.9) * (dWs_[i] / m)
                self.VdBs[i] = 0.9 * self.VdBs[i] + (1 - 0.9) * (dBs_[i] / m)
                
                self.Ws[i] -= alpha * self.VdWs[i]
                self.Bs[i] -= alpha * self.VdBs[i]
                

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        This method performs predictions using the forward propagation process.

        """
        
        _, _, Y_hat = self.forward_prop(X, self.layers)

        return Y_hat
    