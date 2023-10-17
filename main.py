import numpy as np
class XarxaNeuronal:
    # crear classe de la xarxa amb els elements
    def _init_(self, inputs): #constructor
        self.inputs = 2 #creem dos inputs
        self.hidden = 3 #creem tres elements hidden
        self.outputs = 2 #creem dos outputs
        #np.ones --> fa una matriu d'uns(d'aquestes dimensions)
        self.weights_inputs = np.ones ((self.inputs, self.hidden)) #com volem una matriu hi ha dos termes
        self.weights_outputs = np.ones ((self.hidden, self.outputs)) #per fer una matriu posem dos (())
        self.bias_hidden = np.ones(self.hidden) #declarar les bies ocultes
        self.bias_outputs = np.ones(self.outputs) #declarar les bies outputs

    def sigmoid (self, x): #funció sigmoid
        return 1/(1+np.exp(-x)) #fòrmula
    def feedforward(self, inputs):
        #np.dot --> per multiplicar dos matrius
        hidden_input = np.dot(inputs, self.weights_inputs) + self.bias_hidden # multipliquem les dos matrius i sumem la matriu bias
        hidden_output = self.sigmoid (hidden_input) # fa la funció sigmoid amb el càlcul de matrius
        output_input = np.dot(hidden_output, self.weights_outputs)+self.bias_outputs # mateixa operació amb l'altra layer (operació encadenada amb l'altra)
        output = self.sigmoid(output_input) # aplica la fòrmula amb l'última matriu calculada
        return output # retorna la sortida de la xarxa
    def train(self, inputs, targets, epochs):
        for epoch in range(epochs):
            # Feedforward
            hidden_input = np.dot(inputs, self.weights_inputs) + self.bias_hidden
            hidden_output = self.sigmoid(hidden_input)
            output_input = np.dot(hidden_output, self.weights_outputs) + self.bias_outputs
            output = self.sigmoid(output_input)

            # Càlcul de l'error
            error = targets - output

            # Retropropagació
            output_delta = error * self.sigmoid_derivative(output)
            hidden_error = output_delta.dot(self.weights_outputs.T)
            hidden_delta = hidden_error * self.sigmoid_derivative(hidden_output)

            # Actualització de pesos i bies
            self.weights_outputs += hidden_output.T.dot(output_delta)
            self.weights_inputs += inputs.T.dot(hidden_delta)
            self.bias_outputs += np.sum(output_delta)
            self.bias_hidden += np.sum(hidden_delta)

            # Càlcul de l'error mitjà
            mean_error = np.mean(np.abs(error))
            print(f'Epoch: {epoch}, Error Mitjà: {mean_error}')
inputs = np.array([0, 1]) #assignem valors als inputs
xarxa = XarxaNeuronal(inputs) #creem la instància de la classe amb els valor dels inputs
resultat = xarxa.feedforward(inputs) #calculem el resultat
print(resultat) #mostrem el resultat de la sortida
