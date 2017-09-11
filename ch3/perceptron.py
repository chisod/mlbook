import numpy as np

#I assume weights are a normal (pre-multiplier) of the nodes which switches the w_ji to w_ij
#compared to the textbook I'm using
class Perceptron:
	def __init__(self, num_inputs, num_outputs, eta, bias=True):
		self.num_inputs = num_inputs
		self.num_outputs = num_outputs
		self.eta = eta
		self.bias = bias

		self.weights = np.random.rand(num_inputs+1, num_outputs)*0.1-0.5

	def train(self, inputs_array, outputs_array, iterations):
		len_inputs = np.shape(inputs_array)[0]
		if self.bias:
			inputs_array = np.concatenate((inputs_array,-np.ones((len_inputs,1))),axis=1)
		
		for j in range(len(inputs_array)):
			inputs = np.transpose([inputs.array[j]])
			targets = np.transpose([outputs.array[j]])
			activations = self._compute_activations(inputs)
			self.weights -= self.eta*np.transpose(np.dot(inputs, np.transpose(activations-targets)))

	def _compute_activations(self, inputs):
		activations = np.dot(self.weights, inputs)
		on_nodes = np.where(activations>0,1,0)
		return on_nodes

