#option to choose activation function between relu and tanh (default: relu)
def choose_activation(string):
    if string == 'relu':
        return nn.ReLU()
    elif string == 'tanh':
        return nn.Tanh()

#MLP Model
class Classifier(nn.Module):
    def __init__(self, activation_func='relu'):
        super(Classifier, self).__init__()
        self.activation_func = choose_activation(activation_func)
        self.mlp = nn.Sequential(
            #MNIST Input shape of each MNIST image is [1, 28, 28]
            nn.Dropout(0.4),
            nn.Linear(1*28*28, 500),
            self.activation_func,
            nn.BatchNorm1d(500),     #apply batch norm to improve training

            nn.Dropout(0.4),
            nn.Linear(500, 200),
            self.activation_func,
            nn.BatchNorm1d(200),

            nn.Linear(200, 10), 
            nn.Sigmoid()    #squish between 0 and 1. 0 being not classified and 1 being classified
        )
    def forward(self, input):
        #squish 3D tensor of image to "1D"
        input = input.view(input.size(0), -1)
        input = self.mlp(input) # pass through MLP
        return input 
