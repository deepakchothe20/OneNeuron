from utility.model import Perceptron
import pandas as pd
from utility.utility import prepare_data,save_model,save_plot

OR = {
    "x1": [0,0,1,1],
    "x2": [0,1,0,1],
    "y": [0,1,1,1],
}

df = pd.DataFrame(OR)

df

X,y = prepare_data(df)

ETA = 0.3 # 0 and 1
EPOCHS = 10

model = Perceptron(eta=ETA, epochs=EPOCHS)
model.fit(X, y)

_ = model.total_loss()
save_model(model,filename='or.model')
save_plot(df,'or.png',model)