from utility.model import Perceptron
import pandas as pd
from utility.utility import prepare_data,save_model,save_plot
import logging
import os
logging_str = "[%(asctime)s: %(levelname)s: %(module)s] %(message)s"
log_dir='logs'
os.makedirs(log_dir,exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir,"running_logs.log"),level=logging.INFO,format=logging_str)

def main():
    """
    """
    AND = {
    "x1": [0,0,1,1],
    "x2": [0,1,0,1],
    "y": [0,0,0,1],
    }

    df = pd.DataFrame(AND)

    df

    X,y = prepare_data(df)

    ETA = 0.3 # 0 and 1
    EPOCHS = 10

    model = Perceptron(eta=ETA, epochs=EPOCHS)
    model.fit(X, y)

    _ = model.total_loss()
    save_model(model,filename='and.model')
    save_plot(df,'and.png',model)
    logging.info('ABC')

if __name__=='__main__':
    main()