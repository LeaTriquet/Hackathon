import os
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def train(datapath: str = os.path.join('data', 'xgboost', 'data.csv'),
          weigth_path: str = os.path.join('src', 'xgboost', 'xgweigth.model'),
          save_model: bool = True,
          plot_features_importance: bool = True
          ) -> None:
    
    data = pd.read_csv(datapath)

    X = data[['similarity', 'rouge1', 'rouge2', 'rougeL']]
    y = data['godmetrics']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialiser le modèle XGBoost
    xg_reg = xgb.XGBRegressor(objective ='reg:squarederror',
                              colsample_bytree = 0.5,
                              learning_rate = 0.1,
                              max_depth = 10,
                              alpha = 10,
                              n_estimators = 200)
    
    print('start training')
    xg_reg.fit(X_train, y_train)
    print('end training')

    if save_model:
        xg_reg.save_model(weigth_path)
    
    if plot_features_importance:
        xgb.plot_importance(xg_reg, max_num_features=4, importance_type='weight')
        plt.show()

    y_pred = xg_reg.predict(X_train)
    mse_train = mean_squared_error(y_train, y_pred)

    y_pred = xg_reg.predict(X_test)
    mse_test = mean_squared_error(y_test, y_pred)

    print(f"Mean Squared Error: TRAIN: {mse_train:.5f} TEST: {mse_test:.5f}")
    # Results: Mean Squared Error: TRAIN: 0.00959 TEST: 0.00958


if __name__ == '__main__':
    train(save_model=True, plot_features_importance=True)
